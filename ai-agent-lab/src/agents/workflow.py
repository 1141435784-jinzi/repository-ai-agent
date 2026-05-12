"""
=== 企业级 Agent 工作流引擎 ===

【核心职责】：
1. 定义符合 OpenAI API 规范的状态机（AgentState）
2. 实现企业级工具调用流程（本地工具、API工具、MCP工具）
3. 构建支持工具调用循环完整性的工作流
4. 提供异步 Agent 执行接口
5. 集成 RAG 功能

【设计原则】：
1. 合规性：严格遵循 LangGraph 工具调用规范
2. 可靠性：确保工具调用循环完整性
3. 可扩展性：支持动态工具注册和调用
4. 可观测性：完整的执行跟踪和错误处理
"""

from typing import Annotated, List, Dict, Any, Optional, Literal
import asyncio
import logging
import uuid
import json

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from src.config import (
    MAX_ITERATIONS,
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_BASE_TRAVEL_DIR,
)
from src.memory import get_async_checkpointer, get_memory_manager
from src.llm.service import get_llm
from src.prompts import AGENT_PROMPT, SUPERVISOR_PROMPT, TRAVEL_TECH_PROMPT
from src.rag.engine import RAGEngine
from src.tools.tool_manager import tool_manager
from src.utils.logger import WorkflowLogger

logger = logging.getLogger(__name__)
workflow_logger = WorkflowLogger(logger)


# ============================================================
# 全局工具锁：防止并发重复编译
# ============================================================
_agent_lock = asyncio.Lock()
_async_agent = None


# ============================================================
# State Reducer：LangGraph 强制要求，解决状态覆盖问题
# ============================================================
def reduce_str(prev: str, next: Optional[str]) -> str:
    return next if next is not None else prev

def reduce_list(prev: list, next: Optional[list]) -> list:
    return next if next is not None else prev


# ============================================================
# 第一步：状态定义（完全符合 LangGraph 规范）
# ============================================================
class AgentState(TypedDict):
    """多 Agent 状态定义"""
    # 对话消息
    messages: Annotated[list, add_messages]
    trimmed_messages: Annotated[list, reduce_list]

    # 记忆
    memory_context: Annotated[str, reduce_str]

    # 路由
    route: Annotated[str, reduce_str]

    # RAG
    rag_context: Annotated[str, reduce_str]
    rag_sources: Annotated[list, reduce_list]


# ============================================================
# 第二步：初始化 RAG 引擎
# ============================================================
print("📚 正在初始化 Agent 技术知识库...")
agent_tech_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_DIR,
    collection_name="agent_knowledge",
)
print("📚 Agent 技术知识库初始化完成 ✅")

print("✈️ 正在初始化旅游知识库...")
travel_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_TRAVEL_DIR,
    collection_name="travel_knowledge",
)
print("✈️ 旅游知识库初始化完成 ✅")


# ============================================================
# 第三步：节点实现
# ============================================================

def memory_node(state: AgentState, config: RunnableConfig) -> dict:
    """记忆管理节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    last_message = state["messages"][-1]
    current_query = last_message.content if hasattr(last_message, "content") else ""
    
    workflow_logger.node_enter("memory", thread_id)
    
    memory_manager = get_memory_manager()
    result = memory_manager.process_memory(
        messages=state["messages"],
        thread_id=thread_id,
        current_query=current_query,
    )
    
    workflow_logger.memory_update(thread_id, len(result["memory_context"]) if result["memory_context"] else 0)
    workflow_logger.node_exit("memory", thread_id, "记忆处理完成")

    return {
        "memory_context": result["memory_context"],
        "trimmed_messages": result["trimmed_messages"],
    }


def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """意图路由节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("supervisor", thread_id)
    
    try:
        last_message = state["messages"][-1]
        user_text = last_message.content if hasattr(last_message, "content") else ""
        user_model = config.get("configurable", {}).get("model", "")

        supervisor_llm = get_llm(temperature=0, provider=user_model or None, streaming=False)
        workflow_logger.llm_call(thread_id, "supervisor", model=user_model)
        
        prompt_messages = SUPERVISOR_PROMPT.invoke({"messages": [last_message]})
        response = supervisor_llm.invoke(prompt_messages)
        workflow_logger.llm_response(thread_id, len(response.content) if response.content else 0)
        
        route = response.content.strip().lower()

        if "travel" in route or "旅游" in route:
            route = "travel"
        else:
            route = "agent_tech"

        workflow_logger.route_decision(thread_id, route)
        workflow_logger.node_exit("supervisor", thread_id, route)
        return {"route": route}

    except Exception as e:
        workflow_logger.error(thread_id, "supervisor", e)
        return {"route": "agent_tech"}


def route_by_supervisor(state: AgentState) -> str:
    return state.get("route", "agent_tech")


def route_back_to_agent(state: AgentState) -> str:
    """工具执行完，回到当前 Agent"""
    return state["route"]


# ------------------------------
# RAG 节点
# ------------------------------
def agent_tech_rag_node(state: AgentState) -> dict:
    thread_id = str(uuid.uuid4())[:8]  # 生成临时线程ID用于日志
    workflow_logger.node_enter("agent_tech_rag", thread_id)
    
    try:
        question = state["messages"][-1].content
        workflow_logger.rag_query(thread_id, question, "agent_tech_rag")
        
        res = agent_tech_rag.query(question)
        workflow_logger.rag_result(thread_id, "agent_tech_rag", res["sources"], len(res["answer_context"]))
        
        workflow_logger.node_exit("agent_tech_rag", thread_id, f"检索到 {len(res['sources'])} 个来源")
        return {"rag_context": res["answer_context"], "rag_sources": res["sources"]}
    except Exception as e:
        workflow_logger.error(thread_id, "agent_tech_rag", e)
        return {"rag_context": "", "rag_sources": []}

def travel_rag_node(state: AgentState) -> dict:
    thread_id = str(uuid.uuid4())[:8]
    workflow_logger.node_enter("travel_rag", thread_id)
    
    try:
        question = state["messages"][-1].content
        workflow_logger.rag_query(thread_id, question, "travel_rag")
        
        res = travel_rag.query(question)
        workflow_logger.rag_result(thread_id, "travel_rag", res["sources"], len(res["answer_context"]))
        
        workflow_logger.node_exit("travel_rag", thread_id, f"检索到 {len(res['sources'])} 个来源")
        return {"rag_context": res["answer_context"], "rag_sources": res["sources"]}
    except Exception as e:
        workflow_logger.error(thread_id, "travel_rag", e)
        return {"rag_context": "", "rag_sources": []}


# ------------------------------
# 提示词构建
# ------------------------------
def _build_prompt_with_context(state: AgentState, prompt_template, messages: list) -> list:
    prompt_messages = prompt_template.invoke({"messages": messages})

    # 注入记忆
    memory_ctx = state.get("memory_context")
    if memory_ctx:
        sys_msg = prompt_messages.messages[0]
        prompt_messages.messages[0] = SystemMessage(
            content=f"{sys_msg.content}\n\n## 历史上下文\n{memory_ctx}"
        )

    # 注入 RAG
    rag_ctx = state.get("rag_context")
    sources = state.get("rag_sources", [])
    if rag_ctx:
        src_str = "、".join(sources)
        rag_msg = SystemMessage(
            f"参考资料（来源：{src_str}）：\n{rag_ctx}\n\n"
            "如无相关资料请直接回答，不要编造。"
        )
        prompt_messages.messages.insert(-1, rag_msg)

    return prompt_messages.messages


# ------------------------------
# Agent 核心回复逻辑（修复工具调用问题）
# ------------------------------
async def _build_agent_response(
    state: AgentState,
    config: RunnableConfig,
    prompt_template
) -> dict:
    """构建 Agent 响应（支持工具调用）"""
    thread_id = config["configurable"].get("thread_id", "default")
    user_model = config["configurable"].get("model", "")
    
    workflow_logger.node_enter("agent_response", thread_id)
    
    try:
        cleaned_msgs = state["messages"]

        # 构建提示
        prompt_msgs = _build_prompt_with_context(state, prompt_template, cleaned_msgs)

        # 获取工具列表
        tools = await tool_manager.get_tools()
        
        # LLM + 工具绑定（启用流式输出以支持前端流式显示）
        llm = get_llm(provider=user_model, streaming=True)
        workflow_logger.llm_call(thread_id, user_model or "default", 
                                model=llm.model_name if hasattr(llm, 'model_name') else None,
                                prompt_tokens=sum(len(str(m.content)) for m in prompt_msgs))
        
        if tools:
            llm_with_tools = llm.bind_tools(tools)
            workflow_logger.tool_execution(thread_id, "bind_tools", {"tool_count": len(tools)})
        else:
            llm_with_tools = llm

        # 调用 LLM
        resp = llm_with_tools.invoke(prompt_msgs)
        
        # 检查工具调用
        has_tool_calls = hasattr(resp, 'tool_calls') and resp.tool_calls
        response_length = len(resp.content) if resp.content else 0
        
        workflow_logger.llm_response(thread_id, response_length, has_tool_calls)
        
        if has_tool_calls:
            for tc in (resp.tool_calls if isinstance(resp.tool_calls, list) else [resp.tool_calls]):
                tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                workflow_logger.tool_execution(thread_id, tc_name, tc.get("args", {}) if isinstance(tc, dict) else {})

        workflow_logger.node_exit("agent_response", thread_id, f"响应长度: {response_length}")
        return {"messages": [resp]}

    except Exception as e:
        workflow_logger.error(thread_id, "agent_response", e)
        return {
            "messages": [
                AIMessage(content=f"服务异常，请稍后重试：{str(e)[:150]}")
            ]
        }


async def agent_tech_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent技术专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("agent_tech", thread_id)
    result = await _build_agent_response(state, config, AGENT_PROMPT)
    workflow_logger.node_exit("agent_tech", thread_id, "响应生成完成")
    return result

async def travel_node(state: AgentState, config: RunnableConfig) -> dict:
    """旅游规划专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("travel", thread_id)
    result = await _build_agent_response(state, config, TRAVEL_TECH_PROMPT)
    workflow_logger.node_exit("travel", thread_id, "响应生成完成")
    return result


# ------------------------------
# 工具节点（企业级稳定版）
# ------------------------------
async def dynamic_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """动态工具执行节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tools", thread_id)
    
    try:
        tools = await tool_manager.get_tools()
        if not tools:
            workflow_logger.logger.warning(f"[{thread_id[:8]}] 没有可用的工具")
            return {"messages": []}

        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state, config)
        
        # 验证工具执行结果
        if "messages" in result:
            success_count = 0
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    success_count += 1
                    workflow_logger.tool_result(thread_id, msg.name, True, msg.content)
            
            workflow_logger.logger.info(f"[{thread_id[:8]}] 工具执行完成 | 成功: {success_count}/{len(result['messages'])}")
        else:
            workflow_logger.logger.warning(f"[{thread_id[:8]}] 工具节点返回结果中没有 messages")
            
        workflow_logger.node_exit("tools", thread_id, f"执行完成")
        return result

    except Exception as e:
        workflow_logger.error(thread_id, "tools", e)
        last_msg = state["messages"][-1]
        tool_msgs = []

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                t_id = tc.get("id", f"err_{uuid.uuid4().hex[:6]}") if isinstance(tc, dict) else f"err_{uuid.uuid4().hex[:6]}"
                t_name = tc.get("name", "unknown_tool") if isinstance(tc, dict) else "unknown_tool"
                workflow_logger.tool_result(thread_id, t_name, False, str(e))
                tool_msgs.append(
                    ToolMessage(
                        content=f"工具执行失败：{str(e)[:150]}",
                        tool_call_id=t_id,
                        name=t_name
                    )
                )
        return {"messages": tool_msgs}


# ------------------------------
# 工具循环控制（防死循环）
# ------------------------------
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """决定是否继续工具调用循环"""
    last = state["messages"][-1]

    # 最大迭代限制
    ai_count = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if ai_count >= MAX_ITERATIONS:
        logger.info(f"🔄 达到最大迭代次数 {MAX_ITERATIONS}，结束循环")
        return "__end__"

    # 检查是否有工具调用
    if hasattr(last, "tool_calls") and last.tool_calls:
        # 确保 tool_calls 是有效的列表
        if isinstance(last.tool_calls, list) and len(last.tool_calls) > 0:
            logger.info(f"🔄 检测到 {len(last.tool_calls)} 个工具调用，进入工具执行节点")
            return "tools"
        elif last.tool_calls is not None:
            # 处理非列表形式的 tool_calls
            logger.info(f"🔄 检测到工具调用（非标准格式），进入工具执行节点")
            return "tools"

    logger.info(f"🔄 没有工具调用，结束循环")
    return "__end__"


# ============================================================
# 第四步：构建流程图（核心修复）
# ============================================================
def _build_graph() -> StateGraph:
    """构建完整的 Agent 工作流图"""
    g = StateGraph(AgentState)

    # 节点
    g.add_node("memory", memory_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("agent_tech_rag", agent_tech_rag_node)
    g.add_node("agent_tech", agent_tech_node)
    g.add_node("travel_rag", travel_rag_node)
    g.add_node("travel", travel_node)
    g.add_node("tools", dynamic_tool_node)

    # 主线
    g.add_edge(START, "memory")
    g.add_edge("memory", "supervisor")

    # 路由分支
    g.add_conditional_edges(
        "supervisor",
        route_by_supervisor,
        {
            "agent_tech": "agent_tech_rag",
            "travel": "travel_rag",
        }
    )

    # Agent 分支
    g.add_edge("agent_tech_rag", "agent_tech")
    g.add_edge("travel_rag", "travel")

    # 工具循环
    g.add_conditional_edges("agent_tech", should_continue)
    g.add_conditional_edges("travel", should_continue)

    # 工具执行完 → 回到对应 Agent
    g.add_conditional_edges("tools", route_back_to_agent)

    return g


# ============================================================
# 异步图构建（线程安全单例）
# ============================================================
async def build_async_agent_graph():
    """构建异步 Agent 图（线程安全）"""
    graph = _build_graph()
    checkpointer = await get_async_checkpointer()
    return graph.compile(checkpointer=checkpointer)


async def get_async_agent():
    """获取异步 Agent 实例（单例）"""
    global _async_agent
    async with _agent_lock:
        if _async_agent is None:
            _async_agent = await build_async_agent_graph()
    return _async_agent


__all__ = ["AgentState", "get_async_agent"]
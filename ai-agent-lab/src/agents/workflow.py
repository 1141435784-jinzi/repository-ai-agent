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
5. 灵活性：支持多种工具交叉调用

【工作流架构】：
START → memory → supervisor → [agent_rag] → agent → should_continue
    ↓(有工具调用)                              ↓(无工具调用)
tool_selector → [tool_type_node] → tool_handler → should_continue
                                                    ↓
                                               summary → END
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
    PROJECT_ROOT,
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_BASE_SIGHTS_DIR,
    KNOWLEDGE_BASE_TRANSPORT_DIR,
    KNOWLEDGE_BASE_FINANCE_DIR,
    KNOWLEDGE_BASE_FOOD_DIR,
)
from src.memory import get_async_checkpointer, get_memory_manager
from src.llm.gateway import get_llm
from src.prompts import AGENT_PROMPT, SUPERVISOR_PROMPT, TRAVEL_TECH_PROMPT, SIGHTS_PROMPT, FOOD_PROMPT, TRANSPORT_PROMPT, FINANCE_PROMPT
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

def reduce_dict(prev: dict, next: Optional[dict]) -> dict:
    return next if next is not None else prev

def reduce_bool(prev: bool, next: Optional[bool]) -> bool:
    return next if next is not None else prev


# ============================================================
# 第一步：状态定义（完全符合 LangGraph 规范）
# ============================================================
class AgentState(TypedDict):
    """多 Agent 状态定义 - 支持多轮工具调用和 Agent 协作"""
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
    
    # 工具调用状态
    tool_type: Annotated[str, reduce_str]  # local/api/mcp
    tool_error: Annotated[str, reduce_str]  # 工具调用错误信息
    
    # Agent 协作
    collaboration_data: Annotated[dict, reduce_dict]  # Agent 间共享数据
    current_agent: Annotated[str, reduce_str]  # 当前执行的 Agent
    agent_history: Annotated[list, reduce_list]  # Agent 执行历史
    needs_collaboration: Annotated[bool, reduce_bool]  # 是否需要其他 Agent 协作
    collaboration_target: Annotated[str, reduce_str]  # 协作目标 Agent


# ============================================================
# 第二步：初始化 RAG 引擎
# ============================================================
print("📚 正在初始化 Agent 技术知识库...")
agent_tech_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_DIR,
    collection_name="agent_knowledge",
)
print("📚 Agent 技术知识库初始化完成 ✅")

print("🏛️ 正在初始化城市景点知识库...")
sights_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_SIGHTS_DIR,
    collection_name="sights_knowledge",
)
print("🏛️ 城市景点知识库初始化完成 ✅")

print("🚄 正在初始化交通知识库...")
transport_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_TRANSPORT_DIR,
    collection_name="transport_knowledge",
)
print("🚄 交通知识库初始化完成 ✅")

print("💰 正在初始化财务知识库...")
finance_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_FINANCE_DIR,
    collection_name="finance_knowledge",
)
print("💰 财务知识库初始化完成 ✅")

print("🍜 正在初始化美食知识库...")
food_rag = RAGEngine(
    knowledge_dir=KNOWLEDGE_BASE_FOOD_DIR,
    collection_name="food_knowledge",
)
print("🍜 美食知识库初始化完成 ✅")


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
        current_query=current_query
    )
    
    workflow_logger.node_exit("memory", thread_id, "记忆处理完成")
    return {"memory_context": result.get("context", "")}


def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """监督者节点 - 负责路由决策"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("supervisor", thread_id)
    
    try:
        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, "content") else ""
        
        llm = get_llm(provider="deepseek", streaming=False)
        workflow_logger.llm_call(thread_id, "supervisor", model=llm.model_name if hasattr(llm, 'model_name') else None)
        
        prompt = SUPERVISOR_PROMPT.invoke({"messages": [HumanMessage(content=user_query)]})
        resp = llm.invoke(prompt)
        
        # 解析路由决策
        route_decision = resp.content.strip().lower()
        
        # 扩展路由逻辑
        if "sights" in route_decision or "景点" in route_decision or "景区" in route_decision or "旅游景点" in route_decision:
            route = "sights"
        elif "transport" in route_decision or "交通" in route_decision or "航班" in route_decision or "高铁" in route_decision or "地铁" in route_decision:
            route = "transport"
        elif "finance" in route_decision or "财务" in route_decision or "预算" in route_decision or "费用" in route_decision or "花费" in route_decision:
            route = "finance"
        elif "food" in route_decision or "美食" in route_decision or "餐厅" in route_decision or "推荐菜" in route_decision:
            route = "food"
        elif "travel" in route_decision or "旅游" in route_decision:
            route = "sights"  # 默认旅游问题路由到景点专家
        else:
            route = "agent_tech"
        
        workflow_logger.logger.info(f"🔀 [{thread_id[:8]}] 路由决策: {route}")
        workflow_logger.node_exit("supervisor", thread_id, route)
        
        return {"route": route, "current_agent": route}
        
    except Exception as e:
        workflow_logger.error(thread_id, "supervisor", e)
        return {"route": "agent_tech", "current_agent": "agent_tech"}


def agent_tech_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent技术知识库检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("agent_tech_rag", thread_id)
    
    try:
        last_message = state["messages"][-1]
        question = last_message.content if hasattr(last_message, "content") else ""
        
        workflow_logger.rag_query(thread_id, question, "agent_knowledge")
        
        res = agent_tech_rag.query(question)
        workflow_logger.rag_result(thread_id, "agent_knowledge", res["sources"], len(res["answer_context"]))
        
        workflow_logger.node_exit("agent_tech_rag", thread_id, f"检索到 {len(res['sources'])} 个来源")
        return {"rag_context": res["answer_context"], "rag_sources": res["sources"]}
    except Exception as e:
        workflow_logger.error(thread_id, "agent_tech_rag", e)
        return {"rag_context": "", "rag_sources": []}


def travel_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """旅游知识库检索节点 - 同时检索景点和交通知识库"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("travel_rag", thread_id)
    
    try:
        last_message = state["messages"][-1]
        question = last_message.content if hasattr(last_message, "content") else ""
        
        # 检索景点知识库
        workflow_logger.rag_query(thread_id, question, "sights_rag")
        sights_res = sights_rag.query(question)
        
        # 检索交通知识库
        workflow_logger.rag_query(thread_id, question, "transport_rag")
        transport_res = transport_rag.query(question)
        
        # 合并检索结果
        rag_context = sights_res["answer_context"] + "\n\n" + transport_res["answer_context"]
        rag_sources = sights_res["sources"] + transport_res["sources"]
        
        workflow_logger.rag_result(thread_id, "travel_rag", rag_sources, len(rag_context))
        
        workflow_logger.node_exit("travel_rag", thread_id, f"检索到 {len(rag_sources)} 个来源")
        return {"rag_context": rag_context, "rag_sources": rag_sources}
    except Exception as e:
        workflow_logger.error(thread_id, "travel_rag", e)
        return {"rag_context": "", "rag_sources": []}


def _build_prompt_with_context(state: AgentState, prompt_template, messages: list) -> list:
    """构建带上下文的提示词"""
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


async def finance_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """财务知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("finance_rag", thread_id)
    
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else ""
    
    result = finance_rag.query(query)
    rag_context = result.get("answer_context", "")
    rag_sources = result.get("sources", [])
    
    workflow_logger.logger.info(f"📚 [{thread_id[:8]}] [finance_rag] RAG完成 | 来源: {rag_sources} | 上下文长度: {len(rag_context)}")
    workflow_logger.node_exit("finance_rag", thread_id, f"检索到 {len(rag_sources)} 个来源")
    
    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "current_agent": "finance"
    }


async def finance_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """财务规划专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("finance_agent", thread_id)
    result = await _build_agent_response(state, config, FINANCE_PROMPT)
    workflow_logger.node_exit("finance_agent", thread_id, "响应生成完成")
    return result


async def food_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """美食知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("food_rag", thread_id)
    
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else ""
    
    result = food_rag.query(query)
    rag_context = result.get("answer_context", "")
    rag_sources = result.get("sources", [])
    
    workflow_logger.logger.info(f"📚 [{thread_id[:8]}] [food_rag] RAG完成 | 来源: {rag_sources} | 上下文长度: {len(rag_context)}")
    workflow_logger.node_exit("food_rag", thread_id, f"检索到 {len(rag_sources)} 个来源")
    
    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "current_agent": "food"
    }


async def food_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """美食推荐专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("food_agent", thread_id)
    result = await _build_agent_response(state, config, FOOD_PROMPT)
    workflow_logger.node_exit("food_agent", thread_id, "响应生成完成")
    return result


async def sights_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """景点知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("sights_rag", thread_id)
    
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else ""
    
    result = sights_rag.query(query)
    rag_context = result.get("answer_context", "")
    rag_sources = result.get("sources", [])
    
    workflow_logger.logger.info(f"📚 [{thread_id[:8]}] [sights_rag] RAG完成 | 来源: {rag_sources} | 上下文长度: {len(rag_context)}")
    workflow_logger.node_exit("sights_rag", thread_id, f"检索到 {len(rag_sources)} 个来源")
    
    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "current_agent": "sights"
    }


async def sights_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """景点推荐专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("sights_agent", thread_id)
    result = await _build_agent_response(state, config, SIGHTS_PROMPT)
    workflow_logger.node_exit("sights_agent", thread_id, "响应生成完成")
    return result


async def transport_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """交通知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("transport_rag", thread_id)
    
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else ""
    
    result = transport_rag.query(query)
    rag_context = result.get("answer_context", "")
    rag_sources = result.get("sources", [])
    
    workflow_logger.logger.info(f"📚 [{thread_id[:8]}] [transport_rag] RAG完成 | 来源: {rag_sources} | 上下文长度: {len(rag_context)}")
    workflow_logger.node_exit("transport_rag", thread_id, f"检索到 {len(rag_sources)} 个来源")
    
    return {
        "rag_context": rag_context,
        "rag_sources": rag_sources,
        "current_agent": "transport"
    }


async def transport_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """交通出行专家节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("transport_agent", thread_id)
    result = await _build_agent_response(state, config, TRANSPORT_PROMPT)
    workflow_logger.node_exit("transport_agent", thread_id, "响应生成完成")
    return result


# ------------------------------
# 工具选择器节点
# ------------------------------
def tool_selector_node(state: AgentState, config: RunnableConfig) -> dict:
    """工具类型选择器 - 根据工具名判断工具类型"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    last_msg = state["messages"][-1]
    
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        tool_type = "local_tools"
    else:
        # 获取第一个工具调用
        tool_calls = last_msg.tool_calls if isinstance(last_msg.tool_calls, list) else [last_msg.tool_calls]
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else str(tool_call)
        
        # 根据工具名判断类型
        if tool_name.startswith("mcp_") or "_mcp_" in tool_name:
            tool_type = "mcp_tools"
        elif tool_name.startswith("api_") or "_api_" in tool_name:
            tool_type = "api_tools"
        else:
            tool_type = "local_tools"
    
    logger.info(f"🔧 [{thread_id[:8]}] 工具选择: {tool_name} -> {tool_type}")
    return {"tool_type": tool_type}


# ------------------------------
# 工具执行节点（按类型）
# ------------------------------
async def local_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """本地工具执行节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("local_tools", thread_id)
    
    try:
        tools = await tool_manager.get_tools_by_type("local")
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的本地工具")
            return {"tool_error": "没有可用的本地工具"}
        
        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state, config)
        
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    workflow_logger.tool_result(thread_id, msg.name, True, msg.content[:50])
        
        workflow_logger.node_exit("local_tools", thread_id, "执行完成")
        return result
        
    except Exception as e:
        workflow_logger.error(thread_id, "local_tools", e)
        return {"tool_error": str(e)[:150]}


async def api_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """API工具执行节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("api_tools", thread_id)
    
    try:
        tools = await tool_manager.get_tools_by_type("api")
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的API工具")
            return {"tool_error": "没有可用的API工具"}
        
        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state, config)
        
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    workflow_logger.tool_result(thread_id, msg.name, True, msg.content[:50])
        
        workflow_logger.node_exit("api_tools", thread_id, "执行完成")
        return result
        
    except Exception as e:
        workflow_logger.error(thread_id, "api_tools", e)
        return {"tool_error": str(e)[:150]}


async def mcp_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """MCP工具执行节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("mcp_tools", thread_id)
    
    try:
        tools = await tool_manager.get_tools_by_type("mcp")
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的MCP工具")
            return {"tool_error": "没有可用的MCP工具"}
        
        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state, config)
        
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    workflow_logger.tool_result(thread_id, msg.name, True, msg.content[:50])
        
        workflow_logger.node_exit("mcp_tools", thread_id, "执行完成")
        return result
        
    except Exception as e:
        workflow_logger.error(thread_id, "mcp_tools", e)
        return {"tool_error": str(e)[:150]}


# ------------------------------
# 工具结果处理节点
# ------------------------------
async def tool_result_handler(state: AgentState, config: RunnableConfig) -> dict:
    """工具结果处理节点 - 统一处理工具执行结果"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tool_result_handler", thread_id)
    
    try:
        # 检查是否有工具错误
        tool_error = state.get("tool_error", "")
        if tool_error:
            # 创建错误消息
            error_msg = AIMessage(
                content=f"工具执行失败：{tool_error}\n\n我将尝试其他方式回答您的问题。"
            )
            workflow_logger.node_exit("tool_result_handler", thread_id, "工具执行失败")
            return {"messages": [error_msg], "tool_error": ""}
        
        # 获取最后一条消息
        last_msg = state["messages"][-1]
        
        # 如果是 ToolMessage，创建一个总结消息
        if isinstance(last_msg, ToolMessage):
            # 创建一个临时的 AIMessage 来总结工具结果
            # 这将触发 should_continue 检查，决定是否继续调用工具
            summary_msg = AIMessage(
                content=f"工具执行完成，结果：{last_msg.content[:200]}..."
            )
            workflow_logger.node_exit("tool_result_handler", thread_id, "工具执行成功")
            return {"messages": [summary_msg]}
        
        workflow_logger.node_exit("tool_result_handler", thread_id, "无需处理")
        return {}
        
    except Exception as e:
        workflow_logger.error(thread_id, "tool_result_handler", e)
        return {"tool_error": str(e)[:150]}


# ------------------------------
# 工具循环控制（防死循环）
# ------------------------------
def should_continue(state: AgentState) -> Literal["tool_selector", "summary"]:
    """决定是否继续工具调用循环"""
    last = state["messages"][-1]

    # 最大迭代限制
    ai_count = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if ai_count >= MAX_ITERATIONS:
        logger.info(f"🔄 [{thread_id}] 达到最大迭代次数 {MAX_ITERATIONS}，结束循环")
        return "summary"

    # 检查是否有工具调用
    thread_id = state.get("route", "default")
    if hasattr(last, "tool_calls") and last.tool_calls:
        # 确保 tool_calls 是有效的列表
        if isinstance(last.tool_calls, list) and len(last.tool_calls) > 0:
            logger.info(f"🔄 [{thread_id[:8]}] 检测到 {len(last.tool_calls)} 个工具调用")
            return "tool_selector"
        elif last.tool_calls is not None:
            logger.info(f"🔄 [{thread_id[:8]}] 检测到工具调用（非标准格式）")
            return "tool_selector"

    logger.info(f"🔄 [{thread_id[:8]}] 没有工具调用，进入总结")
    return "summary"


# ------------------------------
# 总结节点
# ------------------------------
async def summary_node(state: AgentState, config: RunnableConfig) -> dict:
    """最终总结节点 - 生成最终回复"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("summary", thread_id)
    
    try:
        # 获取对话历史中的关键信息
        messages = state["messages"]
        
        # 找到最后一个有意义的回复
        last_meaningful_msg = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, 'tool_calls'):
                last_meaningful_msg = msg
                break
        
        if last_meaningful_msg:
            # 使用最后一个 AIMessage 作为回复
            workflow_logger.node_exit("summary", thread_id, "使用现有回复")
            return {"messages": [last_meaningful_msg]}
        
        # 如果没有找到有意义的回复，生成一个默认回复
        summary_content = "感谢您的提问！如有其他问题，请随时告诉我。"
        summary_msg = AIMessage(content=summary_content)
        
        workflow_logger.node_exit("summary", thread_id, "生成总结")
        return {"messages": [summary_msg]}
        
    except Exception as e:
        workflow_logger.error(thread_id, "summary", e)
        return {
            "messages": [AIMessage(content="抱歉，总结过程中出现错误。")]
        }


# ------------------------------
# 路由函数
# ------------------------------
def route_by_supervisor(state: AgentState) -> str:
    """根据 supervisor 的决策进行路由"""
    route = state.get("route", "agent_tech")
    return route


def route_back_to_agent(state: AgentState) -> str:
    """工具执行完成后回到对应的 Agent"""
    route = state.get("route", "agent_tech")
    if route == "travel":
        return "travel"
    return "agent_tech"


# ============================================================
# 第四步：构建流程图（企业级架构）
# ============================================================
def _build_graph() -> StateGraph:
    """构建完整的 Agent 工作流图"""
    g = StateGraph(AgentState)

    # 核心节点 - 记忆和监督者
    g.add_node("memory", memory_node)
    g.add_node("supervisor", supervisor_node)
    
    # 核心节点 - Agent技术专家
    g.add_node("agent_tech_rag", agent_tech_rag_node)
    g.add_node("agent_tech", agent_tech_node)
    
    # 核心节点 - 景点专家
    g.add_node("sights_rag", sights_rag_node)
    g.add_node("sights_agent", sights_agent_node)
    
    # 核心节点 - 交通专家
    g.add_node("transport_rag", transport_rag_node)
    g.add_node("transport_agent", transport_agent_node)
    
    # 核心节点 - 财务专家
    g.add_node("finance_rag", finance_rag_node)
    g.add_node("finance_agent", finance_agent_node)
    
    # 核心节点 - 美食专家
    g.add_node("food_rag", food_rag_node)
    g.add_node("food_agent", food_agent_node)
    
    # 工具选择器和执行节点
    g.add_node("tool_selector", tool_selector_node)
    g.add_node("local_tools", local_tool_node)
    g.add_node("api_tools", api_tool_node)
    g.add_node("mcp_tools", mcp_tool_node)
    g.add_node("tool_result_handler", tool_result_handler)
    
    # 总结节点
    g.add_node("summary", summary_node)

    # 主线流程
    g.add_edge(START, "memory")
    g.add_edge("memory", "supervisor")

    # 路由分支：supervisor -> RAG -> Agent
    g.add_conditional_edges(
        "supervisor",
        route_by_supervisor,
        {
            "agent_tech": "agent_tech_rag",
            "sights": "sights_rag",
            "transport": "transport_rag",
            "finance": "finance_rag",
            "food": "food_rag",
        }
    )
    
    # RAG -> Agent 连接
    g.add_edge("agent_tech_rag", "agent_tech")
    g.add_edge("sights_rag", "sights_agent")
    g.add_edge("transport_rag", "transport_agent")
    g.add_edge("finance_rag", "finance_agent")
    g.add_edge("food_rag", "food_agent")

    # Agent 决策：是否调用工具
    g.add_conditional_edges(
        "agent_tech",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "summary": "summary",
        }
    )
    g.add_conditional_edges(
        "sights_agent",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "summary": "summary",
        }
    )
    g.add_conditional_edges(
        "transport_agent",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "summary": "summary",
        }
    )
    g.add_conditional_edges(
        "finance_agent",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "summary": "summary",
        }
    )
    g.add_conditional_edges(
        "food_agent",
        should_continue,
        {
            "tool_selector": "tool_selector",
            "summary": "summary",
        }
    )

    # 工具选择器 -> 按类型执行
    g.add_conditional_edges(
        "tool_selector",
        lambda state: state.get("tool_type", "local_tools"),  # 从状态中获取 tool_type
        {
            "local_tools": "local_tools",
            "api_tools": "api_tools",
            "mcp_tools": "mcp_tools",
        }
    )

    # 工具执行完成 -> 结果处理 -> 回到 supervisor 进行路由决策
    g.add_edge("local_tools", "tool_result_handler")
    g.add_edge("api_tools", "tool_result_handler")
    g.add_edge("mcp_tools", "tool_result_handler")
    g.add_edge("tool_result_handler", "supervisor")

    # 总结 -> 结束
    g.add_edge("summary", END)

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
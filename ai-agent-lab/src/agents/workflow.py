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
2. 可靠性：确保工具调用循环完整性，支持重试机制
3. 可扩展性：支持动态工具注册和调用
4. 可观测性：完整的执行跟踪和错误处理
5. 灵活性：支持多种工具交叉调用
6. 高性能：状态图编译结果缓存

【工作流架构】：
START → memory → supervisor → [agent_rag] → agent → should_continue
    ↓(有工具调用)                              ↓(无工具调用)
tool_selector → [tool_type_node] → tool_handler → supervisor
                                                    ↓
                                               summary → END
"""

from typing import Annotated, List, Dict, Any, Optional, Literal
import asyncio
import logging
import uuid
import json
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    KNOWLEDGE_BASE_PLAN_DIR,
    KNOWLEDGE_BASE_FOOD_DIR,
)
from src.memory import get_async_checkpointer, get_memory_manager
from src.llm.gateway import get_llm
from src.prompts import (
    AGENT_PROMPT, SUPERVISOR_PROMPT, PLAN_PROMPT, 
    SIGHTS_PROMPT, FOOD_PROMPT, TRANSPORT_PROMPT
)
from src.rag.engine import RAGEngine
from src.agents.experts import (
    agent_manager, get_agent_tech_expert, get_plan_expert,
    get_sights_expert, get_food_expert, get_transport_expert
)
from src.tools.tool_manager import tool_manager
from src.utils.logger import WorkflowLogger

logger = logging.getLogger(__name__)
workflow_logger = WorkflowLogger(logger)


# ============================================================
# 全局工具锁：防止并发重复编译
# ============================================================
_agent_lock = asyncio.Lock()
_async_agent = None
_graph_compile_cache = {}  # 状态图编译结果缓存


# ============================================================
# 工具调用异常类
# ============================================================
class ToolCallError(Exception):
    """工具调用异常"""
    pass


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
    """多 Agent 状态定义 - 支持多轮工具调用、Agent 协作、任务分解"""
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
    tool_retry_count: Annotated[int, reduce_str]  # 工具调用重试计数（自愈节点使用）
    has_tool_calls: Annotated[bool, reduce_bool]  # 是否有工具调用
    
    # Agent 协作与执行清单
    collaboration_data: Annotated[dict, reduce_dict]  # Agent 间共享数据
    current_agent: Annotated[str, reduce_str]  # 当前执行的 Agent
    agent_history: Annotated[list, reduce_list]  # Agent 执行历史
    needs_collaboration: Annotated[bool, reduce_bool]  # 是否需要其他 Agent 协作
    collaboration_target: Annotated[str, reduce_str]  # 协作目标 Agent
    collaboration_reason: Annotated[str, reduce_str]  # 协作原因
    execution_plan: Annotated[list, reduce_list]  # 企业级执行清单 (Execution List)
    
    # 任务分解
    task_decomposition: Annotated[dict, reduce_dict]  # 任务分解结果
    subtasks: Annotated[list, reduce_list]  # 子任务列表
    current_subtask: Annotated[int, reduce_str]  # 当前执行的子任务索引
    
    # 反思总结
    reflection_notes: Annotated[list, reduce_list]  # 反思笔记
    key_decisions: Annotated[list, reduce_list]  # 关键决策记录
    
    # 迭代计数（用于防止死循环）
    iteration_count: Annotated[int, reduce_str]  # 当前迭代次数


# ============================================================
# 第二步：初始化领域专家（含 RAG 引擎）
# ============================================================
async def initialize_experts():
    """统一初始化所有领域专家"""
    print("🔄 正在初始化领域专家系统...")
    # 触发单例创建并注册到 manager
    get_agent_tech_expert()
    get_plan_expert()
    get_sights_expert()
    get_food_expert()
    get_transport_expert()
    
    # 统一异步初始化（RAG 引擎等）
    await agent_manager.initialize_all()
    print("✅ 领域专家系统初始化完成")

# 在模块加载时，如果不在异步环境，可以由外部调用 initialize_experts
# 或者在 graph 运行前确保已初始化


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


async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """监督者节点 - 负责路由决策"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("supervisor", thread_id)
    
    try:
        # 获取当前任务内容
        plan = state.get("execution_plan", [])
        current_task_item = next((t for t in plan if t["agent"] == "supervisor" and t["status"] == "pending"), None)
        user_query = current_task_item["task"] if current_task_item else ""
        
        if not user_query:
            last_message = state["messages"][-1]
            user_query = last_message.content if hasattr(last_message, "content") else ""
        
        llm = get_llm(provider="deepseek", streaming=True)
        workflow_logger.llm_call(thread_id, "supervisor", model=llm.model_name if hasattr(llm, 'model_name') else None)
        
        prompt = SUPERVISOR_PROMPT.invoke({"messages": [HumanMessage(content=user_query)]})
        resp = await llm.ainvoke(prompt)
        
        # 解析路由决策
        route = resp.content.strip().lower()
        
        # 验证路由结果是否在可用专家列表中
        available_experts = ["agent_tech", "plan", "sights", "transport", "food"]
        if route not in available_experts:
            # 尝试从文本中提取关键字
            for expert in available_experts:
                if expert in route:
                    route = expert
                    break
            else:
                # 兼容旧名称路由
                if "travel" in route or "finance" in route:
                    route = "plan"
                else:
                    route = "agent_tech" # 兜底路由
        
        workflow_logger.logger.info(f"🔀 [{thread_id[:8]}] 路由决策: {route}")
        
        # 更新执行计划：将当前 supervisor 任务标记为已完成，并插入真正的专家任务
        new_plan = []
        for t in plan:
            if t["agent"] == "supervisor" and t["status"] == "pending":
                # 标记当前 supervisor 任务完成
                new_plan.append({**t, "status": "completed"})
                # 插入新的专家任务
                new_plan.append({
                    "id": len(plan) + 1,
                    "task": user_query,
                    "agent": route,
                    "status": "pending"
                })
            else:
                new_plan.append(t)
        
        workflow_logger.node_exit("supervisor", thread_id, route)
        return {"route": route, "current_agent": route, "execution_plan": new_plan}
        
    except Exception as e:
        workflow_logger.error(thread_id, "supervisor", e)
        return {"route": "agent_tech", "current_agent": "agent_tech"}


async def agent_tech_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent技术知识库检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("agent_tech_rag", thread_id)
    
    try:
        # 从执行计划中获取当前子任务作为查询词
        plan = state.get("execution_plan", [])
        question = next((t["task"] for t in plan if t["status"] == "pending"), "")
        if not question:
            last_message = state["messages"][-1]
            question = last_message.content if hasattr(last_message, "content") else ""
        
        workflow_logger.rag_query(thread_id, question, "agent_knowledge")
        
        agent = agent_manager.get_agent("agent_tech")
        res = await agent.query_rag(question) if agent else {"found": False, "answer_context": "", "sources": []}
        
        workflow_logger.rag_result(thread_id, "agent_knowledge", res["sources"], len(res["answer_context"]))
        
        workflow_logger.node_exit("agent_tech_rag", thread_id, f"检索到 {len(res['sources'])} 个来源")
        return {"rag_context": res["answer_context"], "rag_sources": res["sources"]}
    except Exception as e:
        workflow_logger.error(thread_id, "agent_tech_rag", e)
        return {"rag_context": "", "rag_sources": []}


async def plan_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """规划知识库检索节点 - 整合旅行规划与财务预算"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("plan_rag", thread_id)
    
    try:
        plan = state.get("execution_plan", [])
        question = next((t["task"] for t in plan if t["status"] == "pending"), "")
        if not question:
            last_message = state["messages"][-1]
            question = last_message.content if hasattr(last_message, "content") else ""
        
        workflow_logger.rag_query(thread_id, question, "plan_knowledge")
        
        agent = agent_manager.get_agent("plan")
        res = await agent.query_rag(question) if agent else {"found": False, "answer_context": "", "sources": []}
        
        workflow_logger.rag_result(thread_id, "plan_rag", res["sources"], len(res["answer_context"]))
        workflow_logger.node_exit("plan_rag", thread_id, f"检索到 {len(res['sources'])} 个来源")
        return {"rag_context": res["answer_context"], "rag_sources": res["sources"]}
    except Exception as e:
        workflow_logger.error(thread_id, "plan_rag", e)
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
        src_str = "、".join(list(set(sources))) # 去重
        rag_msg = SystemMessage(
            content=f"【知识库参考资料】(来源: {src_str})\n{rag_ctx}\n\n"
            "【重要指令】:\n"
            f"1. 如果你的回答采用了上述参考资料中的任何信息（如景点名称、美食特色等），请务必在回答末尾精确标注：`（数据来源：{src_str}）`。\n"
            "2. 即使你认为这是常识，只要参考资料中提供了，也请标注来源。\n"
            "3. 如果参考资料完全无关，请在末尾标注：`（以上内容基于 AI 通用知识）`。"
        )
        
        # 始终插在系统消息（索引 0）之后，确保最高优先级
        prompt_messages.messages.insert(1, rag_msg)

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

        # 获取当前任务指令（从执行计划中提取）
        plan = state.get("execution_plan", [])
        current_task = next((t["task"] for t in plan if t["status"] == "pending"), "处理用户请求")
        
        # 构建提示
        prompt_msgs = _build_prompt_with_context(state, prompt_template, cleaned_msgs)
        
        # 注入当前任务指令，引导 Agent 专注
        task_msg = SystemMessage(
            content=f"【当前子任务指令】：{current_task}\n"
            "请仅针对此子任务进行回答或执行工具调用。不要重复回答其他专家已经处理过的内容。"
        )
        # 插在 SystemMessage 之后
        prompt_msgs.insert(1, task_msg)

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

        # 调用 LLM（使用异步 ainvoke 以支持流式输出）
        resp = await llm_with_tools.ainvoke(prompt_msgs)
        
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
    
    # 更新执行清单状态
    plan = state.get("execution_plan", [])
    new_plan = [
        {**t, "status": "completed" if t["agent"] == "agent_tech" and t["status"] == "pending" else t["status"]}
        for t in plan
    ]
    
    result = await _build_agent_response(state, config, AGENT_PROMPT)
    workflow_logger.node_exit("agent_tech", thread_id, "响应生成完成")
    return {**result, "current_agent": "agent_tech", "execution_plan": new_plan}


async def plan_node(state: AgentState, config: RunnableConfig) -> dict:
    """规划专家节点 - 整合旅行规划与财务预算"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("plan", thread_id)
    
    # 更新执行清单状态
    plan = state.get("execution_plan", [])
    new_plan = [
        {**t, "status": "completed" if t["agent"] == "plan" and t["status"] == "pending" else t["status"]}
        for t in plan
    ]
    
    result = await _build_agent_response(state, config, PLAN_PROMPT)
    workflow_logger.node_exit("plan", thread_id, "响应生成完成")
    return {**result, "current_agent": "plan", "execution_plan": new_plan}


async def food_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """美食知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("food_rag", thread_id)
    
    # 查找分配给 food agent 的第一个 pending 任务
    plan = state.get("execution_plan", [])
    query = next((t["task"] for t in plan if t["agent"] == "food" and t["status"] == "pending"), "")
    if not query:
        last_message = state["messages"][-1]
        query = last_message.content if hasattr(last_message, "content") else ""
    
    agent = agent_manager.get_agent("food")
    res = await agent.query_rag(query) if agent else {"found": False, "answer_context": "", "sources": []}
    
    rag_context = res.get("answer_context", "")
    rag_sources = res.get("sources", [])
    
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
    
    plan = state.get("execution_plan", [])
    new_plan = [{**t, "status": "completed" if t["agent"] == "food" and t["status"] == "pending" else t["status"]} for t in plan]
    
    result = await _build_agent_response(state, config, FOOD_PROMPT)
    workflow_logger.node_exit("food_agent", thread_id, "响应生成完成")
    return {**result, "current_agent": "food", "execution_plan": new_plan}


async def sights_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """景点知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("sights_rag", thread_id)
    
    # 查找分配给 sights agent 的第一个 pending 任务
    plan = state.get("execution_plan", [])
    query = next((t["task"] for t in plan if t["agent"] == "sights" and t["status"] == "pending"), "")
    if not query:
        last_message = state["messages"][-1]
        query = last_message.content if hasattr(last_message, "content") else ""
    
    agent = agent_manager.get_agent("sights")
    res = await agent.query_rag(query) if agent else {"found": False, "answer_context": "", "sources": []}
    
    rag_context = res.get("answer_context", "")
    rag_sources = res.get("sources", [])
    
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
    
    plan = state.get("execution_plan", [])
    new_plan = [{**t, "status": "completed" if t["agent"] == "sights" and t["status"] == "pending" else t["status"]} for t in plan]
    
    result = await _build_agent_response(state, config, SIGHTS_PROMPT)
    workflow_logger.node_exit("sights_agent", thread_id, "响应生成完成")
    return {**result, "current_agent": "sights", "execution_plan": new_plan}


async def transport_rag_node(state: AgentState, config: RunnableConfig) -> dict:
    """交通知识检索节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("transport_rag", thread_id)
    
    # 查找分配给 transport agent 的第一个 pending 任务
    plan = state.get("execution_plan", [])
    query = next((t["task"] for t in plan if t["agent"] == "transport" and t["status"] == "pending"), "")
    if not query:
        last_message = state["messages"][-1]
        query = last_message.content if hasattr(last_message, "content") else ""
    
    agent = agent_manager.get_agent("transport")
    res = await agent.query_rag(query) if agent else {"found": False, "answer_context": "", "sources": []}
    
    rag_context = res.get("answer_context", "")
    rag_sources = res.get("sources", [])
    
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
    
    plan = state.get("execution_plan", [])
    new_plan = [{**t, "status": "completed" if t["agent"] == "transport" and t["status"] == "pending" else t["status"]} for t in plan]
    
    result = await _build_agent_response(state, config, TRANSPORT_PROMPT)
    workflow_logger.node_exit("transport_agent", thread_id, "响应生成完成")
    return {**result, "current_agent": "transport", "execution_plan": new_plan}


# ------------------------------
# 任务分解节点（新增）
# ------------------------------
async def task_decomposition_node(state: AgentState, config: RunnableConfig) -> dict:
    """任务分解节点 - 将复杂任务分解为子任务并生成执行计划"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("task_decomposition", thread_id)
    
    try:
        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, "content") else ""
        
        # 分析任务复杂度
        intent_keywords = ["天气", "景点", "美食", "餐厅", "预算", "花费", "交通", "酒店", "机票", "门票"]
        found_intents = [kw for kw in intent_keywords if kw in user_query]
        
        complexity_indicators = [
            len(found_intents) >= 2, # 包含两个以上领域关键词
            len(user_query) > 60,    # 长度超过60字
            "规划" in user_query or "安排" in user_query or "推荐" in user_query,
            "、" in user_query or "，" in user_query and len(user_query) > 30,
        ]
        
        if sum(complexity_indicators) >= 2 or len(found_intents) >= 3:
            llm = get_llm(provider="deepseek", streaming=True)
            
            decomposition_prompt = f"""
            请将以下用户请求分解为多个逻辑独立的子任务，并为每个子任务指定最合适的专家。
            
            用户请求：{user_query}
            
            可选专家列表：
            - plan: 旅行目的地推荐、大行程规划、签证政策、费用精算、汇率换算、保险建议、开支优化
            - sights: 具体景点解说、门票政策、开放时间、景区内路径规划
            - transport: 航班/车次查询、交通方案对比、换乘指引、接驳指南
            - food: 菜品推荐、餐厅点评、预订建议、美食街区探店
            - agent_tech: AI技术问题、数学计算、时间/天气查询等通用任务
            
            请输出JSON格式，包含以下字段：
            - "is_complex": true
            - "execution_plan": 子任务清单，每个任务包含:
                - "id": 任务编号 (1, 2, 3...)
                - "task": 任务简述
                - "agent": 推荐专家名称 (plan/sights/transport/food/agent_tech)
                - "status": "pending"
            - "reason": 分解逻辑说明
            
            示例输出：
            {{
                "is_complex": true,
                "execution_plan": [
                    {{"id": 1, "task": "查询深圳天气", "agent": "agent_tech", "status": "pending"}},
                    {{"id": 2, "task": "推荐深圳景点", "agent": "sights", "status": "pending"}}
                ],
                "reason": "需要先了解天气再规划户外景点"
            }}
            """
            
            resp = await llm.ainvoke(decomposition_prompt)
            try:
                # 去除可能的 markdown 代码块标记
                content = resp.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                decomposition = json.loads(content)
                if decomposition.get("is_complex") and decomposition.get("execution_plan"):
                    workflow_logger.logger.info(f"🔍 [{thread_id[:8]}] 任务分解完成: {len(decomposition['execution_plan'])} 个步骤")
                    return {
                        "task_decomposition": decomposition,
                        "execution_plan": decomposition["execution_plan"],
                        "current_subtask": 0,
                    }
            except Exception as e:
                workflow_logger.logger.error(f"❌ 任务分解解析失败: {e}")
        
        # 简单任务，路由到 supervisor 进行意图识别
        workflow_logger.node_exit("task_decomposition", thread_id, "简单任务，生成路由计划")
        return {
            "task_decomposition": {"is_complex": False},
            "execution_plan": [{"id": 1, "task": user_query, "agent": "supervisor", "status": "pending"}]
        }
        
    except Exception as e:
        workflow_logger.error(thread_id, "task_decomposition", e)
        return {"execution_plan": [{"id": 1, "task": "处理用户请求", "agent": "supervisor", "status": "pending"}]}


# ------------------------------
# Agent 协作决策节点（新增）
# ------------------------------
def collaboration_decision_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent 协作决策节点 - 判断是否需要其他 Agent 协作或工具调用"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("collaboration_decision", thread_id)
    
    try:
        last_message = state["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else ""
        
        # 检查是否有工具调用
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        if has_tool_calls:
            # 确保 tool_calls 是有效的列表
            is_valid_tool_call = isinstance(last_message.tool_calls, list) and len(last_message.tool_calls) > 0
            if not is_valid_tool_call:
                is_valid_tool_call = last_message.tool_calls is not None
        
        # 检查是否提到其他领域
        collaboration_triggers = {
            "plan": ["规划", "计划", "安排", "旅游", "旅行", "预算", "费用", "价格", "财务", "花销", "汇率"],
            "sights": ["景点", "景区", "风景", "观光"],
            "transport": ["交通", "高铁", "航班", "地铁", "公交", "机票", "车票"],
            "food": ["美食", "餐厅", "吃饭", "推荐菜", "特产"],
        }
        
        current_agent = state.get("current_agent", "")
        needs_collaboration = False
        target_agent = ""
        reason = ""
        
        # 检查当前 Agent 是否需要其他 Agent 的帮助
        for agent, triggers in collaboration_triggers.items():
            if agent != current_agent:  # 不是当前 Agent
                if any(trigger in content for trigger in triggers):
                    needs_collaboration = True
                    target_agent = agent
                    reason = f"当前 Agent ({current_agent}) 需要 {agent} Agent 的专业知识"
                    break
        
        # 检查任务分解是否需要多 Agent 协作
        subtasks = state.get("subtasks", [])
        if subtasks and len(subtasks) > 1:
            # 多子任务可能需要协作
            needs_collaboration = True
            target_agent = "supervisor"
            reason = "多子任务需要协作处理"
        
        workflow_logger.logger.info(f"🤝 [{thread_id[:8]}] 协作决策: {needs_collaboration} -> {target_agent}, 工具调用: {has_tool_calls}")
        workflow_logger.node_exit("collaboration_decision", thread_id, 
                                  f"协作需求: {needs_collaboration}, 目标: {target_agent}")
        
        return {
            "needs_collaboration": needs_collaboration,
            "collaboration_target": target_agent,
            "collaboration_reason": reason,
            "has_tool_calls": has_tool_calls
        }
        
    except Exception as e:
        workflow_logger.error(thread_id, "collaboration_decision", e)
        return {
            "needs_collaboration": False,
            "collaboration_target": "",
            "collaboration_reason": "",
            "has_tool_calls": False
        }


# ------------------------------
# 反思节点（新增）
# ------------------------------
def reflection_node(state: AgentState, config: RunnableConfig) -> dict:
    """反思节点 - 回顾对话历史，记录关键决策并进行上下文压缩（裁剪）"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("reflection", thread_id)
    
    try:
        messages = state["messages"]
        agent_history = state.get("agent_history", [])
        key_decisions = []
        
        # 1. 提取关键决策
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls = msg.tool_calls if isinstance(msg.tool_calls, list) else [msg.tool_calls]
                for tc in tool_calls:
                    tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                    key_decisions.append({
                        "step": i,
                        "decision_type": "tool_call",
                        "tool_name": tool_name
                    })
        
        # 2. 上下文压缩（剪枝策略）：
        # 如果消息超过 15 条，保留前 2 条（系统/起始）和最后 8 条
        # 这样既保留了任务目标，又保留了最近的上下文，同时移除了中间冗余的工具调用过程
        trimmed_messages = messages
        if len(messages) > 15:
            trimmed_messages = messages[:2] + messages[-8:]
            workflow_logger.logger.info(f"✂️ [{thread_id[:8]}] 上下文剪枝: {len(messages)} -> {len(trimmed_messages)} 条消息")
        
        workflow_logger.node_exit("reflection", thread_id, "反思与剪枝完成")
        return {
            "key_decisions": key_decisions,
            "messages": trimmed_messages, # 更新消息列表以释放 Token 压力
            "reflection_notes": [f"已执行步数: {len(agent_history)}"]
        }
        
    except Exception as e:
        workflow_logger.error(thread_id, "reflection", e)
        return {"key_decisions": [], "reflection_notes": []}


# ------------------------------
# 带重试机制的工具调用辅助函数
# ------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ToolCallError, ConnectionError, asyncio.TimeoutError)),
    reraise=True
)
async def _call_tool_with_retry(tool_node: ToolNode, state: AgentState, config: RunnableConfig) -> dict:
    """带重试机制的工具调用"""
    result = await tool_node.ainvoke(state, config)
    return result


# ------------------------------
# 统一工具执行节点（支持混合调用和重试）
# ------------------------------
async def unified_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """统一工具执行节点 - 支持多种类型工具同时调用，并具备重试机制"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tools", thread_id)
    
    try:
        # 获取所有可用工具
        tools = await tool_manager.get_tools()
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的工具")
            return {"tool_error": "没有可用的工具"}
        
        # 创建 LangGraph 的 ToolNode
        tool_node = ToolNode(tools)
        
        # 使用带重试的工具调用
        result = await _call_tool_with_retry(tool_node, state, config)
        
        # 记录工具执行结果日志
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    success = not str(msg.content).startswith("Error")
                    workflow_logger.tool_result(thread_id, msg.name, success, str(msg.content)[:100])
        
        workflow_logger.node_exit("tools", thread_id, "执行完成")
        return result
        
    except ToolCallError as e:
        workflow_logger.error(thread_id, "tools", f"工具调用失败（已重试）: {e}")
        return {"tool_error": f"工具调用失败: {str(e)[:150]}"}
    except Exception as e:
        workflow_logger.error(thread_id, "tools", e)
        return {"tool_error": str(e)[:150]}


# ------------------------------
# 工具结果处理节点
# ------------------------------
async def tool_result_handler(state: AgentState, config: RunnableConfig) -> dict:
    """工具结果处理节点 - 统一处理工具执行结果并更新任务状态"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tool_result_handler", thread_id)
    
    try:
        # 检查是否有工具错误
        tool_error = state.get("tool_error", "")
        if tool_error:
            # 记录错误并增加重试计数
            retry_count = state.get("tool_retry_count", 0) + 1
            workflow_logger.node_exit("tool_result_handler", thread_id, f"工具报错，重试计数: {retry_count}")
            return {"tool_retry_count": retry_count}
        
        # 成功执行，重置重试计数
        workflow_logger.node_exit("tool_result_handler", thread_id, "工具执行成功")
        return {"tool_retry_count": 0}
        
    except Exception as e:
        workflow_logger.error(thread_id, "tool_result_handler", e)
        return {"tool_error": str(e)[:150]}


async def self_healing_node(state: AgentState, config: RunnableConfig) -> dict:
    """自愈节点 - 分析工具错误并尝试修复参数或更换策略"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("self_healing", thread_id)
    
    try:
        retry_count = state.get("tool_retry_count", 0)
        if retry_count >= 3:
            # 超过最大尝试次数，告知用户并停止
            workflow_logger.node_exit("self_healing", thread_id, "已达最大重试次数，放弃自愈")
            return {"messages": [AIMessage(content="抱歉，我多次尝试调用工具都失败了，可能暂时无法为您处理此请求。")]}

        # 获取最后的消息（通常是工具调用和对应的错误）
        last_msgs = state["messages"][-2:]
        error_info = state.get("tool_error", "未知错误")
        
        llm = get_llm(provider="deepseek", streaming=True)
        
        healing_prompt = f"""
        你是一个自愈专家。上一个工具调用失败了。
        
        工具调用信息：{last_msgs[0]}
        错误信息：{error_info}
        
        请分析原因并给出一个修正后的建议：
        1. 如果是参数格式问题，请给出正确的参数。
        2. 如果是工具不可用，请建议更换其他工具或直接回答。
        
        请直接输出修复建议或说明，不要带多余废话。
        """
        
        resp = await llm.ainvoke(healing_prompt)
        workflow_logger.node_exit("self_healing", thread_id, "已生成修复策略")
        return {"messages": [AIMessage(content=f"🔧 自动尝试修复中: {resp.content}")]}
        
    except Exception as e:
        workflow_logger.error(thread_id, "self_healing", e)
        return {}


# ------------------------------
# 工具循环控制（防死循环）
# ------------------------------
def should_continue(state: AgentState) -> Literal["tool_selector", "summary"]:
    """决定是否继续工具调用循环"""
    last = state["messages"][-1]
    thread_id = state.get("route", "default")

    # 最大迭代限制
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= MAX_ITERATIONS:
        logger.info(f"🔄 [{thread_id[:8]}] 达到最大迭代次数 {MAX_ITERATIONS}，结束循环")
        return "summary"

    # 检查是否有工具调用
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_selector"

    return "summary"


# ------------------------------
# 总结节点
# ------------------------------
async def summary_node(state: AgentState, config: RunnableConfig) -> dict:
    """最终总结节点 - 生成最终回复，并确保合并所有子任务的来源信息"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("summary", thread_id)
    
    try:
        messages = state["messages"]
        plan = state.get("execution_plan", [])
        
        # 如果是复杂分解任务，且任务已完成
        if plan and len(plan) > 1:
            # 【修复】找到最后一条用户消息的位置，只保留之后的 AI 消息
            # 这样可以保留同一个对话轮次内多个 Agent 的回复（多任务协作），同时排除历史对话轮次的回复
            last_human_idx = None
            for i, m in enumerate(messages):
                if isinstance(m, HumanMessage):
                    last_human_idx = i
            
            # 只保留最后一条用户消息之后的 AI 消息
            if last_human_idx is not None:
                ai_messages = [m for m in messages[last_human_idx+1:] if isinstance(m, AIMessage) and m.content and not m.tool_calls]
            else:
                ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content and not m.tool_calls]
            
            # 过滤掉一些简短的过渡消息（如“我先查一下...”、“好的，我为您规划...”）
            meaningful_msgs = []
            seen_content = set()
            for m in ai_messages:
                content = m.content.strip()
                if len(content) > 30 and content not in seen_content:
                    meaningful_msgs.append(m)
                    seen_content.add(content)
            
            if len(meaningful_msgs) > 1:
                # 拼接所有有意义的回复（多 Agent 协作场景）
                combined_content = "\n\n".join([m.content for m in meaningful_msgs])
                workflow_logger.node_exit("summary", thread_id, f"合并了 {len(meaningful_msgs)} 条子任务回复")
                return {"messages": [AIMessage(content=combined_content)]}

        # 寻找最后一条有意义的 AI 回复
        last_meaningful_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and not m.tool_calls), None)
        
        if last_meaningful_msg:
            workflow_logger.node_exit("summary", thread_id, "使用单条有意义回复")
            return {"messages": [last_meaningful_msg]}
        
        summary_msg = AIMessage(content="感谢您的提问！如有其他问题，请随时告诉我。")
        workflow_logger.node_exit("summary", thread_id, "生成默认总结")
        return {"messages": [summary_msg]}
        
    except Exception as e:
        workflow_logger.error(thread_id, "summary", e)
        return {"messages": [AIMessage(content="抱歉，总结过程中出现错误。")]}


# ------------------------------
# 企业级执行清单路由 (Execution List Router)
# ------------------------------
def execution_list_router(state: AgentState) -> str:
    """根据执行清单决定下一个节点"""
    plan = state.get("execution_plan", [])
    
    # 寻找第一个 pending 的任务
    next_task = next((t for t in plan if t["status"] == "pending"), None)
    
    if not next_task:
        # 所有任务已完成，进入反思
        return "reflection"
    
    # 映射专家名称到 RAG 节点
    agent_map = {
        "plan": "plan_rag",
        "sights": "sights_rag",
        "transport": "transport_rag",
        "food": "food_rag",
        "agent_tech": "agent_tech_rag",
        "supervisor": "supervisor"
    }
    
    target = agent_map.get(next_task["agent"], "agent_tech_rag")
    
    # 更新任务状态为 in_progress（模拟）
    # 注意：在路由器里不直接修改 state，而是在节点进入时修改
    return target


def collaboration_or_tool_decision(state: AgentState) -> str:
    """协作和工具调用决策路由 (增强自愈路径)"""
    last_msg = state["messages"][-1]
    
    # 1. 如果有工具错误且未达上限，去自愈节点
    tool_error = state.get("tool_error", "")
    retry_count = state.get("tool_retry_count", 0)
    if tool_error and retry_count > 0:
        return "self_healing"

    # 2. 如果最后一条消息是工具调用请求，去执行工具
    is_tool_call = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
    if is_tool_call:
        return "tool_selector"
    
    # 3. 如果最后一条消息是工具执行结果，返回对应的 Agent 进行总结
    is_tool_result = isinstance(last_msg, ToolMessage) or (isinstance(last_msg, dict) and last_msg.get("type") == "tool")
    if is_tool_result:
        current_agent = state.get("current_agent") or "agent_tech"
        agent_node_map = {
            "agent_tech": "agent_tech",
            "plan": "plan",
            "sights": "sights_agent",
            "transport": "transport_agent",
            "food": "food_agent"
        }
        return agent_node_map.get(current_agent, "agent_tech")

    # 4. 如果当前子任务已完成，回到执行清单路由器
    return "execution_router"


# ============================================================
# 第四步：构建流程图（企业级架构 - 优化版）
# ============================================================
def _build_graph() -> StateGraph:
    """构建完整的 Agent 工作流图 (SOTA 架构)"""
    g = StateGraph(AgentState)

    # 基础节点
    g.add_node("memory", memory_node)
    g.add_node("task_decomposition", task_decomposition_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("reflection", reflection_node)
    g.add_node("summary", summary_node)
    
    # 执行清单辅助节点
    g.add_node("execution_router_node", lambda x: x) # 占位节点
    
    # 专家节点
    g.add_node("agent_tech_rag", agent_tech_rag_node)
    g.add_node("agent_tech", agent_tech_node)
    g.add_node("plan_rag", plan_rag_node)
    g.add_node("plan", plan_node)
    g.add_node("sights_rag", sights_rag_node)
    g.add_node("sights_agent", sights_agent_node)
    g.add_node("transport_rag", transport_rag_node)
    g.add_node("transport_agent", transport_agent_node)
    g.add_node("food_rag", food_rag_node)
    g.add_node("food_agent", food_agent_node)
    
    # 工具与自愈
    g.add_node("tools", unified_tool_node)
    g.add_node("tool_result_handler", tool_result_handler)
    g.add_node("self_healing", self_healing_node)

    # --- 流程连线 ---
    
    # 1. 启动与计划
    g.add_edge(START, "memory")
    g.add_edge("memory", "task_decomposition")
    g.add_edge("task_decomposition", "execution_router_node")
    
    # 2. 执行清单动态路由
    g.add_conditional_edges(
        "execution_router_node",
        execution_list_router,
        {
            "plan_rag": "plan_rag",
            "sights_rag": "sights_rag",
            "transport_rag": "transport_rag",
            "food_rag": "food_rag",
            "agent_tech_rag": "agent_tech_rag",
            "supervisor": "supervisor",
            "reflection": "reflection"
        }
    )
    
    # 3. 专家协作
    g.add_edge("agent_tech_rag", "agent_tech")
    g.add_edge("plan_rag", "plan")
    g.add_edge("sights_rag", "sights_agent")
    g.add_edge("transport_rag", "transport_agent")
    g.add_edge("food_rag", "food_agent")
    
    # 所有专家执行完后进入决策
    for node in ["agent_tech", "plan", "sights_agent", "transport_agent", "food_agent", "supervisor"]:
        g.add_conditional_edges(
            node,
            collaboration_or_tool_decision,
            {
                "tool_selector": "tools",
                "self_healing": "self_healing",
                "execution_router": "execution_router_node",
                "agent_tech": "agent_tech",
                "plan": "plan",
                "sights_agent": "sights_agent",
                "transport_agent": "transport_agent",
                "food_agent": "food_agent"
            }
        )

    # 4. 工具循环与自愈
    g.add_edge("tools", "tool_result_handler")
    g.add_conditional_edges(
        "tool_result_handler",
        collaboration_or_tool_decision,
        {
            "self_healing": "self_healing",
            "tool_selector": "tools", # 理论上 handler 不会直接回 tools，但保留兼容
            "execution_router": "execution_router_node",
            "agent_tech": "agent_tech",
            "plan": "plan",
            "sights_agent": "sights_agent",
            "transport_agent": "transport_agent",
            "food_agent": "food_agent"
        }
    )
    
    # 自愈后尝试重新执行专家逻辑
    g.add_edge("self_healing", "execution_router_node")

    # 5. 结束
    g.add_edge("reflection", "summary")
    g.add_edge("summary", END)

    return g


# ============================================================
# 异步图构建（线程安全单例 + 编译结果缓存）
# ============================================================
async def build_async_agent_graph(graph_id: str = "default"):
    """构建异步 Agent 图（线程安全，带编译结果缓存）"""
    global _graph_compile_cache
    
    # 检查缓存
    if graph_id in _graph_compile_cache:
        logger.info(f"📦 使用缓存的状态图: {graph_id}")
        return _graph_compile_cache[graph_id]
    
    # 如果没有缓存，重新构建
    logger.info(f"🔨 构建新的状态图: {graph_id}")
    graph = _build_graph()
    checkpointer = await get_async_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    # 缓存编译结果
    _graph_compile_cache[graph_id] = compiled_graph
    logger.info(f"💾 状态图已缓存: {graph_id}")
    
    return compiled_graph


async def get_async_agent(graph_id: str = "default"):
    """获取异步 Agent 实例（单例，支持多图缓存）"""
    global _async_agent
    
    # 如果请求的是默认图，使用全局单例
    if graph_id == "default":
        async with _agent_lock:
            if _async_agent is None:
                _async_agent = await build_async_agent_graph(graph_id)
        return _async_agent
    
    # 非默认图，直接从缓存获取或构建
    return await build_async_agent_graph(graph_id)


def clear_graph_cache(graph_id: str = None):
    """清除状态图缓存"""
    global _graph_compile_cache, _async_agent
    
    if graph_id is None:
        # 清除所有缓存
        _graph_compile_cache = {}
        _async_agent = None
        logger.info("🗑️ 所有状态图缓存已清除")
    elif graph_id in _graph_compile_cache:
        del _graph_compile_cache[graph_id]
        if graph_id == "default":
            _async_agent = None
        logger.info(f"🗑️ 状态图缓存已清除: {graph_id}")


__all__ = [
    "AgentState",
    "get_async_agent",
    "initialize_experts",
    "clear_graph_cache",
]
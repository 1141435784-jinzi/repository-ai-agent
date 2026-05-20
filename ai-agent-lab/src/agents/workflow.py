"""
=== 企业级 Agent 工作流引擎 (RAG作为工具版) ===

【核心职责】：
1. 定义符合 OpenAI API 规范的状态机（AgentState）
2. 实现企业级工具调用流程（RAG作为工具之一）
3. 构建支持工具调用循环完整性的工作流
4. 提供异步 Agent 执行接口

【设计原则】：
1. RAG作为工具：按需检索，而非每次都检索
2. 专家自治：专家自己决定何时调用RAG工具
3. 合规性：严格遵循 LangGraph 工具调用规范
4. 可靠性：确保工具调用循环完整性，支持重试机制
5. 可观测性：完整的执行跟踪和错误处理

【工作流架构】：
START → memory → task_decomposition → supervisor → [agent] → should_continue
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
)
from src.memory import get_async_checkpointer, get_memory_manager
from src.llm.gateway import get_llm
from src.prompts import (
    AGENT_PROMPT, SUPERVISOR_PROMPT, PLAN_PROMPT,
    SIGHTS_PROMPT, FOOD_PROMPT, TRANSPORT_PROMPT
)
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
_graph_compile_cache = {}


# ============================================================
# State Reducer
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
# 状态定义
# ============================================================
class AgentState(TypedDict):
    """多 Agent 状态定义 - 支持多轮工具调用、Agent 协作、任务分解"""
    messages: Annotated[list, add_messages]
    trimmed_messages: Annotated[list, reduce_list]
    memory_context: Annotated[str, reduce_str]
    route: Annotated[str, reduce_str]
    tool_type: Annotated[str, reduce_str]
    tool_error: Annotated[str, reduce_str]
    tool_retry_count: Annotated[int, reduce_str]
    has_tool_calls: Annotated[bool, reduce_bool]
    collaboration_data: Annotated[dict, reduce_dict]
    current_agent: Annotated[str, reduce_str]
    agent_history: Annotated[list, reduce_list]
    needs_collaboration: Annotated[bool, reduce_bool]
    collaboration_target: Annotated[str, reduce_str]
    collaboration_reason: Annotated[str, reduce_str]
    execution_plan: Annotated[list, reduce_list]
    task_decomposition: Annotated[dict, reduce_dict]
    subtasks: Annotated[list, reduce_list]
    current_subtask: Annotated[int, reduce_str]
    reflection_notes: Annotated[list, reduce_list]
    key_decisions: Annotated[list, reduce_list]
    iteration_count: Annotated[int, reduce_str]


# ============================================================
# 初始化领域专家
# ============================================================
async def initialize_experts():
    """统一初始化所有领域专家"""
    print("🔄 正在初始化领域专家系统...")
    get_agent_tech_expert()
    get_plan_expert()
    get_sights_expert()
    get_food_expert()
    get_transport_expert()
    await agent_manager.initialize_all()
    print("✅ 领域专家系统初始化完成")


# ============================================================
# 节点实现
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

        route = resp.content.strip().lower()

        available_experts = ["agent_tech", "plan", "sights", "transport", "food"]
        if route not in available_experts:
            for expert in available_experts:
                if expert in route:
                    route = expert
                    break
            else:
                if "travel" in route or "finance" in route:
                    route = "plan"
                else:
                    route = "agent_tech"

        workflow_logger.logger.info(f"🔀 [{thread_id[:8]}] 路由决策: {route}")

        new_plan = []
        for t in plan:
            if t["agent"] == "supervisor" and t["status"] == "pending":
                new_plan.append({**t, "status": "completed"})
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


def _build_prompt_with_memory(state: AgentState, prompt_template, messages: list) -> list:
    """构建带记忆上下文的提示词"""
    prompt_messages = prompt_template.invoke({"messages": messages})

    memory_ctx = state.get("memory_context")
    if memory_ctx:
        sys_msg = prompt_messages.messages[0]
        prompt_messages.messages[0] = SystemMessage(
            content=f"{sys_msg.content}\n\n## 历史上下文\n{memory_ctx}"
        )

    return prompt_messages.messages


async def _build_agent_response(
    state: AgentState,
    config: RunnableConfig,
    prompt_template
) -> dict:
    """构建 Agent 响应（支持工具调用，包括RAG工具）"""
    thread_id = config["configurable"].get("thread_id", "default")
    user_model = config["configurable"].get("model", "")

    workflow_logger.node_enter("agent_response", thread_id)

    try:
        cleaned_msgs = state["messages"]

        plan = state.get("execution_plan", [])
        current_task = next((t["task"] for t in plan if t["status"] == "pending"), "处理用户请求")

        prompt_msgs = _build_prompt_with_memory(state, prompt_template, cleaned_msgs)

        task_msg = SystemMessage(
            content=f"【当前子任务指令】：{current_task}\n"
            "请仅针对此子任务进行回答或执行工具调用。"
        )
        prompt_msgs.insert(1, task_msg)

        tools = await tool_manager.get_tools()

        llm = get_llm(provider=user_model, streaming=True)
        workflow_logger.llm_call(thread_id, user_model or "default",
                                model=llm.model_name if hasattr(llm, 'model_name') else None,
                                prompt_tokens=sum(len(str(m.content)) for m in prompt_msgs))

        if tools:
            llm_with_tools = llm.bind_tools(tools)
            workflow_logger.tool_execution(thread_id, "bind_tools", {"tool_count": len(tools)})
        else:
            llm_with_tools = llm

        resp = await llm_with_tools.ainvoke(prompt_msgs)

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


def create_expert_node(agent_name: str, prompt_template) -> callable:
    """创建专家节点工厂函数 - 消除重复代码"""
    async def expert_node(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        workflow_logger.node_enter(agent_name, thread_id)

        plan = state.get("execution_plan", [])
        new_plan = [
            {**t, "status": "completed" if t["agent"] == agent_name and t["status"] == "pending" else t["status"]}
            for t in plan
        ]

        result = await _build_agent_response(state, config, prompt_template)
        workflow_logger.node_exit(agent_name, thread_id, "响应生成完成")
        return {**result, "current_agent": agent_name, "execution_plan": new_plan}

    return expert_node


# 使用工厂函数创建专家节点
agent_tech_node = create_expert_node("agent_tech", AGENT_PROMPT)
plan_node = create_expert_node("plan", PLAN_PROMPT)
food_agent_node = create_expert_node("food", FOOD_PROMPT)
sights_agent_node = create_expert_node("sights", SIGHTS_PROMPT)
transport_agent_node = create_expert_node("transport", TRANSPORT_PROMPT)


# ============================================================
# 任务分解节点
# ============================================================
async def task_decomposition_node(state: AgentState, config: RunnableConfig) -> dict:
    """任务分解节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("task_decomposition", thread_id)

    try:
        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, "content") else ""

        intent_keywords = ["天气", "景点", "美食", "餐厅", "预算", "花费", "交通", "酒店", "机票", "门票"]
        found_intents = [kw for kw in intent_keywords if kw in user_query]

        complexity_indicators = [
            len(found_intents) >= 2,
            len(user_query) > 60,
            "规划" in user_query or "安排" in user_query or "推荐" in user_query,
            "、" in user_query or "，" in user_query and len(user_query) > 30,
        ]

        if sum(complexity_indicators) >= 2 or len(found_intents) >= 3:
            llm = get_llm(provider="deepseek", streaming=True)

            decomposition_prompt = f"""
            请将以下用户请求分解为多个逻辑独立的子任务，并为每个子任务指定最合适的专家。

            用户请求：{user_query}

            可选专家：
            - plan: 旅行目的地推荐、行程规划、签证政策、预算精算
            - sights: 景点解说、门票政策、开放时间、游览路线
            - transport: 航班车次查询、交通方案对比、换乘指引
            - food: 菜品推荐、餐厅点评、订餐建议
            - agent_tech: AI技术问题、通用任务

            请输出JSON格式：
            {{
                "is_complex": true,
                "execution_plan": [
                    {{"id": 1, "task": "任务描述", "agent": "专家名", "status": "pending"}}
                ],
                "reason": "分解逻辑说明"
            }}
            """

            resp = await llm.ainvoke(decomposition_prompt)
            try:
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

        workflow_logger.node_exit("task_decomposition", thread_id, "简单任务，生成路由计划")
        return {
            "task_decomposition": {"is_complex": False},
            "execution_plan": [{"id": 1, "task": user_query, "agent": "supervisor", "status": "pending"}]
        }

    except Exception as e:
        workflow_logger.error(thread_id, "task_decomposition", e)
        return {"execution_plan": [{"id": 1, "task": "处理用户请求", "agent": "supervisor", "status": "pending"}]}


# ============================================================
# 协作决策节点
# ============================================================
def collaboration_decision_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent 协作决策节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("collaboration_decision", thread_id)

    try:
        last_message = state["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else ""

        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        if has_tool_calls:
            is_valid_tool_call = isinstance(last_message.tool_calls, list) and len(last_message.tool_calls) > 0
            if not is_valid_tool_call:
                is_valid_tool_call = last_message.tool_calls is not None

        collaboration_triggers = {
            "plan": ["规划", "计划", "安排", "旅游", "旅行", "预算", "费用", "价格"],
            "sights": ["景点", "景区", "风景", "观光"],
            "transport": ["交通", "高铁", "航班", "地铁", "公交", "机票", "车票"],
            "food": ["美食", "餐厅", "吃饭", "推荐菜", "特产"],
        }

        current_agent = state.get("current_agent", "")
        needs_collaboration = False
        target_agent = ""
        reason = ""

        for agent, triggers in collaboration_triggers.items():
            if agent != current_agent:
                if any(trigger in content for trigger in triggers):
                    needs_collaboration = True
                    target_agent = agent
                    reason = f"当前 Agent ({current_agent}) 需要 {agent} Agent 的专业知识"
                    break

        subtasks = state.get("subtasks", [])
        if subtasks and len(subtasks) > 1:
            needs_collaboration = True
            target_agent = "supervisor"
            reason = "多子任务需要协作处理"

        workflow_logger.logger.info(f"🤝 [{thread_id[:8]}] 协作决策: {needs_collaboration} -> {target_agent}, 工具调用: {has_tool_calls}")
        workflow_logger.node_exit("collaboration_decision", thread_id, f"协作需求: {needs_collaboration}, 目标: {target_agent}")

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


# ============================================================
# 反思节点
# ============================================================
def reflection_node(state: AgentState, config: RunnableConfig) -> dict:
    """反思节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("reflection", thread_id)

    try:
        messages = state["messages"]
        agent_history = state.get("agent_history", [])
        key_decisions = []

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

        trimmed_messages = messages
        if len(messages) > 15:
            trimmed_messages = messages[:2] + messages[-8:]
            workflow_logger.logger.info(f"✂️ [{thread_id[:8]}] 上下文剪枝: {len(messages)} -> {len(trimmed_messages)} 条消息")

        workflow_logger.node_exit("reflection", thread_id, "反思与剪枝完成")
        return {
            "key_decisions": key_decisions,
            "messages": trimmed_messages,
            "reflection_notes": [f"已执行步数: {len(agent_history)}"]
        }

    except Exception as e:
        workflow_logger.error(thread_id, "reflection", e)
        return {"key_decisions": [], "reflection_notes": []}


async def unified_tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """统一工具执行节点（重试机制由中间件处理）"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tools", thread_id)

    try:
        tools = await tool_manager.get_tools()
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的工具")
            return {"tool_error": "没有可用的工具"}

        tool_node = ToolNode(tools)
        result = await tool_node.ainvoke(state, config)

        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    success = not str(msg.content).startswith("Error")
                    workflow_logger.tool_result(thread_id, msg.name, success, str(msg.content)[:100])

        workflow_logger.node_exit("tools", thread_id, "执行完成")
        return result

    except Exception as e:
        workflow_logger.error(thread_id, "tools", e)
        return {"tool_error": str(e)[:150]}


async def tool_result_handler(state: AgentState, config: RunnableConfig) -> dict:
    """工具结果处理节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tool_result_handler", thread_id)

    try:
        tool_error = state.get("tool_error", "")
        if tool_error:
            retry_count = state.get("tool_retry_count", 0) + 1
            workflow_logger.node_exit("tool_result_handler", thread_id, f"工具报错，重试计数: {retry_count}")
            return {"tool_retry_count": retry_count}

        workflow_logger.node_exit("tool_result_handler", thread_id, "工具执行成功")
        return {"tool_retry_count": 0}

    except Exception as e:
        workflow_logger.error(thread_id, "tool_result_handler", e)
        return {"tool_error": str(e)[:150]}


async def self_healing_node(state: AgentState, config: RunnableConfig) -> dict:
    """自愈节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("self_healing", thread_id)

    try:
        retry_count = state.get("tool_retry_count", 0)
        if retry_count >= 3:
            workflow_logger.node_exit("self_healing", thread_id, "已达最大重试次数，放弃自愈")
            return {"messages": [AIMessage(content="抱歉，我多次尝试调用工具都失败了，可能暂时无法为您处理此请求。")]}

        last_msgs = state["messages"][-2:]
        error_info = state.get("tool_error", "未知错误")

        llm = get_llm(provider="deepseek", streaming=True)

        healing_prompt = f"""
        你是一个自愈专家。上一个工具调用失败了。

        工具调用信息：{last_msgs[0]}
        错误信息：{error_info}

        请分析原因并给出一个修正后的建议。
        请直接输出修复建议或说明，不要带多余废话。
        """

        resp = await llm.ainvoke(healing_prompt)
        workflow_logger.node_exit("self_healing", thread_id, "已生成修复策略")
        return {"messages": [AIMessage(content=f"🔧 自动尝试修复中: {resp.content}")]}

    except Exception as e:
        workflow_logger.error(thread_id, "self_healing", e)
        return {}


# ============================================================
# 路由决策
# ============================================================
def should_continue(state: AgentState) -> Literal["tool_selector", "summary", "execution_router_node"]:
    """决定是否继续工具调用循环"""
    last = state["messages"][-1]
    thread_id = state.get("route", "default")

    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= MAX_ITERATIONS:
        logger.info(f"🔄 [{thread_id[:8]}] 达到最大迭代次数 {MAX_ITERATIONS}，结束循环")
        return "summary"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_selector"

    plan = state.get("execution_plan", [])
    has_pending_tasks = any(t.get("status") == "pending" for t in plan)
    if has_pending_tasks:
        return "execution_router_node"

    return "summary"


async def summary_node(state: AgentState, config: RunnableConfig) -> dict:
    """最终总结节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("summary", thread_id)

    try:
        messages = state["messages"]
        plan = state.get("execution_plan", [])

        if plan and len(plan) > 1:
            last_human_idx = None
            for i, m in enumerate(messages):
                if isinstance(m, HumanMessage):
                    last_human_idx = i

            if last_human_idx is not None:
                ai_messages = [m for m in messages[last_human_idx+1:] if isinstance(m, AIMessage) and m.content and not m.tool_calls]
            else:
                ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content and not m.tool_calls]

            meaningful_msgs = []
            seen_content = set()
            for m in ai_messages:
                content = m.content.strip()
                if len(content) > 30 and content not in seen_content:
                    meaningful_msgs.append(m)
                    seen_content.add(content)

            if len(meaningful_msgs) > 1:
                combined_content = "\n\n".join([m.content for m in meaningful_msgs])
                workflow_logger.node_exit("summary", thread_id, f"合并了 {len(meaningful_msgs)} 条子任务回复")
                return {"messages": [AIMessage(content=combined_content)]}

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


def execution_list_router(state: AgentState) -> str:
    """根据执行清单决定下一个节点"""
    plan = state.get("execution_plan", [])

    next_task = next((t for t in plan if t["status"] == "pending"), None)

    if not next_task:
        return "reflection"

    agent_map = {
        "plan": "plan",
        "sights": "sights_agent",
        "transport": "transport_agent",
        "food": "food_agent",
        "agent_tech": "agent_tech",
        "supervisor": "supervisor"
    }

    target = agent_map.get(next_task["agent"], "agent_tech")
    return target


def collaboration_or_tool_decision(state: AgentState) -> str:
    """协作和工具调用决策路由"""
    last_msg = state["messages"][-1]

    tool_error = state.get("tool_error", "")
    retry_count = state.get("tool_retry_count", 0)
    if tool_error and retry_count > 0:
        return "self_healing"

    is_tool_call = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
    if is_tool_call:
        return "tool_selector"

    is_tool_result = isinstance(last_msg, ToolMessage) or (isinstance(last_msg, dict) and last_msg.get("type") == "tool")
    if is_tool_result:
        plan = state.get("execution_plan", [])
        has_pending_tasks = any(t.get("status") == "pending" for t in plan)
        if has_pending_tasks:
            return "execution_router_node"
        current_agent = state.get("current_agent") or "agent_tech"
        agent_node_map = {
            "agent_tech": "agent_tech",
            "plan": "plan",
            "sights": "sights_agent",
            "transport": "transport_agent",
            "food": "food_agent"
        }
        return agent_node_map.get(current_agent, "agent_tech")

    return "execution_router_node"


# ============================================================
# 构建流程图
# ============================================================
def _build_graph() -> StateGraph:
    """构建完整的 Agent 工作流图 (RAG作为工具版)"""
    g = StateGraph(AgentState)

    # 基础节点
    g.add_node("memory", memory_node)
    g.add_node("task_decomposition", task_decomposition_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("reflection", reflection_node)
    g.add_node("summary", summary_node)
    g.add_node("execution_router_node", lambda x: x)

    # 专家节点（每个专家内置RAG工具）
    g.add_node("agent_tech", agent_tech_node)
    g.add_node("plan", plan_node)
    g.add_node("sights_agent", sights_agent_node)
    g.add_node("transport_agent", transport_agent_node)
    g.add_node("food_agent", food_agent_node)

    # 工具与自愈
    g.add_node("tools", unified_tool_node)
    g.add_node("tool_result_handler", tool_result_handler)
    g.add_node("self_healing", self_healing_node)

    # 流程连线
    g.add_edge(START, "memory")
    g.add_edge("memory", "task_decomposition")
    g.add_edge("task_decomposition", "execution_router_node")

    # 执行清单动态路由（直接到专家节点，无需RAG节点）
    g.add_conditional_edges(
        "execution_router_node",
        execution_list_router,
        {
            "plan": "plan",
            "sights_agent": "sights_agent",
            "transport_agent": "transport_agent",
            "food_agent": "food_agent",
            "agent_tech": "agent_tech",
            "supervisor": "supervisor",
            "reflection": "reflection"
        }
    )

    # 专家执行完后进入决策
    for node in ["agent_tech", "plan", "sights_agent", "transport_agent", "food_agent", "supervisor"]:
        g.add_conditional_edges(
            node,
            collaboration_or_tool_decision,
            {
                "tool_selector": "tools",
                "self_healing": "self_healing",
                "execution_router_node": "execution_router_node",
                "agent_tech": "agent_tech",
                "plan": "plan",
                "sights_agent": "sights_agent",
                "transport_agent": "transport_agent",
                "food_agent": "food_agent"
            }
        )

    # 工具循环与自愈
    g.add_edge("tools", "tool_result_handler")
    g.add_conditional_edges(
        "tool_result_handler",
        collaboration_or_tool_decision,
        {
            "self_healing": "self_healing",
            "tool_selector": "tools",
            "execution_router_node": "execution_router_node",
            "agent_tech": "agent_tech",
            "plan": "plan",
            "sights_agent": "sights_agent",
            "transport_agent": "transport_agent",
            "food_agent": "food_agent"
        }
    )

    g.add_edge("self_healing", "execution_router_node")
    g.add_edge("reflection", "summary")
    g.add_edge("summary", END)

    return g


# ============================================================
# 异步图构建
# ============================================================
async def build_async_agent_graph(graph_id: str = "default"):
    """构建异步 Agent 图（线程安全，带编译结果缓存）"""
    global _graph_compile_cache

    if graph_id in _graph_compile_cache:
        logger.info(f"📦 使用缓存的状态图: {graph_id}")
        return _graph_compile_cache[graph_id]

    logger.info(f"🔨 构建新的状态图: {graph_id}")
    graph = _build_graph()
    checkpointer = await get_async_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    _graph_compile_cache[graph_id] = compiled_graph
    logger.info(f"💾 状态图已缓存: {graph_id}")

    return compiled_graph


async def get_async_agent(graph_id: str = "default"):
    """获取异步 Agent 实例"""
    global _async_agent

    if graph_id == "default":
        async with _agent_lock:
            if _async_agent is None:
                _async_agent = await build_async_agent_graph(graph_id)
        return _async_agent

    return await build_async_agent_graph(graph_id)


def clear_graph_cache(graph_id: str = None):
    """清除状态图缓存"""
    global _graph_compile_cache, _async_agent

    if graph_id is None:
        _graph_compile_cache = {}
        _async_agent = None
        logger.info("🗑️ 所有状态图缓存已清除")
    elif graph_id in _graph_compile_cache:
        del _graph_compile_cache[graph_id]
        if graph_id == "default":
            _async_agent = None
        logger.info(f"🗑️ 状态图缓存已清除: {graph_id}")

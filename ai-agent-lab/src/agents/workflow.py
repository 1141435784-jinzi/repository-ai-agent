"""
=== 企业级 Agent 工作流引擎 (增强版) ===

【核心职责】：
1. 定义符合 OpenAI API 规范的状态机（AgentState）
2. 实现企业级工具调用流程
3. 构建支持工具调用循环完整性的工作流
4. 提供异步 Agent 执行接口

【设计原则】：
1. RAG作为工具：按需检索
2. 专家自治：专家自己决定何时调用RAG工具
3. 星型拓扑：Supervisor作为唯一路由中心
4. 智能编排：支持复杂任务拆解、依赖编排、错误恢复

【工作流架构】：
START → supervisor → [expert_agent] → tools → supervisor → summary → END
"""

from typing import Annotated, List, Dict, Any, Optional, Literal
import asyncio
import logging
import json

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from src.config import MAX_ITERATIONS
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
from src.tools import tool_api
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
# 状态定义
# ============================================================
class AgentState(TypedDict):
    """增强版多 Agent 状态定义"""
    messages: Annotated[list, add_messages]
    memory_context: Annotated[str, lambda prev, next: next if next else prev]
    route: Annotated[str, lambda prev, next: next if next else prev]
    current_agent: Annotated[str, lambda prev, next: next if next else prev]
    execution_plan: Annotated[list, lambda prev, next: next if next else prev]
    iteration_count: Annotated[int, lambda prev, next: next if next else prev]
    task_errors: Annotated[list, lambda prev, next: next if next else prev]


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
# 工具函数 - 任务复杂度评估
# ============================================================
def _evaluate_task_complexity(user_query: str) -> int:
    """评估任务复杂度（1-5级）"""
    complexity = 1
    
    # 关键词复杂度
    intent_keywords = {
        "景点": 2, "美食": 2, "餐厅": 2, "预算": 2, "花费": 2,
        "交通": 2, "酒店": 2, "机票": 2, "门票": 2, "天气": 2,
        "规划": 3, "安排": 3, "推荐": 3, "行程": 3,
        "攻略": 3, "路线": 3, "旅行": 3, "旅游": 3
    }
    
    for keyword, score in intent_keywords.items():
        if keyword in user_query:
            complexity = max(complexity, score)
    
    # 长度复杂度
    if len(user_query) > 50:
        complexity += 1
    if len(user_query) > 100:
        complexity += 1
    
    # 多意图检测
    found_intents = [kw for kw in intent_keywords if kw in user_query]
    if len(found_intents) >= 2:
        complexity += 1
    if len(found_intents) >= 3:
        complexity += 1
    
    # 特殊符号检测
    if "、" in user_query or "，" in user_query:
        complexity += 1
    
    return min(complexity, 5)


# ============================================================
# 工具函数 - 任务拆解
# ============================================================
async def _decompose_task(user_query: str) -> list:
    """将复杂任务拆解为原子子任务"""
    llm = get_llm(provider="deepseek", streaming=False)
    
    decomposition_prompt = f"""
    请将以下用户请求分解为多个逻辑独立的原子子任务，并为每个子任务指定最合适的专家。

    用户请求：{user_query}

    可选专家及职责：
    - plan: 旅行目的地推荐、行程规划、签证政策、预算精算
    - sights: 景点解说、门票政策、开放时间、游览路线
    - transport: 航班车次查询、交通方案对比、换乘指引
    - food: 菜品推荐、餐厅点评、订餐建议
    - agent_tech: AI技术问题、通用任务、天气查询、其他未分类任务

    请输出JSON格式：
    {{
        "subtasks": [
            {{"id": 1, "task": "任务描述", "agent": "专家名", "dependencies": []}}
        ]
    }}
    """

    resp = await llm.ainvoke(decomposition_prompt)
    
    try:
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        result = json.loads(content)
        return result.get("subtasks", [])
    except Exception:
        # 解析失败，返回原始任务
        return [{"id": 1, "task": user_query, "agent": "agent_tech", "dependencies": []}]


# ============================================================
# 工具函数 - 专家路由映射
# ============================================================
def _get_expert_for_task(task: str) -> str:
    """根据任务内容自动匹配专家"""
    expert_mapping = {
        "景点": "sights",
        "景区": "sights", 
        "风景": "sights",
        "观光": "sights",
        "门票": "sights",
        "美食": "food",
        "餐厅": "food",
        "吃饭": "food",
        "推荐菜": "food",
        "特产": "food",
        "交通": "transport",
        "高铁": "transport",
        "航班": "transport",
        "地铁": "transport",
        "公交": "transport",
        "机票": "transport",
        "车票": "transport",
        "规划": "plan",
        "计划": "plan",
        "安排": "plan",
        "旅游": "plan",
        "旅行": "plan",
        "预算": "plan",
        "费用": "plan",
        "价格": "plan",
        "签证": "plan"
    }
    
    for keyword, expert in expert_mapping.items():
        if keyword in task:
            return expert
    
    # 默认使用 agent_tech
    return "agent_tech"


# ============================================================
# 核心节点实现
# ============================================================

async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """增强版监督者节点 - 完整任务管理能力"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("supervisor", thread_id)

    try:
        # 获取用户查询
        last_message = state["messages"][-1]
        user_query = last_message.content if hasattr(last_message, "content") else ""

        # 获取执行计划状态
        plan = state.get("execution_plan", [])
        completed_tasks = [t for t in plan if t.get("status") == "completed"]
        pending_tasks = [t for t in plan if t.get("status") == "pending"]
        in_progress_tasks = [t for t in plan if t.get("status") == "in_progress"]
        
        # 获取任务错误记录
        task_errors = state.get("task_errors", [])

        # 1️⃣ 如果没有执行计划，创建初始计划
        if not plan:
            workflow_logger.logger.info(f"🔍 [{thread_id[:8]}] 评估任务复杂度")
            
            # 评估复杂度
            complexity = _evaluate_task_complexity(user_query)
            workflow_logger.logger.info(f"⚡ [{thread_id[:8]}] 任务复杂度等级: {complexity}/5")
            
            # 根据复杂度决定是否拆解
            if complexity >= 3:
                # 复杂任务：拆解为多个子任务
                subtasks = await _decompose_task(user_query)
                plan = []
                for subtask in subtasks:
                    agent = subtask.get("agent") or _get_expert_for_task(subtask["task"])
                    plan.append({
                        "id": subtask["id"],
                        "task": subtask["task"],
                        "agent": agent,
                        "status": "pending",
                        "dependencies": subtask.get("dependencies", []),
                        "attempts": 0
                    })
                workflow_logger.logger.info(f"📋 [{thread_id[:8]}] 任务拆解完成: {len(plan)} 个子任务")
            else:
                # 简单任务：直接分配
                agent = _get_expert_for_task(user_query)
                plan = [{
                    "id": 1,
                    "task": user_query,
                    "agent": agent,
                    "status": "pending",
                    "dependencies": [],
                    "attempts": 0
                }]

        # 2️⃣ 判断是否需要总结（所有任务完成或达到最大迭代）
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= MAX_ITERATIONS:
            workflow_logger.node_exit("supervisor", thread_id, "决策：达到最大迭代次数，总结")
            return {"route": "summary", "execution_plan": plan}
        
        # 检查是否所有任务完成
        all_completed = len(completed_tasks) == len(plan)
        if all_completed:
            workflow_logger.node_exit("supervisor", thread_id, "决策：所有任务完成，总结")
            return {"route": "summary", "execution_plan": plan}

        # 3️⃣ 检查是否有任务需要纠错或补发
        if task_errors:
            last_error = task_errors[-1]
            task_id = last_error.get("task_id")
            error_type = last_error.get("type", "")
            
            # 找到出错的任务
            failed_task = next((t for t in plan if t["id"] == task_id), None)
            
            if failed_task and failed_task.get("attempts", 0) < 2:
                # 重试任务
                failed_task["attempts"] += 1
                failed_task["status"] = "pending"
                workflow_logger.logger.info(f"🔄 [{thread_id[:8]}] 任务 {task_id} 重试中，第 {failed_task['attempts']} 次尝试")
            elif failed_task:
                # 超过重试次数，降级到 agent_tech
                failed_task["agent"] = "agent_tech"
                failed_task["status"] = "pending"
                failed_task["attempts"] = 0
                workflow_logger.logger.info(f"🔀 [{thread_id[:8]}] 任务 {task_id} 降级到 agent_tech")

        # 4️⃣ 找到可以执行的下一个任务（考虑依赖）
        next_task = None
        for task in pending_tasks:
            # 检查依赖是否都已完成
            dependencies = task.get("dependencies", [])
            deps_completed = all(
                any(t["id"] == dep and t["status"] == "completed" for t in plan)
                for dep in dependencies
            )
            if deps_completed:
                next_task = task
                break
        
        # 如果没有考虑依赖的任务，取第一个pending任务
        if not next_task and pending_tasks:
            next_task = pending_tasks[0]

        # 5️⃣ 分配任务给专家
        if next_task:
            target_agent = next_task["agent"]
            
            # 标记任务为进行中
            new_plan = []
            for t in plan:
                if t["id"] == next_task["id"]:
                    new_plan.append({**t, "status": "in_progress"})
                else:
                    new_plan.append(t)

            workflow_logger.logger.info(f"🔀 [{thread_id[:8]}] 路由决策: {target_agent} (任务 {next_task['id']})")
            workflow_logger.node_exit("supervisor", thread_id, f"路由到: {target_agent}")
            return {
                "route": target_agent,
                "current_agent": target_agent,
                "execution_plan": new_plan
            }

        # 6️⃣ 默认路由到 agent_tech（通用专家）
        workflow_logger.node_exit("supervisor", thread_id, "路由到: agent_tech (默认)")
        return {"route": "agent_tech", "current_agent": "agent_tech", "execution_plan": plan}

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
    """构建 Agent 响应（支持工具调用）"""
    thread_id = config["configurable"].get("thread_id", "default")
    user_model = config["configurable"].get("model", "")

    workflow_logger.node_enter("agent_response", thread_id)

    try:
        cleaned_msgs = state["messages"]
        plan = state.get("execution_plan", [])
        current_task = next((t["task"] for t in plan if t["status"] == "in_progress"), "处理用户请求")

        prompt_msgs = _build_prompt_with_memory(state, prompt_template, cleaned_msgs)

        task_msg = SystemMessage(
            content=f"【当前任务】：{current_task}\n请针对此任务进行回答或执行工具调用。"
        )
        prompt_msgs.insert(1, task_msg)

        tools = tool_api.to_langchain_tools()
        llm = get_llm(provider=user_model, streaming=True)

        llm_with_tools = llm.bind_tools(tools) if tools else llm
        resp = await llm_with_tools.ainvoke(prompt_msgs)

        has_tool_calls = hasattr(resp, 'tool_calls') and resp.tool_calls
        if has_tool_calls:
            for tc in (resp.tool_calls if isinstance(resp.tool_calls, list) else [resp.tool_calls]):
                tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                workflow_logger.tool_execution(thread_id, tc_name, tc.get("args", {}) if isinstance(tc, dict) else {})

        workflow_logger.node_exit("agent_response", thread_id, f"响应长度: {len(resp.content) if resp.content else 0}")
        return {"messages": [resp]}

    except Exception as e:
        workflow_logger.error(thread_id, "agent_response", e)
        return {"messages": [AIMessage(content=f"服务异常，请稍后重试：{str(e)[:150]}")]}


def create_expert_node(agent_name: str, prompt_template) -> callable:
    """创建专家节点工厂函数"""
    async def expert_node(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        workflow_logger.node_enter(agent_name, thread_id)

        # 更新执行计划状态
        plan = state.get("execution_plan", [])
        new_plan = []
        for t in plan:
            if t.get("status") == "in_progress":
                # 检查是否有工具调用错误
                last_msg = state["messages"][-1] if state["messages"] else None
                has_error = False
                
                if last_msg:
                    if isinstance(last_msg, ToolMessage):
                        has_error = str(last_msg.content).startswith("Error")
                    elif hasattr(last_msg, 'content'):
                        has_error = "Error" in str(last_msg.content) or "error" in str(last_msg.content).lower()
                
                if has_error:
                    # 记录错误
                    task_errors = state.get("task_errors", [])
                    task_errors.append({
                        "task_id": t["id"],
                        "task": t["task"],
                        "agent": agent_name,
                        "type": "tool_error",
                        "message": str(last_msg.content) if hasattr(last_msg, 'content') else "Unknown error"
                    })
                    new_plan.append({**t, "status": "pending", "attempts": t.get("attempts", 0) + 1})
                else:
                    new_plan.append({**t, "status": "completed"})
            else:
                new_plan.append(t)

        result = await _build_agent_response(state, config, prompt_template)
        workflow_logger.node_exit(agent_name, thread_id, "响应生成完成")
        
        # 如果有错误，传递错误信息
        if "task_errors" in result:
            return {**result, "current_agent": agent_name, "execution_plan": new_plan}
        else:
            task_errors = state.get("task_errors", [])
            return {**result, "current_agent": agent_name, "execution_plan": new_plan, "task_errors": task_errors}

    return expert_node


# 创建专家节点（agent_tech 作为默认专家）
agent_tech_node = create_expert_node("agent_tech", AGENT_PROMPT)
plan_node = create_expert_node("plan", PLAN_PROMPT)
food_agent_node = create_expert_node("food", FOOD_PROMPT)
sights_agent_node = create_expert_node("sights", SIGHTS_PROMPT)
transport_agent_node = create_expert_node("transport", TRANSPORT_PROMPT)


# ============================================================
# 工具执行节点
# ============================================================
async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    """工具执行节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("tools", thread_id)

    try:
        tools = tool_api.to_langchain_tools()
        if not tools:
            logger.warning(f"[{thread_id[:8]}] 没有可用的工具")
            return {}

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
        return {}


# ============================================================
# 总结节点
# ============================================================
async def summary_node(state: AgentState, config: RunnableConfig) -> dict:
    """最终总结节点"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    workflow_logger.node_enter("summary", thread_id)

    try:
        messages = state["messages"]
        plan = state.get("execution_plan", [])
        
        # 如果有多个任务，尝试合并结果
        if plan and len(plan) > 1:
            ai_messages = []
            for m in messages:
                if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                    content = m.content.strip()
                    if len(content) > 30:
                        ai_messages.append(m)
            
            if len(ai_messages) > 1:
                combined_content = "\n\n".join([f"📌 {m.content}" for m in ai_messages])
                workflow_logger.node_exit("summary", thread_id, f"合并了 {len(ai_messages)} 条子任务回复")
                return {"messages": [AIMessage(content=combined_content)]}

        # 找最后一条有意义的回复
        last_meaningful_msg = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and not m.tool_calls),
            None
        )

        if last_meaningful_msg:
            workflow_logger.node_exit("summary", thread_id, "使用最终回复")
            return {"messages": [last_meaningful_msg]}

        summary_msg = AIMessage(content="感谢您的提问！如有其他问题，请随时告诉我。")
        workflow_logger.node_exit("summary", thread_id, "生成默认总结")
        return {"messages": [summary_msg]}

    except Exception as e:
        workflow_logger.error(thread_id, "summary", e)
        return {"messages": [AIMessage(content="抱歉，总结过程中出现错误。")]}


# ============================================================
# 路由决策函数
# ============================================================
def route_to_expert(state: AgentState) -> str:
    """路由到专家节点"""
    route = state.get("route", "agent_tech")
    
    route_map = {
        "plan": "plan",
        "sights": "sights_agent",
        "transport": "transport_agent",
        "food": "food_agent",
        "agent_tech": "agent_tech",
        "summary": "summary"
    }
    
    return route_map.get(route, "agent_tech")


def should_call_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """判断是否需要调用工具"""
    last_msg = state["messages"][-1]
    has_tool_calls = hasattr(last_msg, 'tool_calls') and last_msg.tool_calls
    
    if has_tool_calls:
        return "tools"
    
    if isinstance(last_msg, ToolMessage):
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        return "supervisor"
    
    return "supervisor"


# ============================================================
# 构建流程图
# ============================================================
def _build_graph() -> StateGraph:
    """构建增强版 Agent 工作流图"""
    g = StateGraph(AgentState)

    # 核心节点
    g.add_node("supervisor", supervisor_node)
    g.add_node("summary", summary_node)
    g.add_node("tools", tools_node)

    # 专家节点（agent_tech 作为默认专家）
    g.add_node("agent_tech", agent_tech_node)
    g.add_node("plan", plan_node)
    g.add_node("sights_agent", sights_agent_node)
    g.add_node("transport_agent", transport_agent_node)
    g.add_node("food_agent", food_agent_node)

    # 流程连线 - 星型拓扑
    g.add_edge(START, "supervisor")

    # Supervisor 路由到专家或总结
    g.add_conditional_edges(
        "supervisor",
        route_to_expert,
        {
            "plan": "plan",
            "sights_agent": "sights_agent",
            "transport_agent": "transport_agent",
            "food_agent": "food_agent",
            "agent_tech": "agent_tech",
            "summary": "summary"
        }
    )

    # 所有专家节点执行完后判断是否调用工具
    for expert_node in ["agent_tech", "plan", "sights_agent", "transport_agent", "food_agent"]:
        g.add_conditional_edges(
            expert_node,
            should_call_tools,
            {
                "tools": "tools",
                "supervisor": "supervisor"
            }
        )

    # 工具执行完返回 supervisor
    g.add_edge("tools", "supervisor")

    # 总结结束
    g.add_edge("summary", END)

    return g


# ============================================================
# 异步图构建
# ============================================================
async def build_async_agent_graph(*, config: dict | None = None):
    global _graph_compile_cache
    graph_id = "default"
    
    if graph_id in _graph_compile_cache:
        return _graph_compile_cache[graph_id]

    graph = _build_graph()
    checkpointer = await get_async_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    _graph_compile_cache[graph_id] = compiled_graph
    return compiled_graph


async def get_async_agent(graph_id: str = "default"):
    """获取异步执行图实例"""
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

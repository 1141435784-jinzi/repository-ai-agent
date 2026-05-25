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

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import Literal
import asyncio
import logging

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from src.config import MAX_ITERATIONS
from src.memory import get_async_checkpointer
from src.memory.short_term_memory import ShortTermMemoryManager
from src.memory.long_term_memory import LongTermMemoryManager
from src.llm.gateway import get_llm
from src.prompts import SUPERVISOR_PROMPT
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
# 自定义合并器：区分 None 和空列表/空字符串
def merge_with_empty(prev, next):
    """合并函数：允许设置为空列表或空字符串
    
    策略：
    - 如果 next 为 None，保留旧值 prev
    - 否则用 next 替换 prev（支持直接替换更新）
    - 适用于需要每次更新都保存最新值的场景（如 messages_context）
    """
    if next is None:
        return prev
    return next

class AgentState(TypedDict):
    """企业级多 Agent 状态定义"""
    from langchain_core.messages import BaseMessage
    
    # ========== 消息层 ==========
    # 原始对话历史（无限增长，仅用于审计和回溯）
    messages: Annotated[list[BaseMessage], add_messages]
    # 增强后的上下文消息列表，每次调用时重新构建
    # 包含：长期记忆(SystemMessage) + 短期记忆(SystemMessage) + 近N轮对话原文
    # 快照中保存的是当时实际发送给 LLM 的完整上下文
    messages_context: Annotated[list[BaseMessage], merge_with_empty]

    # ========== 任务层 ==========
    # 当前路由决策（下一个要执行的 Agent）
    route: Annotated[str, merge_with_empty]
    # 当前执行 Agent 名称
    current_agent: Annotated[str, merge_with_empty]
    # 任务执行计划（子任务列表及依赖关系）
    execution_plan: Annotated[list, merge_with_empty]
    # 任务执行历史（已完成任务的记录）
    task_history: Annotated[list, merge_with_empty]
    # 迭代计数（防止无限循环）
    iteration_count: Annotated[int, merge_with_empty]
    # 任务执行错误记录
    task_errors: Annotated[list, merge_with_empty]

    # ========== 系统层 ==========
    # 会话唯一标识（格式：thread_{user_id}_{timestamp}）
    thread_id: Annotated[str, merge_with_empty]
    # 用户标识
    user_id: Annotated[str, merge_with_empty]
    # 会话创建时间
    created_at: Annotated[str, merge_with_empty]
    # 最后更新时间
    last_updated: Annotated[str, merge_with_empty]
    # 工具调用历史（审计用）
    tool_call_history: Annotated[list, merge_with_empty]


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

            # 重新计算任务状态（因为 plan 已更新）
            completed_tasks = [t for t in plan if t.get("status") == "completed"]
            pending_tasks = [t for t in plan if t.get("status") == "pending"]
            in_progress_tasks = [t for t in plan if t.get("status") == "in_progress"]

        # 2️⃣ 判断是否需要总结（所有任务完成或达到最大迭代）
        iteration_count = state.get("iteration_count", 0) + 1
        if iteration_count >= MAX_ITERATIONS:
            workflow_logger.node_exit("supervisor", thread_id, "决策：达到最大迭代次数，总结")
            return {"route": "summary", "execution_plan": plan, "iteration_count": iteration_count}
        
        # 检查是否所有任务完成
        all_completed = len(completed_tasks) == len(plan)
        if all_completed:
            workflow_logger.node_exit("supervisor", thread_id, "决策：所有任务完成，总结")
            return {"route": "summary", "execution_plan": plan, "iteration_count": iteration_count}

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
                "execution_plan": new_plan,
                "iteration_count": iteration_count,
            }

        # 6️⃣ 默认路由到 agent_tech（通用专家）
        workflow_logger.node_exit("supervisor", thread_id, "路由到: agent_tech (默认)")
        return {
            "route": "agent_tech", 
            "current_agent": "agent_tech", 
            "execution_plan": plan,
            "iteration_count": iteration_count,
        }

    except Exception as e:
        workflow_logger.error(thread_id, "supervisor", e)
        return {"route": "agent_tech", "current_agent": "agent_tech", "iteration_count": iteration_count}

async def _build_agent_response(
    state: AgentState,
    config: RunnableConfig,
    agent_name: str
) -> dict:
    """构建 Agent 响应（支持工具调用）- 通过调用专家 Agent 的 process() 方法"""
    from langchain_core.messages import AIMessage

    thread_id = config["configurable"].get("thread_id", "default")

    workflow_logger.node_enter("agent_response", thread_id)

    try:
        # 获取压缩后的消息上下文（动态压缩）
        messages = state["messages"]
        
        # 加载并注入长期记忆上下文
        user_id = config.get("configurable", {}).get("user_id")
        long_term_memory_context = ""
        if user_id:
            try:
                long_term_memory = LongTermMemoryManager()
                user_query = str(messages[-1].content) if messages else ""
                long_term_memory_context = long_term_memory.build_memory_context(user_id, user_query)
            except Exception as e:
                logger.error(f"加载长期记忆失败: {e}")
        
        # 使用短期记忆管理器处理消息（动态压缩 + 语义检索）
        try:
            memory_manager = ShortTermMemoryManager()
            compressed_msgs = memory_manager.process_memory(
                messages=messages,
                thread_id=thread_id,
                current_query=str(messages[-1].content) if messages else "",
            )
        except Exception as e:
            logger.error(f"加载短期记忆失败: {e}")
            compressed_msgs = messages if messages else []
        
        # 注入长期记忆上下文到消息列表
        if long_term_memory_context:
            long_term_memory_message = SystemMessage(content=f"【长期记忆】\n{long_term_memory_context}")
            compressed_msgs = [long_term_memory_message] + compressed_msgs
        
        if not compressed_msgs:
            return {"messages": [AIMessage(content="未收到用户消息")]}

        # 获取当前任务
        plan = state.get("execution_plan", [])
        current_task = next((t["task"] for t in plan if t["status"] == "in_progress"), "处理用户请求")

        # 获取对应的专家 Agent 实例
        expert = agent_manager.get_agent(agent_name)
        if not expert:
            logger.warning(f"未找到专家 Agent: {agent_name}")
            return {"messages": [AIMessage(content=f"未找到专家: {agent_name}")]}

        # 调用专家的 process() 方法
        result = await expert.process(
            query=user_query,
            messages=compressed_msgs,
            config=config,
            context={
                "execution_plan": plan,
                "current_task": current_task
            }
        )

        # 处理工具调用
        needs_tool_execution = result.get("needs_tool_execution", False)
        tool_calls = result.get("tool_calls", [])
        
        if needs_tool_execution and tool_calls:
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                workflow_logger.tool_execution(thread_id, tc_name, tc.get("args", {}) if isinstance(tc, dict) else {})

        # 构建响应消息
        response_content = result.get("response", "")
        
        # 创建带工具调用的响应消息
        # 确保工具调用格式符合 LangChain 要求（必须包含 id 字段）
        from langchain_core.messages import ToolCall
        formatted_tool_calls = []
        for i, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                # 添加必要的 id 字段
                formatted_tool_calls.append(ToolCall(
                    id=str(i),
                    name=tc.get("name", ""),
                    args=tc.get("args", {})
                ))
            elif hasattr(tc, 'name') and hasattr(tc, 'args'):
                # 如果已经是 ToolCall 对象，确保有 id
                if hasattr(tc, 'id') and tc.id:
                    formatted_tool_calls.append(tc)
                else:
                    formatted_tool_calls.append(ToolCall(
                        id=str(i),
                        name=tc.name,
                        args=tc.args
                    ))
        
        resp = AIMessage(content=response_content)
        if formatted_tool_calls:
            resp.tool_calls = formatted_tool_calls

        workflow_logger.node_exit("agent_response", thread_id, f"响应长度: {len(response_content)}")
        return {
            "messages": [resp],
            "messages_context": compressed_msgs,
        }

    except Exception as e:
        workflow_logger.error(thread_id, "agent_response", e)
        return {"messages": [AIMessage(content=f"服务异常，请稍后重试：{str(e)[:150]}")]}


def create_expert_node(agent_name: str) -> callable:
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

        # 调用专家的 process() 方法获取响应
        result = await _build_agent_response(state, config, agent_name)
        workflow_logger.node_exit(agent_name, thread_id, "响应生成完成")
        
        # 如果有错误，传递错误信息
        if "task_errors" in result:
            return {**result, "current_agent": agent_name, "execution_plan": new_plan}
        else:
            task_errors = state.get("task_errors", [])
            return {**result, "current_agent": agent_name, "execution_plan": new_plan, "task_errors": task_errors}

    return expert_node


# 创建专家节点（agent_tech 作为默认专家）
agent_tech_node = create_expert_node("agent_tech")
plan_node = create_expert_node("plan")
food_agent_node = create_expert_node("food")
sights_agent_node = create_expert_node("sights")
transport_agent_node = create_expert_node("transport")


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
    """最终总结节点 - 仅在任务完成时清理状态"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    user_id = config.get("configurable", {}).get("user_id")
    workflow_logger.node_enter("summary", thread_id)

    try:
        messages = state["messages"]
        plan = state.get("execution_plan", [])
        
        # 保存对话到长期记忆
        if user_id:
            try:
                long_term_memory = LongTermMemoryManager()
                human_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
                ai_msgs = [m.content for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
                
                if human_msgs and ai_msgs:
                    last_human = human_msgs[-1]
                    last_ai = ai_msgs[-1]
                    long_term_memory.save_conversation_turn(user_id, last_human, last_ai)
                    logger.info(f"已保存对话到长期记忆: user_id={user_id}")
            except Exception as e:
                logger.error(f"保存长期记忆失败: {e}")
        
        # 检查是否所有任务都已完成
        all_tasks_completed = plan and all(t.get("status") == "completed" for t in plan)
        
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
                
                if all_tasks_completed:
                    return {
                        "messages": [AIMessage(content=combined_content)],
                        "execution_plan": [],
                        "iteration_count": 0,
                        "task_errors": [],
                        "route": "",
                        "current_agent": ""
                    }
                else:
                    return {"messages": [AIMessage(content=combined_content)]}

        # 找最后一条有意义的回复
        last_meaningful_msg = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content and not m.tool_calls),
            None
        )

        if last_meaningful_msg:
            workflow_logger.node_exit("summary", thread_id, "使用最终回复")
            
            if all_tasks_completed:
                return {
                    "messages": [last_meaningful_msg],
                    "execution_plan": [],
                    "iteration_count": 0,
                    "task_errors": [],
                    "route": "",
                    "current_agent": ""
                }
            else:
                return {"messages": [last_meaningful_msg]}

        summary_msg = AIMessage(content="感谢您的提问！如有其他问题，请随时告诉我。")
        workflow_logger.node_exit("summary", thread_id, "生成默认总结")
        
        if all_tasks_completed:
            return {
                "messages": [summary_msg],
                "execution_plan": [],
                "iteration_count": 0,
                "task_errors": [],
                "route": "",
                "current_agent": ""
            }
        else:
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
async def build_async_agent_graph(
    config: RunnableConfig | None = None
) -> StateGraph:
    # 你的构建逻辑
    graph = _build_graph()
    checkpointer = await get_async_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


async def get_async_graph(graph_id: str = "default"):
    """获取异步执行图实例"""
    global _async_agent

    if graph_id == "default":
        async with _agent_lock:
            if _async_agent is None:
                _async_agent = await build_async_agent_graph()
        return _async_agent

    return await build_async_agent_graph()


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

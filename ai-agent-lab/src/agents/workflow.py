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
    HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
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
from src.prompts import COMPLEXITY_EVALUATION_PROMPT, ROUTER_PROMPT, TASK_DECOMPOSITION_PROMPT
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
_system_initialized = False
skill_manager = None
mcp_manager = None


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
# 工具函数 - 任务复杂度评估（LLM驱动）
# ============================================================
async def _evaluate_task_complexity(user_query: str) -> int:
    """评估任务复杂度（1-5级）- 使用 LLM 进行语义级复杂度评估"""
    try:
        llm = get_llm(streaming=False)
        
        # 使用 COMPLEXITY_EVALUATION_PROMPT 构建提示词
        prompt = COMPLEXITY_EVALUATION_PROMPT.format(user_query=user_query)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        complexity = int(response.content.strip())
        return max(1, min(complexity, 5))  # 确保在 1-5 范围内
        
    except Exception as e:
        workflow_logger.logger.warning(f"LLM 复杂度评估失败，使用规则回退: {e}")
        # 回退到规则评估
        return _evaluate_task_complexity_fallback(user_query)

def _evaluate_task_complexity_fallback(user_query: str) -> int:
    """规则回退版本的复杂度评估"""
    complexity = 1
    
    intent_keywords = {
        "景点": 2, "美食": 2, "餐厅": 2, "预算": 2, "花费": 2,
        "交通": 2, "酒店": 2, "机票": 2, "门票": 2, "天气": 2,
        "规划": 3, "安排": 3, "推荐": 3, "行程": 3,
        "攻略": 3, "路线": 3, "旅行": 3, "旅游": 3
    }
    
    for keyword, score in intent_keywords.items():
        if keyword in user_query:
            complexity = max(complexity, score)
    
    if len(user_query) > 50:
        complexity += 1
    if len(user_query) > 100:
        complexity += 1
    
    found_intents = [kw for kw in intent_keywords if kw in user_query]
    if len(found_intents) >= 2:
        complexity += 1
    if len(found_intents) >= 3:
        complexity += 1
    
    if "、" in user_query or "，" in user_query:
        complexity += 1
    
    return min(complexity, 5)


# ============================================================
# 工具函数 - 任务拆解
# ============================================================
async def _decompose_task(user_query: str) -> list:
    """将复杂任务拆解为原子子任务"""
    llm = get_llm(provider="deepseek", streaming=False)
    
    # 使用统一管理的任务拆解提示词
    decomposition_prompt = TASK_DECOMPOSITION_PROMPT.format(user_query=user_query)

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
# 工具函数 - 专家路由映射（LLM驱动）
# ============================================================
async def _get_expert_for_task(task: str) -> str:
    """使用 LLM 进行语义级意图识别，匹配最合适的专家"""
    try:
        llm = get_llm(streaming=False)
        
        # 动态获取已注册的专家列表
        agents = agent_manager.list_agents()
        if not agents:
            return _get_expert_for_task_fallback(task)
        
        # 构建专家列表描述
        agent_descriptions = []
        for agent in agents:
            capabilities = ", ".join(agent.get("capabilities", [])[:3])
            agent_descriptions.append(f"- {agent['name']}: {agent['description']}")
        
        # 使用 ROUTER_PROMPT 进行意图识别
        prompt = ROUTER_PROMPT.format(
            agent_list="\n".join(agent_descriptions),
            user_query=task
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        selected_agent = response.content.strip().lower()
        
        # 验证返回的专家是否存在
        valid_agents = [a["name"].lower() for a in agents]
        if selected_agent in valid_agents:
            return selected_agent
        
        # 如果返回的专家不存在，使用回退
        workflow_logger.logger.warning(f"LLM 返回的专家 '{selected_agent}' 不存在，使用回退")
        return _get_expert_for_task_fallback(task)
        
    except Exception as e:
        workflow_logger.logger.warning(f"LLM 意图识别失败，使用规则回退: {e}")
        return _get_expert_for_task_fallback(task)

def _get_expert_for_task_fallback(task: str) -> str:
    """规则回退版本：根据关键词匹配专家"""
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
# 工具函数 - 对话历史分析
# ============================================================
def _analyze_conversation_context(messages: list) -> dict:
    """分析对话历史上下文，提取有用特征"""
    context = {
        "turn_count": 0,
        "recent_tool_failures": 0,
        "is_repetitive": False,
        "has_pending_tool_call": False,
        "context_summary": ""
    }
    
    if not messages:
        return context
    
    # 统计对话轮数（只计算用户和AI的消息）
    user_messages = [m for m in messages if hasattr(m, 'type') and m.type == 'human']
    context["turn_count"] = len(user_messages)
    
    # 检查最近的工具调用失败
    recent_messages = messages[-5:]  # 最近5条消息
    for msg in recent_messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # 检查工具调用是否失败
            for tool_call in msg.tool_calls:
                if hasattr(tool_call, 'status') and tool_call.status == 'error':
                    context["recent_tool_failures"] += 1
    
    # 检测重复提问模式
    if len(user_messages) >= 2:
        recent_queries = [m.content[:50] for m in user_messages[-3:]]
        # 如果最近3个查询中有重复
        if len(recent_queries) != len(set(recent_queries)):
            context["is_repetitive"] = True
    
    # 检查是否有待处理的工具调用
    for msg in messages[-3:]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # 检查是否有未完成的工具调用
            for tool_call in msg.tool_calls:
                if not hasattr(tool_call, 'status') or tool_call.status != 'success':
                    context["has_pending_tool_call"] = True
                    break
    
    # 生成上下文摘要（最近3条消息）
    context_snippets = []
    for m in messages[-3:]:
        if hasattr(m, 'content'):
            content = m.content[:30] + "..." if len(m.content) > 30 else m.content
            context_snippets.append(f"{m.type}: {content}")
    context["context_summary"] = "\n".join(context_snippets)
    
    return context

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
        
        # ============= 增强对话历史利用 =============
        # 获取完整对话历史
        messages = state.get("messages", [])
        
        # 分析历史对话特征
        conversation_context = _analyze_conversation_context(messages)
        
        # 检查是否有重复提问模式（用户反复询问相同问题）
        if conversation_context.get("is_repetitive"):
            workflow_logger.logger.warning(f"🔄 [{thread_id[:8]}] 检测到重复提问模式")
        
        # 检查历史中是否有工具调用失败记录
        if conversation_context.get("recent_tool_failures") > 0:
            workflow_logger.logger.warning(f"⚠️ [{thread_id[:8]}] 检测到 {conversation_context['recent_tool_failures']} 次工具调用失败")
        
        # 检查对话轮数
        if conversation_context.get("turn_count") > 3:
            workflow_logger.logger.info(f"📢 [{thread_id[:8]}] 对话已进行 {conversation_context['turn_count']} 轮")

        # 1️⃣ 如果没有执行计划，创建初始计划
        if not plan:
            workflow_logger.logger.info(f"🔍 [{thread_id[:8]}] 评估任务复杂度")
            
            # 评估复杂度
            complexity = await _evaluate_task_complexity(user_query)
            workflow_logger.logger.info(f"⚡ [{thread_id[:8]}] 任务复杂度等级: {complexity}/5")
            
            # 根据复杂度决定是否拆解
            if complexity >= 3:
                # 复杂任务：拆解为多个子任务
                subtasks = await _decompose_task(user_query)
                plan = []
                for subtask in subtasks:
                    agent = subtask.get("agent") or await _get_expert_for_task(subtask["task"])
                    plan.append({
                        "id": subtask["id"],
                        "task": subtask["task"],
                        "agent": agent,
                        "status": "pending",
                        "dependencies": subtask.get("dependencies", []),
                        "attempts": 0,
                        "priority": subtask.get("priority", "medium")  # 新增优先级字段
                    })
                workflow_logger.logger.info(f"📋 [{thread_id[:8]}] 任务拆解完成: {len(plan)} 个子任务")
            else:
                # 简单任务：直接分配
                agent = await _get_expert_for_task(user_query)
                plan = [{
                    "id": 1,
                    "task": user_query,
                    "agent": agent,
                    "status": "pending",
                    "dependencies": [],
                    "attempts": 0,
                    "priority": "medium"  # 默认中等优先级
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

        # 4️⃣ 找到可以执行的下一个任务（考虑依赖和优先级）
        next_task = None
        
        # 筛选出依赖已完成的任务
        ready_tasks = []
        for task in pending_tasks:
            dependencies = task.get("dependencies", [])
            deps_completed = all(
                any(t["id"] == dep and t["status"] == "completed" for t in plan)
                for dep in dependencies
            )
            if deps_completed:
                ready_tasks.append(task)
        
        # 按优先级排序：high > medium > low
        if ready_tasks:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            ready_tasks.sort(key=lambda t: priority_order.get(t.get("priority", "medium")))
            next_task = ready_tasks[0]
            workflow_logger.logger.info(f"🎯 [{thread_id[:8]}] 选择高优先级任务: {next_task['id']} (优先级: {next_task.get('priority')})")
        
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
        
        # 获取用户查询
        user_query = str(messages[-1].content) if messages else ""
        
        # 加载并注入长期记忆上下文
        user_id = config.get("configurable", {}).get("user_id")
        long_term_memory_context = ""
        if user_id:
            try:
                long_term_memory = LongTermMemoryManager()
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
def _extract_user_text(state: AgentState) -> str:
    """从 state 中提取用户消息文本"""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("type") == "human":
            content_list = msg.get("content", [])
            if content_list and isinstance(content_list, list):
                return content_list[0].get("text", "")
    return ""

def _build_agent_routing_prompt(user_text: str, agents: List[Dict[str, Any]]) -> str:
    """构建 LLM 路由提示词"""
    if not agents:
        return ""

    agent_descriptions = []
    for agent in agents:
        capabilities = ", ".join(agent.get("capabilities", [])[:5])
        agent_descriptions.append(
            f"- {agent['name']}: {agent['description']} (能力: {capabilities})"
        )

    # 使用 ChatPromptTemplate 的 format 方法构建提示词
    return ROUTER_PROMPT.format(
        agent_list="\n".join(agent_descriptions),
        user_query=user_text
    )

def _map_agent_name_to_node(agent_name: str) -> str:
    """将 agent 名称映射到图节点名称"""
    mapping = {
        "plan": "plan",
        "sights": "sights_agent",
        "transport": "transport_agent",
        "food": "food_agent",
        "agent_tech": "agent_tech",
        "summary": "summary"
    }
    return mapping.get(agent_name, "agent_tech")

async def route_to_expert(state: AgentState) -> str:
    """
    LLM-Based 动态路由：从用户消息识别意图 → 动态路由到对应专家

    相比硬编码的关键词匹配，LLM 路由的优势：
    1. 动态扩展：新注册 agent 无需修改路由逻辑
    2. 语义理解：能理解同义词和上下文
    3. 多意图处理：能判断主要意图和次要意图
    """
    user_text = _extract_user_text(state)

    agents = agent_manager.list_agents()
    if not agents:
        workflow_logger.warning("router", "No agents registered, using fallback")
        return "agent_tech"

    routing_prompt = _build_agent_routing_prompt(user_text, agents)

    try:
        llm = get_llm(streaming=False)
        response = await llm.ainvoke([HumanMessage(content=routing_prompt)])
        selected_agent = response.content.strip().lower()

        if selected_agent not in [a["name"] for a in agents]:
            if "summary" in selected_agent:
                selected_agent = "summary"
            elif any(kw in user_text.lower() for kw in ["总结", "汇总", "整理"]):
                selected_agent = "summary"
            else:
                selected_agent = "agent_tech"

        node_name = _map_agent_name_to_node(selected_agent)
        workflow_logger.info("router", f"LLM routed '{user_text[:30]}...' → {node_name}")
        return node_name

    except Exception as e:
        workflow_logger.error("router", f"LLM routing failed: {e}, using keyword fallback")
        return _keyword_fallback_route(user_text)

def _keyword_fallback_route(text: str) -> str:
    """关键词回退路由 - 当 LLM 不可用时的备选方案"""
    text = text.strip().lower()

    if any(key in text for key in ["规划", "计划", "攻略", "行程", "安排", "玩几天", "怎么玩", "路线", "旅游", "旅行", "逛一逛", "游玩", "日程"]):
        return "plan"
    elif any(key in text for key in ["景点", "景区", "去哪玩", "好去处", "打卡", "好玩", "必去", "名胜", "公园", "海边", "观景"]):
        return "sights_agent"
    elif any(key in text for key in ["交通", "机票", "高铁", "火车", "动车", "怎么去", "打车", "地铁", "公交", "航班", "路线"]):
        return "transport_agent"
    elif any(key in text for key in ["美食", "吃什么", "餐厅", "好吃", "小吃", "特产", "粤菜", "早茶", "宵夜", "推荐菜"]):
        return "food_agent"
    elif any(key in text for key in ["总结", "汇总", "整理"]):
        return "summary"
    else:
        return "agent_tech"

def should_call_tools(state: AgentState) -> Literal["tools", "supervisor"]:
    """判断是否需要调用工具"""
    last_msg = state["messages"][-1]
    has_tool_calls = hasattr(last_msg, 'tool_calls') and last_msg.tool_calls
    
    if has_tool_calls:
        return "tools"
    
    return "supervisor"


# ============================================================
# 系统初始化
# ============================================================
async def _initialize_components():
    """初始化系统组件（Skill管理器、MCP管理器、专家系统），防止重复初始化"""
    global skill_manager, mcp_manager, _system_initialized

    if _system_initialized:
        return
    try:
        logger.info("正在初始化 Skill 管理器...")
        from src.tools.skills import SkillManager
        skill_manager = SkillManager()
        manager_skill_count = skill_manager.initialize()
        logger.info(f"✅ 已加载 {manager_skill_count} 个 Skill")
    except Exception as e:
        logger.warning(f"Skill 管理器初始化失败：{e}")
        logger.warning("Skill 功能可能不可用，但服务将继续运行")
    
    try:
        logger.info("正在初始化 MCP 管理器...")
        from src.tools import get_mcp_manager
        mcp_manager = await get_mcp_manager()
        logger.info("✅ MCP 管理器初始化成功")
    except Exception as e:
        logger.warning(f"MCP 管理器初始化失败：{e}")
        logger.warning("MCP 功能可能不可用，但服务将继续运行")

    await initialize_experts()

    _system_initialized = True


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
    await _initialize_components()
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

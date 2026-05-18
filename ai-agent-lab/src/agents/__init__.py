"""
=== Agent 系统 ===

【功能】：
1. 领域专家 Agent：专注于特定专业领域的技术助手
2. Agent 管理器：统一管理所有 Agent 实例
3. 工作流引擎：基于 LangGraph 的多 Agent 协作工作流
4. 工具管理器：统一管理 MCP、API、本地工具

【架构】：
用户输入 → Supervisor（意图分类）
    ├── "agent_tech"  → AI Agent 开发专家（RAG + LLM + 工具）
    ├── "plan"        → 规划专家（整合旅行规划与财务预算）（RAG + LLM + 工具）
    ├── "sights"      → 景点推荐专家（RAG + LLM + 工具）
    ├── "food"        → 美食推荐专家（RAG + LLM + 工具）
    └── "transport"   → 交通出行专家（RAG + LLM + 工具）

【模块】：
- agent_base.py: 领域专家 Agent 基类和全局管理器
- workflow.py: LangGraph 工作流引擎和 Supervisor 路由
- tool_manager.py: 动态工具管理器
- experts/: 专业领域 Agent 实现
    - agent_tech.py: AI Agent 开发专家
    - agent_plan.py: 规划专家（整合旅行规划与财务预算）
    - agent_sights.py: 景点推荐专家
    - agent_food.py: 美食推荐专家
    - agent_transport.py: 交通出行专家
"""

# ============================================================
# Agent 基类和管理器
# ============================================================
from .experts.base import (
    DomainExpertAgent,
    AgentManager,
    agent_manager,
)

# ============================================================
# 工作流引擎
# ============================================================
from .workflow import (
    AgentState,
    get_async_agent,
    initialize_experts,
)

# ============================================================
# 工具管理器
# ============================================================
from src.tools.tool_manager import (
    DynamicToolManager,
    tool_manager,
    get_tool_system_stats,
    refresh_tool_system,
    get_tool_info,
)

# ============================================================
# 专业领域 Agent
# ============================================================
from .experts.agent_tech import (
    AgentTechExpert,
    get_agent_tech_expert,
)

from .experts.agent_sights import (
    SightsExpert,
    get_sights_expert,
)

from .experts.agent_food import (
    FoodExpert,
    get_food_expert,
)

from .experts.agent_transport import (
    TransportExpert,
    get_transport_expert,
)

from .experts.agent_plan import (
    PlanExpert,
    get_plan_expert,
)


# ============================================================
# 便捷函数
# ============================================================
def get_expert_rag_engine(expert_name: str):
    """获取指定专家的RAG引擎
    
    Args:
        expert_name: 专家名称 (agent_tech, plan, sights, food, transport)
        
    Returns:
        RAGEngine: 专家对应的RAG引擎实例，用于增量更新
        None: 如果专家不存在或未初始化
    """
    expert = agent_manager.get_agent(expert_name)
    if expert and hasattr(expert, 'get_rag_engine'):
        return expert.get_rag_engine()
    return None


# ============================================================
# 导出列表
# ============================================================
__all__ = [
    # Agent 基类和管理器
    "DomainExpertAgent",
    "AgentManager",
    "agent_manager",
    
    # 工作流引擎
    "AgentState",
    "get_async_agent",
    "initialize_experts",
    
    # 工具管理器
    "DynamicToolManager",
    "tool_manager",
    
    # 工具管理 API
    "get_tool_system_stats",
    "refresh_tool_system",
    "get_tool_info",
    
    # 专业领域 Agent
    "AgentTechExpert",
    "get_agent_tech_expert",
    "PlanExpert",
    "get_plan_expert",
    "SightsExpert",
    "get_sights_expert",
    "FoodExpert",
    "get_food_expert",
    "TransportExpert",
    "get_transport_expert",
    
    # 便捷函数
    "get_expert_rag_engine",
]
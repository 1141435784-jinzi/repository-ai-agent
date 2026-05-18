"""
=== 专业 Agent 模块 ===

【功能】：
1. Agent 技术助手：基于 Agent 开发知识库
2. 规划专家：基于 Plan 知识库（整合旅行规划与财务预算）
3. 景点推荐专家：基于城市景点知识库
4. 美食推荐专家：基于美食知识库
5. 交通出行专家：基于交通知识库

【设计原则】：
1. 每个专业 Agent 专注于特定领域
2. 可以有自己的 RAG 知识库
3. 可以有自己的工具集
4. 支持多 Agent 协作和工具交叉调用
"""

# 导出专业 Agent
from .agent_tech import AgentTechExpert, get_agent_tech_expert
from .agent_sights import SightsExpert, get_sights_expert
from .agent_food import FoodExpert, get_food_expert
from .agent_transport import TransportExpert, get_transport_expert
from .agent_plan import PlanExpert, get_plan_expert
from .base import DomainExpertAgent, agent_manager

__all__ = [
    "AgentTechExpert",
    "get_agent_tech_expert",
    "SightsExpert",
    "get_sights_expert",
    "FoodExpert",
    "get_food_expert",
    "TransportExpert",
    "get_transport_expert",
    "PlanExpert",
    "get_plan_expert",
    "DomainExpertAgent",
    "agent_manager",
]
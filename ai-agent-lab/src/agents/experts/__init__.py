"""
=== 专业 Agent 模块 ===

【功能】：
1. Agent 技术助手：基于 Agent 开发知识库
2. 景点推荐专家：基于城市景点知识库
3. 美食推荐专家：基于美食知识库
4. 交通出行专家：基于交通知识库
5. 财务规划专家：基于财务知识库

【设计原则】：
1. 每个专业 Agent 专注于特定领域
2. 可以有自己的 RAG 知识库
3. 可以有自己的工具集
4. 支持多 Agent 协作和工具交叉调用
"""

# 导出专业 Agent
from .agent_tech import TechSpecialist, get_tech_specialist
from .agent_sights import SightsSpecialist, get_sights_specialist
from .agent_food import FoodSpecialist, get_food_specialist
from .agent_transport import TransportSpecialist, get_transport_specialist
from .agent_finance import FinanceSpecialist, get_finance_specialist
from .base import DomainSpecialistAgent, agent_manager

__all__ = [
    "TechSpecialist",
    "get_tech_specialist",
    "SightsSpecialist",
    "get_sights_specialist",
    "FoodSpecialist",
    "get_food_specialist",
    "TransportSpecialist",
    "get_transport_specialist",
    "FinanceSpecialist",
    "get_finance_specialist",
    "DomainSpecialistAgent",
    "agent_manager",
]
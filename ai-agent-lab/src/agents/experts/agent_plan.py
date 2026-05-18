"""
=== 规划专家 ===

【功能】：
1. 基于 Plan 知识库回答旅行规划与财务预算问题
2. 使用 RAG 检索相关知识
3. 专注于旅行目的地推荐、行程安排及预算精算
4. 支持工具调用（汇率查询、预算计算器、天气等）
"""

from src.prompts import PLAN_PROMPT
from src.config import KNOWLEDGE_BASE_PLAN_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class PlanExpert(DomainExpertAgent):
    """规划专家 - 继承自领域专家基类，整合旅行规划与财务预算功能"""
    
    def __init__(self):
        """初始化规划专家"""
        super().__init__(
            name="plan",
            description="规划专家",
            capabilities=[
                # 旅行规划能力
                "整体旅行目的地推荐",
                "跨城市/多天大行程规划",
                "签证与出入境政策咨询",
                "目的地文化概览与综合安全",
                "酒店与票务预订流程指导",
                # 财务预算能力
                "详细旅行费用估算与精算",
                "多币种汇率实时查询与换算",
                "旅行保险方案对比与建议",
                "旅行开支优化与省钱攻略",
                "支付安全与防财务诈骗提示",
                "旅行预算法规与财务风险评估"
            ],
            knowledge_dir=KNOWLEDGE_BASE_PLAN_DIR,
            collection_name="plan_knowledge",
            prompt_template=PLAN_PROMPT,
            domain_metadata={
                "domain": "travel_planning_and_finance",
                "response_type": "planning_and_budget_advice"
            }
        )


# 全局规划专家实例
_plan_expert = None


def get_plan_expert() -> PlanExpert:
    """获取全局规划专家实例"""
    global _plan_expert
    if _plan_expert is None:
        _plan_expert = PlanExpert()
        agent_manager.register_agent(_plan_expert)
    return _plan_expert


__all__ = [
    "PlanExpert",
    "get_plan_expert",
]
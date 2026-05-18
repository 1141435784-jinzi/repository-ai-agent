"""
=== 美食推荐专家 ===

【功能】：
1. 基于 Food 知识库回答美食相关问题
2. 使用 RAG 检索相关知识
3. 专注于美食推荐和餐厅介绍
4. 支持工具调用（地图导航、价格查询等）
"""

from src.prompts import FOOD_PROMPT
from src.config import KNOWLEDGE_BASE_FOOD_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class FoodExpert(DomainExpertAgent):
    """美食推荐专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化美食推荐专家"""
        super().__init__(
            name="food",
            description="美食推荐专家",
            capabilities=[
                "具体特色菜品与地道小吃推荐",
                "知名餐厅与网红店深度点评",
                "美食历史文化与就餐礼仪",
                "餐厅定位建议与预订流程说明",
                "特色美食街区探店路线规划",
                "根据口味偏好与消费档次进行筛选"
            ],
            knowledge_dir=KNOWLEDGE_BASE_FOOD_DIR,
            collection_name="food_knowledge",
            prompt_template=FOOD_PROMPT,
            domain_metadata={
                "domain": "food_recommendation",
                "response_type": "food_advice"
            }
        )


# 全局美食推荐专家实例
_food_expert = None


def get_food_expert() -> FoodExpert:
    """获取全局美食推荐专家实例"""
    global _food_expert
    if _food_expert is None:
        _food_expert = FoodExpert()
        agent_manager.register_agent(_food_expert)
    return _food_expert


__all__ = [
    "FoodExpert",
    "get_food_expert",
]

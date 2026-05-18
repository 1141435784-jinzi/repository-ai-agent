"""
=== 景点推荐专家 ===

【功能】：
1. 基于 Sights 知识库回答景点相关问题
2. 使用 RAG 检索相关知识
3. 专注于城市景点介绍和旅游攻略
4. 支持工具调用（地图导航、天气查询等）
"""

from src.prompts import SIGHTS_PROMPT
from src.config import KNOWLEDGE_BASE_SIGHTS_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class SightsExpert(DomainExpertAgent):
    """景点推荐专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化景点推荐专家"""
        super().__init__(
            name="sights",
            description="城市景点推荐专家",
            capabilities=[
                "景点深度介绍与历史文化解说",
                "景点门票价格、政策与预订提示",
                "景点开放时间与最佳游览季节建议",
                "景区内详细游玩路径与打卡点推荐",
                "景点周边配套设施（洗手间、寄存等）查询",
                "避坑指南与游览注意事项"
            ],
            knowledge_dir=KNOWLEDGE_BASE_SIGHTS_DIR,
            collection_name="sights_knowledge",
            prompt_template=SIGHTS_PROMPT,
            domain_metadata={
                "domain": "sightseeing",
                "response_type": "sight_advice"
            }
        )


# 全局景点推荐专家实例
_sights_expert = None


def get_sights_expert() -> SightsExpert:
    """获取全局景点推荐专家实例"""
    global _sights_expert
    if _sights_expert is None:
        _sights_expert = SightsExpert()
        agent_manager.register_agent(_sights_expert)
    return _sights_expert


__all__ = [
    "SightsExpert",
    "get_sights_expert",
]

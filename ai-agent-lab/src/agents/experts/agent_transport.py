"""
=== 交通出行专家 ===

【功能】：
1. 基于 Transport 知识库回答交通相关问题
2. 使用 RAG 检索相关知识
3. 专注于交通出行规划和票务信息
4. 支持工具调用（航班查询、火车票查询等）
"""

from src.prompts import TRANSPORT_PROMPT
from src.config import KNOWLEDGE_BASE_TRANSPORT_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class TransportExpert(DomainExpertAgent):
    """交通出行专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化交通出行专家"""
        super().__init__(
            name="transport",
            description="交通出行专家",
            capabilities=[
                "具体航班、车次实时动态查询",
                "跨城交通方式方案对比与选择建议",
                "城市内地铁/公交换乘方案指引",
                "机场、火车站与酒店间的接驳指南",
                "出行耗时精准预估与避峰建议",
                "交通票务政策与行李规定说明"
            ],
            knowledge_dir=KNOWLEDGE_BASE_TRANSPORT_DIR,
            collection_name="transport_knowledge",
            prompt_template=TRANSPORT_PROMPT,
            domain_metadata={
                "domain": "transportation",
                "response_type": "transport_advice"
            }
        )


# 全局交通出行专家实例
_transport_expert = None


def get_transport_expert() -> TransportExpert:
    """获取全局交通出行专家实例"""
    global _transport_expert
    if _transport_expert is None:
        _transport_expert = TransportExpert()
        agent_manager.register_agent(_transport_expert)
    return _transport_expert


__all__ = [
    "TransportExpert",
    "get_transport_expert",
]

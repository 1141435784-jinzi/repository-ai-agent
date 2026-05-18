"""
=== AI Agent 开发专家 ===

【功能】：
1. 基于 Agent 开发知识库回答技术问题
2. 使用 RAG 检索相关知识
3. 专注于 AI Agent 开发领域
4. 支持工具调用（计算、搜索等辅助工具）
"""

from src.prompts import AGENT_PROMPT
from src.config import KNOWLEDGE_BASE_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class AgentTechExpert(DomainExpertAgent):
    """AI Agent 开发专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化 AI Agent 开发专家"""
        super().__init__(
            name="agent_tech",
            description="AI Agent 开发技术专家",
            capabilities=[
                "AI Agent 架构设计",
                "LLM 集成与优化", 
                "RAG 检索增强生成",
                "LangChain/LangGraph 框架",
                "Prompt 工程",
                "向量数据库与 Embedding",
                "多 Agent 协作系统"
            ],
            knowledge_dir=KNOWLEDGE_BASE_DIR,
            collection_name="agent_knowledge",
            prompt_template=AGENT_PROMPT,
            domain_metadata={
                "domain": "ai_agent_development",
                "response_type": "technical_advice"
            }
        )


# 全局 AI Agent 开发专家实例
_agent_tech_expert = None


def get_agent_tech_expert() -> AgentTechExpert:
    """获取全局 AI Agent 开发专家实例"""
    global _agent_tech_expert
    if _agent_tech_expert is None:
        _agent_tech_expert = AgentTechExpert()
        agent_manager.register_agent(_agent_tech_expert)
    return _agent_tech_expert


# 导出接口
__all__ = [
    "AgentTechExpert",
    "get_agent_tech_expert",
]

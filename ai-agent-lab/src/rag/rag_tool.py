"""
=== RAG 工具模块 ===

【职责】：
1. 提供可作为工具调用的 RAG 实现
2. 支持按需检索，而不是每次都检索
3. 符合 LangChain Tool 规范

【设计原则】：
1. 按需检索：Agent 自己决定何时调用
2. 可配置：支持不同的知识库和集合
3. 流式支持：返回格式支持流式输出
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun

from src.rag.engine import RAGEngine

logger = logging.getLogger("rag_tool")


class RAGTool:
    """RAG 工具类 - 封装知识库检索为可调用工具"""

    def __init__(
        self,
        knowledge_dir: str,
        collection_name: str,
        name: str = "query_knowledge_base",
        description: str = "查询领域知识库，返回相关信息。当需要了解具体景点、美食、交通、预算等专业领域知识时使用。"
    ):
        """
        Args:
            knowledge_dir: 知识库目录路径
            collection_name: ChromaDB 集合名称
            name: 工具名称
            description: 工具描述
        """
        self.name = name
        self.description = description
        self._rag_engine = RAGEngine(
            knowledge_dir=knowledge_dir,
            collection_name=collection_name
        )
        self._knowledge_dir = knowledge_dir
        self._collection_name = collection_name

    def query(self, question: str) -> Dict[str, Any]:
        """执行 RAG 查询

        Args:
            question: 查询问题

        Returns:
            Dict包含:
                - answer_context: 检索到的上下文
                - sources: 来源列表
                - found: 是否找到相关内容
        """
        try:
            result = self._rag_engine.query(question)
            return result
        except Exception as e:
            logger.error(f"RAG 查询失败: {e}")
            return {
                "found": False,
                "answer_context": "",
                "sources": []
            }


def create_rag_tool(
    knowledge_dir: str,
    collection_name: str,
    name: str,
    description: str
) -> RAGTool:
    """工厂函数：创建 RAG 工具

    Args:
        knowledge_dir: 知识库目录
        collection_name: 集合名称
        name: 工具名称
        description: 工具描述

    Returns:
        RAGTool 实例
    """
    return RAGTool(
        knowledge_dir=knowledge_dir,
        collection_name=collection_name,
        name=name,
        description=description
    )


# 预定义的专家领域 RAG 工具
def get_plan_rag_tool() -> RAGTool:
    """获取规划专家 RAG 工具"""
    from src.config import KNOWLEDGE_BASE_PLAN_DIR
    return create_rag_tool(
        knowledge_dir=KNOWLEDGE_BASE_PLAN_DIR,
        collection_name="plan_knowledge",
        name="query_plan_knowledge",
        description=(
            "查询旅行规划和财务预算相关的知识库。适用于："
            "目的地推荐、行程规划、签证政策、酒店预订、预算精算、"
            "汇率换算、旅行保险、开支优化等问题的知识检索。"
        )
    )


def get_food_rag_tool() -> RAGTool:
    """获取美食专家 RAG 工具"""
    from src.config import KNOWLEDGE_BASE_FOOD_DIR
    return create_rag_tool(
        knowledge_dir=KNOWLEDGE_BASE_FOOD_DIR,
        collection_name="food_knowledge",
        name="query_food_knowledge",
        description=(
            "查询美食相关的知识库。适用于："
            "菜品推荐、餐厅点评、美食文化、就餐礼仪、"
            "订餐建议、美食街区路线等问题的知识检索。"
        )
    )


def get_sights_rag_tool() -> RAGTool:
    """获取景点专家 RAG 工具"""
    from src.config import KNOWLEDGE_BASE_SIGHTS_DIR
    return create_rag_tool(
        knowledge_dir=KNOWLEDGE_BASE_SIGHTS_DIR,
        collection_name="sights_knowledge",
        name="query_sights_knowledge",
        description=(
            "查询景点相关的知识库。适用于："
            "景点介绍、门票政策、开放时间、游览路线、"
            "景区设施、避坑指南等问题的知识检索。"
        )
    )


def get_transport_rag_tool() -> RAGTool:
    """获取交通专家 RAG 工具"""
    from src.config import KNOWLEDGE_BASE_TRANSPORT_DIR
    return create_rag_tool(
        knowledge_dir=KNOWLEDGE_BASE_TRANSPORT_DIR,
        collection_name="transport_knowledge",
        name="query_transport_knowledge",
        description=(
            "查询交通出行相关的知识库。适用于："
            "航班车次查询、交通方案对比、换乘指引、"
            "接驳指南、票务政策、行李规定等问题的知识检索。"
        )
    )


def get_agent_tech_rag_tool() -> RAGTool:
    """获取 AI Agent 技术专家 RAG 工具"""
    from src.config import KNOWLEDGE_BASE_DIR
    return create_rag_tool(
        knowledge_dir=KNOWLEDGE_BASE_DIR,
        collection_name="agent_knowledge",
        name="query_agent_tech_knowledge",
        description=(
            "查询 AI Agent 开发相关的技术知识库。适用于："
            "LangChain/LangGraph 框架问题、Prompt 工程、RAG 实现、"
            "向量数据库、工具调用架构等 AI 技术问题的知识检索。"
        )
    )


__all__ = [
    "RAGTool",
    "create_rag_tool",
    "get_plan_rag_tool",
    "get_food_rag_tool",
    "get_sights_rag_tool",
    "get_transport_rag_tool",
    "get_agent_tech_rag_tool"
]

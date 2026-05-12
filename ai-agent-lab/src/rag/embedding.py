"""
=== Embedding 模型服务 — 全局单例，统一管理向量化能力 ===

【知识点】为什么要把 Embedding 模型抽成独立服务？

1. 避免重复加载：Embedding 模型约 400MB，加载耗时 2~5 秒
   - RAG 引擎需要 Embedding（文档向量化 + 查询向量化）
   - 对话记忆需要 Embedding（历史对话向量化 + 语义检索）
   - 如果各自加载，内存浪费 400MB，启动多等 3 秒

2. 统一管理：模型配置（名称、设备、归一化）只在一处定义
   - 修改模型只改这一个文件，不用到处找

3. 统一接口：全部使用 LangChain 的 HuggingFaceEmbeddings
   - embed_query(text) → list[float]：单条文本向量化
   - embed_documents(texts) → list[list[float]]：批量文本向量化
   - RAG 引擎、对话记忆、Rerank 等模块统一调用同一个实例

【现实例子 — 电商平台】
就像公司的翻译部门：商品搜索、客服系统、推荐引擎都需要"文本转向量"的能力，
与其每个部门各养一个翻译团队，不如成立一个共享的翻译中心，统一提供服务。
"""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ============================================================
# 全局单例实例
# ============================================================
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取全局 Embedding 模型单例

    【知识点】HuggingFaceEmbeddings 是 LangChain 对 SentenceTransformer 的封装：
    - embed_query(text)：单条文本 → list[float]
    - embed_documents(texts)：批量文本 → list[list[float]]
    - 可直接传给 Chroma、FAISS 等 LangChain 向量数据库
    - 也可用于对话记忆的语义检索（embed_query 获取向量）

    Args:
        无参数

    Returns:
        HuggingFaceEmbeddings: 全局 Embedding 模型单例，
                               模型名称由 config.EMBEDDING_MODEL 配置
    """
    global _embeddings
    if _embeddings is None:
        logger.info(f"正在加载 Embedding 模型: {EMBEDDING_MODEL}")
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding 模型加载完成 ✅")
        except Exception as e:
            logger.error(f"加载 Embedding 模型失败: {e}")
    return _embeddings

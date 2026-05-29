"""
=== 长期记忆管理模块 ===

【功能】
基于企业级架构的长期记忆管理，包括：
1. 语义记忆：对话历史的向量化存储与语义检索
2. 用户画像：结构化用户偏好和特征存储
3. 知识积累：实体和关系的持久化存储

【企业级长期记忆设计】
1. 向量数据库：使用 ChromaDB 存储对话嵌入，支持语义召回
2. 用户画像：KV 结构化存储用户偏好、特征、上下文
3. Rerank 重排序：Cross-Encoder 精排提高准确率
4. 增量更新：支持知识的高效增量更新，避免重复计算

【三层记忆架构中的位置】
- 长期记忆位于最底层，存储用户画像、历史知识
- 通过语义检索召回相关片段，注入到短期记忆中
- 支持跨会话的知识复用（同一用户的不同会话共享记忆）
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config import (
    CHROMA_DB_DIR,
    LONG_MEMORY_SEMANTIC_TOP_K,
    LONG_MEMORY_RERANK_TOP_N,
    LONG_MEMORY_PROFILE_DIR,
)
from src.rag.embedding import get_embeddings

logger = logging.getLogger(__name__)

LONG_TERM_COLLECTION_NAME: str = "long_term_memory"
USER_PROFILE_COLLECTION: str = "user_profiles"
SEMANTIC_TOP_K: int = LONG_MEMORY_SEMANTIC_TOP_K
RERANK_TOP_N: int = LONG_MEMORY_RERANK_TOP_N
USER_PROFILE_DIR: str = LONG_MEMORY_PROFILE_DIR


class SemanticMemoryStore:
    """语义记忆存储 — 对话历史的向量化存储与检索

    【核心思想】
    - 每轮对话（问+答）作为独立文档，转换为嵌入向量
    - 按 user_id 隔离存储，支持跨会话检索（同一用户的不同会话共享记忆）
    - 用户提问时，语义匹配最相关的历史片段

    【企业级特性】
    - 向量检索：基于 ChromaDB 的余弦相似度检索
    - 用户隔离：只返回同一 user_id 下的记忆
    - Rerank 重排序：Cross-Encoder 精排提高准确率
    - 元数据存储：时间戳、消息预览等辅助信息
    """

    def __init__(self) -> None:
        self._embedding_model = get_embeddings()
        memory_db_dir = os.path.join(CHROMA_DB_DIR, "long_term_memory")
        self._client = chromadb.PersistentClient(
            path=memory_db_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=LONG_TERM_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._init_reranker()
        logger.info(f"语义记忆存储初始化完成，当前记录数: {self._collection.count()}")

    def _init_reranker(self) -> None:
        """初始化 Cross-Encoder Rerank 模型

        【知识点】Bi-Encoder vs Cross-Encoder：
        - Bi-Encoder（Embedding 模型）：速度快，适合初筛
        - Cross-Encoder（Rerank 模型）：精度高，适合精排

        最佳实践：先检索 Top K，再重排 Top N
        """
        try:
            self._reranker_model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3"
            )
            self._reranker = CrossEncoderReranker(
                model=self._reranker_model,
                top_n=RERANK_TOP_N,
            )
            logger.info("Rerank 重排序器就绪 (bge-reranker-v2-m3)")
        except Exception as e:
            self._reranker = None
            logger.warning(f"Rerank 模型加载失败，回退到无 Rerank 模式: {e}")

    def add_conversation_turn(self, user_id: str, human_msg: str, ai_msg: str) -> None:
        doc_text = f"用户: {human_msg}\n助手: {ai_msg}"
        doc_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        embedding = self._embedding_model.embed_query(doc_text)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "human_msg": human_msg[:200],
            }],
        )

    def search_relevant_memories(self, user_id: str, query: str, top_k: int = SEMANTIC_TOP_K) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []
        
        # 第一步：向量检索获取候选
        query_embedding = self._embedding_model.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where={"user_id": user_id},
        )
        
        documents = results.get("documents", [[]])[0]
        if not documents:
            return []
        
        # 第二步：Rerank 重排序（如果可用）
        if self._reranker:
            from langchain_core.documents import Document
            from langchain_core.retrievers import BaseRetriever
            
            class DummyRetriever(BaseRetriever):
                def __init__(self, docs):
                    self.docs = docs
                
                def _get_relevant_documents(self, query):
                    return [Document(page_content=doc) for doc in self.docs]
            
            dummy_retriever = DummyRetriever(documents)
            ranked_docs = self._reranker.compress_documents(
                query=query,
                documents=[Document(page_content=doc) for doc in documents]
            )
            return [doc.page_content for doc in ranked_docs]
        
        # 无 Rerank 模式：直接返回前 RERANK_TOP_N 个
        return documents[:RERANK_TOP_N]


class UserProfileStore:
    """用户画像存储 — 结构化用户偏好和特征

    【核心思想】
    - 以 user_id 为 Key，存储结构化用户信息
    - 包括：偏好设置、历史交互摘要、实体关系等
    - 支持快速读取，用于上下文增强

    【企业级特性】
    - KV 存储：基于 ChromaDB 的 metadata 过滤实现
    - 增量更新：只更新变化的字段
    - 自动构建：首次交互时自动创建空画像
    """

    def __init__(self) -> None:
        self._embedding_model = get_embeddings()
        profile_db_dir = USER_PROFILE_DIR if os.path.isabs(USER_PROFILE_DIR) else os.path.join(CHROMA_DB_DIR, USER_PROFILE_DIR)
        self._client = chromadb.PersistentClient(
            path=profile_db_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=USER_PROFILE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"用户画像存储初始化完成，当前记录数: {self._collection.count()}")

    def get_profile(self, user_id: str) -> dict:
        results = self._collection.get(
            where={"user_id": user_id},
            limit=1,
        )
        if results["documents"]:
            try:
                return json.loads(results["documents"][0])
            except json.JSONDecodeError:
                return {}
        return self._create_empty_profile(user_id)

    def _create_empty_profile(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": {},
            "interaction_summary": [],
            "entities": [],
            "custom_data": {},
        }

    def update_profile(self, user_id: str, profile_data: dict) -> None:
        profile_data["updated_at"] = datetime.now().isoformat()
        existing = self.get_profile(user_id)
        merged = {**existing, **profile_data}
        doc_id = f"profile_{user_id}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[json.dumps(merged, ensure_ascii=False)],
            metadatas=[{"user_id": user_id}],
        )

    def add_interaction_summary(self, user_id: str, summary: str) -> None:
        profile = self.get_profile(user_id)
        profile["interaction_summary"].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        })
        if len(profile["interaction_summary"]) > 10:
            profile["interaction_summary"] = profile["interaction_summary"][-10:]
        self.update_profile(user_id, profile)

    def set_preference(self, user_id: str, key: str, value: Any) -> None:
        profile = self.get_profile(user_id)
        profile["preferences"][key] = value
        self.update_profile(user_id, profile)


class LongTermMemoryManager:
    """长期记忆管理器

    【职责】
    1. 管理语义记忆：对话历史的向量化存储与检索
    2. 管理用户画像：结构化用户偏好和特征

    【企业级特性】
    - 语义检索 + Rerank：高精度召回相关历史
    - 用户画像：跨会话持久化用户偏好
    - 上下文增强：检索结果和画像注入到对话上下文
    """

    def __init__(self) -> None:
        self._semantic_store = SemanticMemoryStore()
        self._profile_store = UserProfileStore()
        logger.info("长期记忆管理器初始化完成 ✅")

    def save_conversation_turn(self, user_id: str, human_msg: str, ai_msg: str) -> None:
        if human_msg and ai_msg:
            self._semantic_store.add_conversation_turn(user_id, human_msg, ai_msg)

    def recall_relevant_memories(self, user_id: str, query: str) -> list[str]:
        return self._semantic_store.search_relevant_memories(user_id, query)

    def get_user_profile(self, user_id: str) -> dict:
        return self._profile_store.get_profile(user_id)

    def update_user_profile(self, user_id: str, profile_data: dict) -> None:
        self._profile_store.update_profile(user_id, profile_data)

    def add_interaction_summary_to_profile(self, user_id: str, summary: str) -> None:
        self._profile_store.add_interaction_summary(user_id, summary)

    def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        self._profile_store.set_preference(user_id, key, value)

    def build_memory_context(self, user_id: str, current_query: str) -> str:
        parts = []
        profile = self._profile_store.get_profile(user_id)
        if profile.get("interaction_summary"):
            recent = profile["interaction_summary"][-3:]
            summary_text = "\n".join(f"- {s['summary']}" for s in recent)
            parts.append(f"[历史交互摘要]\n{summary_text}")
        memories = self._semantic_store.search_relevant_memories(user_id, current_query)
        if memories:
            parts.append(f"[相关历史对话]\n" + "\n---\n".join(memories))
        return "\n\n".join(parts) if parts else ""


_long_term_memory_manager: Optional[LongTermMemoryManager] = None


def get_long_term_memory_manager() -> LongTermMemoryManager:
    global _long_term_memory_manager
    if _long_term_memory_manager is None:
        _long_term_memory_manager = LongTermMemoryManager()
    return _long_term_memory_manager

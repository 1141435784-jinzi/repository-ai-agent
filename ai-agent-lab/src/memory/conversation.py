"""
=== 对话记忆增强模块 — 滑动窗口 + 摘要压缩 + 语义检索 ===

【知识点】企业级 Agent 的三层记忆架构：

1. 滑动窗口（Short-term）：最近 N 轮对话保持原文，保证即时上下文连贯
2. 摘要压缩（Summary）：超出窗口的历史对话由 LLM 压缩为摘要，节省 token
3. 语义检索（Semantic）：所有历史对话向量化存储，当前问题语义匹配相关片段

【现实例子 — 银行客服系统】
- 用户上午问了贷款利率（第 1~5 轮）
- 下午又问了信用卡额度（第 6~15 轮）
- 晚上问"我上午问的那个利率是多少来着？"
  → 滑动窗口只有最近 10 轮（信用卡相关），找不到
  → 语义检索从向量库中匹配到"贷款利率"相关的历史片段
  → Agent 成功回忆起上午的对话内容

【为什么不能只用滑动窗口？】
- 窗口太大 → token 爆炸，成本高、延迟大
- 窗口太小 → 丢失重要上下文，Agent "失忆"
- 摘要 + 语义检索 = 用最少的 token 覆盖最多的历史信息
"""

import hashlib
import json
import logging
import os
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import CHROMA_DB_DIR
from src.rag.embedding import get_embeddings
from src.llm.service import get_llm

logger = logging.getLogger(__name__)

# ============================================================
# 配置常量
# ============================================================
# 滑动窗口大小：保留最近 N 轮对话（1 轮 = 1 条 Human + 1 条 AI）
WINDOW_SIZE: int = 10
# 语义检索返回的历史片段数量
SEMANTIC_TOP_K: int = 3
# 对话记忆向量集合名称
MEMORY_COLLECTION_NAME: str = "conversation_memory"


# ============================================================
# 语义记忆存储（基于 ChromaDB 向量检索）
# ============================================================

class SemanticMemoryStore:
    """语义记忆存储 — 将对话历史向量化，支持语义检索

    【知识点】语义记忆的核心思想：
    - 每轮对话（问+答）作为一个文档，用 Embedding 模型转为向量
    - 存入向量数据库（ChromaDB），按 thread_id 隔离
    - 用户提问时，用当前问题做语义检索，找到历史中最相关的对话片段
    - 这些片段作为"回忆"注入到 LLM 上下文中

    【企业实战】为什么用独立的 collection 而不是复用知识库的？
    - 知识库是静态文档，对话记忆是动态生成的
    - 两者的检索场景不同：知识库检索技术概念，记忆检索用户历史
    - 分开管理方便独立清理、备份、监控
    """

    def __init__(self) -> None:
        """初始化语义记忆存储

        Args:
            无参数

        Returns:
            None
        """
        # 【知识点】从 embedding_service 获取全局共享的 Embedding 模型实例
        # 统一使用 HuggingFaceEmbeddings，与 RAG 引擎共用同一个模型
        self._embedding_model = get_embeddings()

        # 【知识点】ChromaDB 持久化客户端，数据存磁盘
        # 使用独立子目录，避免与 RAG 引擎的 ChromaDB 实例冲突
        memory_db_dir = os.path.join(CHROMA_DB_DIR, "conversation_memory")
        self._client = chromadb.PersistentClient(
            path=memory_db_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"语义记忆存储初始化完成，当前记录数: {self._collection.count()}")

    def add_conversation_turn(self, thread_id: str, human_msg: str, ai_msg: str) -> None:
        """将一轮对话存入语义记忆

        【知识点】存储格式：将 Human + AI 拼接为一个文档
        - 这样检索时能同时匹配到问题和回答
        - metadata 中记录 thread_id 和时间戳，方便过滤和排序

        Args:
            thread_id: 会话 ID，用于隔离不同用户/会话的记忆
            human_msg: 用户消息文本
            ai_msg: AI 回复文本

        Returns:
            None
        """
        # 拼接为完整的对话片段
        doc_text = f"用户: {human_msg}\n助手: {ai_msg}"
        # 生成唯一 ID（thread_id + 时间戳）
        doc_id = f"{thread_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 计算 Embedding 向量（使用 LangChain 统一接口）
        embedding = self._embedding_model.embed_query(doc_text)

        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "human_msg": human_msg[:200],  # 截断存储，防止 metadata 过大
            }],
        )

    def search_relevant_memories(self, thread_id: str, query: str, top_k: int = SEMANTIC_TOP_K ) -> list[str]:
        """语义检索相关的历史对话片段

        【知识点】检索流程：
        1. 将当前用户问题转为向量
        2. 在该 thread_id 的记忆中做余弦相似度检索
        3. 返回最相关的 top_k 个历史对话片段

        Args:
            thread_id: 会话 ID，只检索当前会话的记忆
            query: 当前用户问题，用于语义匹配
            top_k: 返回的最相关片段数量

        Returns:
            list[str]: 相关的历史对话片段列表，按相似度降序排列
        """
        # 检查该 thread_id 下是否有记忆
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self._embedding_model.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where={"thread_id": thread_id},
        )

        # 提取文档文本
        documents = results.get("documents", [[]])[0]
        return documents


# ============================================================
# 对话摘要压缩器
# ============================================================

class ConversationSummarizer:
    """对话摘要压缩器 — 将超出窗口的历史对话压缩为摘要

    【知识点】摘要压缩的核心思想：
    - 最近 N 轮对话保持原文（滑动窗口），保证即时上下文
    - 窗口之前的历史对话，由 LLM 压缩为一段简洁的摘要
    - 摘要 + 窗口内原文 = 完整的记忆上下文，但 token 数大幅减少

    【企业实战 — token 成本对比】
    假设每轮对话平均 200 token，50 轮对话 = 10000 token
    - 不压缩：每次请求带 10000 token 历史 → 成本高、延迟大
    - 压缩后：摘要 ~500 token + 最近 10 轮 2000 token = 2500 token → 节省 75%
    """

    def __init__(self) -> None:
        """初始化摘要压缩器

        Args:
            无参数

        Returns:
            None
        """
        # 【知识点】用低 temperature 的 LLM 做摘要，确保输出稳定、不发散
        self._llm = get_llm(temperature=0.1)

    def summarize(self, messages: list) -> str:
        """将一组对话消息压缩为摘要

        【知识点】摘要 Prompt 设计要点：
        - 要求保留关键事实（人名、数字、结论）
        - 要求保留用户偏好和意图
        - 要求简洁，不超过 300 字
        - 不要求保留对话格式，只保留信息

        Args:
            messages: 需要压缩的消息列表（HumanMessage / AIMessage 混合）

        Returns:
            str: 压缩后的对话摘要文本
        """
        if not messages:
            return ""

        # 将消息列表格式化为文本
        conversation_text = self._format_messages(messages)

        summary_prompt = [
            SystemMessage(content=(
                "你是一个对话摘要助手。请将以下对话历史压缩为简洁的摘要。\n\n"
                "要求：\n"
                "1. 保留所有关键事实信息（人名、数字、日期、结论、决策）\n"
                "2. 保留用户的偏好和意图\n"
                "3. 保留重要的问答结论\n"
                "4. 摘要不超过 300 字\n"
                "5. 使用第三人称描述（'用户询问了...'、'助手回答了...'）\n"
                "6. 按时间顺序组织信息"
            )),
            HumanMessage(content=f"请压缩以下对话历史：\n\n{conversation_text}"),
        ]

        response = self._llm.invoke(summary_prompt)
        return response.content

    def _format_messages(self, messages: list) -> str:
        """将消息列表格式化为可读文本

        Args:
            messages: 消息列表

        Returns:
            str: 格式化后的对话文本
        """
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                lines.append(f"助手: {msg.content}")
        return "\n".join(lines)


# ============================================================
# 记忆管理器（整合三层记忆）
# ============================================================

class ConversationMemoryManager:
    """对话记忆管理器 — 整合滑动窗口 + 摘要压缩 + 语义检索

    【知识点】三层记忆的协作流程：

    用户发送新消息时：
    1. 从 state.messages 获取完整对话历史（Checkpointer 恢复的）
    2. 如果历史超过 WINDOW_SIZE 轮：
       a. 将窗口外的旧消息压缩为摘要
       b. 用当前问题做语义检索，找到相关的历史片段
    3. 组装最终的记忆上下文：摘要 + 语义检索结果 + 最近 N 轮原文
    4. 将记忆上下文注入到 Agent 的 system prompt 中

    Agent 回复后：
    5. 将本轮对话（问+答）存入语义记忆向量库

    【图中的位置】
    START → memory_node → intent_route → rag_node → agent_node → ...
                ↑
      在意图路由之前处理记忆，确保后续所有节点都能使用增强后的上下文
    """

    def __init__(self) -> None:
        """初始化记忆管理器

        Args:
            无参数

        Returns:
            None
        """
        # 历史语义检索实例
        self._semantic_store = SemanticMemoryStore()
        # 历史摘要实例
        self._summarizer = ConversationSummarizer()

        # 缓存每个 thread 的摘要，避免重复压缩
        self._summary_cache: dict[str, str] = {}
        # 增量缓存：记录每个 thread 已处理的旧消息状态
        self._processed_messages_state: dict[str, list] = {}
        logger.info("对话记忆管理器初始化完成 ✅")

    def process_memory( self, messages: list, thread_id: str, current_query: str ) -> dict:
        """处理记忆 — 核心方法，返回增强后的记忆上下文

        【知识点】处理逻辑：
        1. 统计对话轮数（只计 HumanMessage）
        2. 如果 <= WINDOW_SIZE 轮：不做任何处理，直接用原始 messages
        3. 如果 > WINDOW_SIZE 轮：
           - 分割为 [旧消息 | 最近 N 轮]
           - 旧消息 → 压缩为摘要
           - 当前问题 → 语义检索相关历史
           - 返回：摘要 + 语义检索结果 + 最近 N 轮原文

        Args:
            messages: 完整的对话历史消息列表（从 Checkpointer 恢复的）
            thread_id: 会话 ID
            current_query: 当前用户问题

        Returns:
            dict: 包含以下字段：
                - trimmed_messages: 裁剪后的消息列表（最近 N 轮）
                - memory_context: 记忆上下文字符串（摘要 + 语义检索结果）
                - needs_trim: 是否进行了裁剪
        """
        # 统计对话轮数（每条 HumanMessage 算一轮）
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))

        if human_count <= WINDOW_SIZE:
            # 未超出窗口，不需要处理
            return {
                "trimmed_messages": messages,
                "memory_context": "",
                "needs_trim": False,
            }

        # 超出窗口 — 需要裁剪 + 压缩 + 语义检索
        logger.info(
            f"对话轮数 {human_count} 超出窗口 {WINDOW_SIZE}，"
            f"启动摘要压缩 + 语义检索"
        )

        # 分割消息：找到最近 WINDOW_SIZE 轮的起始位置
        window_messages, old_messages = self._split_messages(messages)

        # 超出窗口大小的历史消息处理
        # 1. 压缩旧消息为摘要
        summary = self._get_or_create_summary(thread_id, old_messages)

        # 2. 语义检索相关历史
        relevant_memories = self._semantic_store.search_relevant_memories(
            thread_id=thread_id,
            query=current_query,
        )

        # 3. 组装记忆上下文
        memory_context = self._build_memory_context(summary, relevant_memories)

        return {
            "trimmed_messages": window_messages,
            "memory_context": memory_context,
            "needs_trim": True,
        }

    def save_conversation_turn(self, thread_id: str, human_msg: str, ai_msg: str) -> None:
        """保存一轮对话到语义记忆

        【知识点】在 Agent 回复后调用，将本轮对话向量化存储
        后续可以通过语义检索找回这段对话

        Args:
            thread_id: 会话 ID
            human_msg: 用户消息
            ai_msg: AI 回复

        Returns:
            None
        """
        if human_msg and ai_msg:
            self._semantic_store.add_conversation_turn(thread_id, human_msg, ai_msg)

    def _split_messages(self, messages: list) -> tuple[list, list]:
        """将消息列表分割为 [最近 N 轮] 和 [旧消息]

        【知识点】分割逻辑：
        - 从后往前数 WINDOW_SIZE 个 HumanMessage
        - 该 HumanMessage 及之后的所有消息 = 窗口内（保留原文）
        - 之前的所有消息 = 窗口外（需要压缩）

        Args:
            messages: 完整消息列表

        Returns:
            tuple: (window_messages, old_messages)
        """
        # 从后往前找第 WINDOW_SIZE 个 HumanMessage 的位置
        human_indices = [
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        ]

        if len(human_indices) <= WINDOW_SIZE:
            return messages, []

        # 窗口起始位置：倒数第 WINDOW_SIZE 个 HumanMessage
        window_start_idx = human_indices[-WINDOW_SIZE]

        old_messages = messages[:window_start_idx]
        window_messages = messages[window_start_idx:]

        return window_messages, old_messages

    def _get_or_create_summary(self, thread_id: str, old_messages: list) -> str:
        """获取或创建对话摘要（增量缓存策略）

        【知识点】增量摘要缓存策略：
        - 记录每个 thread 已处理的旧消息状态
        - 只对新增加的旧消息进行压缩
        - 增量更新缓存的摘要
        - 用户重复提问100次时，第2-100次直接命中缓存

        【企业级优势】：
        - 性能提升100倍：从100次LLM调用减少到1次
        - 成本控制：避免重复的LLM摘要生成
        - 实时性：支持流式对话的增量更新

        Args:
            thread_id: 会话 ID
            old_messages: 需要压缩的旧消息列表

        Returns:
            str: 对话摘要文本
        """
        if not old_messages:
            return ""
        
        # 1. 获取当前 thread 的已处理消息状态
        processed_messages = self._processed_messages_state.get(thread_id, [])
        
        # 2. 找出新增的旧消息（增量部分）
        new_old_messages = self._get_new_old_messages(
            old_messages, processed_messages
        )
        
        # 3. 获取当前会话缓存的摘要
        last_summary = self._summary_cache.get(f"{thread_id}_last", "")
        
        if not new_old_messages:
            # 没有新增消息，直接返回缓存的摘要
            logger.info(f"无新增旧消息，使用缓存的摘要（长度: {len(last_summary)} 字）")
            return last_summary
        
        # 4. 只压缩新增的消息（增量压缩）
        logger.info(f"正在压缩 {len(new_old_messages)} 条新增旧消息为摘要...")
        new_summary = self._summarizer.summarize(new_old_messages)
        
        # 5. 合并摘要（增量更新）
        if last_summary:
            combined_summary = self._merge_summaries(last_summary, new_summary)
        else:
            combined_summary = new_summary
        
        # 6. 更新缓存和状态
        self._summary_cache[f"{thread_id}_last"] = combined_summary
        self._processed_messages_state[thread_id] = old_messages.copy()
        
        logger.info(f"增量摘要完成，总长度: {len(combined_summary)} 字")
        return combined_summary
    
    def _get_new_old_messages(self, current_old_messages: list, processed_messages: list) -> list:
        """找出新增的旧消息（增量部分）
        
        Args:
            current_old_messages: 当前所有的旧消息
            processed_messages: 已处理的旧消息
            
        Returns:
            list: 新增的旧消息列表
        """
        if not processed_messages:
            return current_old_messages
        
        # 使用消息内容的哈希值进行比较
        processed_hashes = set()
        for msg in processed_messages:
            msg_hash = self._calculate_message_hash(msg)
            processed_hashes.add(msg_hash)
        
        new_messages = []
        for msg in current_old_messages:
            msg_hash = self._calculate_message_hash(msg)
            if msg_hash not in processed_hashes:
                new_messages.append(msg)
        
        return new_messages
    
    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """合并两个摘要（企业级简化版）
        
        【知识点】合并策略：
        1. 先尝试简单合并
        2. 如果超过300字，请求LLM重新压缩
        3. 信任LLM的输出，不强制截断
        
        【企业级价值】：
        - 保持简单可靠：避免过度复杂的后处理
        - 信任LLM能力：提示词已要求"不超过300字"
        - 成本可控：重压缩只在必要时触发
        
        Args:
            old_summary: 旧的摘要
            new_summary: 新的摘要
            
        Returns:
            str: 合并后的摘要（LLM会尽量控制在300字内）
        """
        if not old_summary:
            return new_summary
        if not new_summary:
            return old_summary
        
        # 1. 先尝试简单合并
        combined = f"{old_summary}\n\n{new_summary}"
        
        # 2. 检查是否超过300字（中文字数）
        chinese_char_count = self._count_chinese_chars(combined)
        if chinese_char_count <= 300:
            # 未超限，直接返回
            return combined
        
        # 3. 超过300字，需要重新压缩
        logger.info(f"合并摘要超限（{chinese_char_count}字 > 300字），启动重压缩")
        
        # 使用摘要器重新压缩合并后的摘要
        recompressed_summary = self._recompress_merged_summaries(old_summary, new_summary)
        
        # 4. 记录重压缩结果（信任LLM的输出，不强制截断）
        recompressed_char_count = self._count_chinese_chars(recompressed_summary)
        logger.info(f"重压缩完成，长度: {recompressed_char_count} 字")
        
        return recompressed_summary
    
    def _recompress_merged_summaries(self, old_summary: str, new_summary: str) -> str:
        """重新压缩合并后的摘要
        
        【提示词设计】：
        - 明确要求"摘要不超过300字"
        - 要求合并关键信息，去除冗余
        - 信任LLM会遵守长度约束
        
        Args:
            old_summary: 旧的摘要
            new_summary: 新的摘要
            
        Returns:
            str: 重新压缩后的摘要（LLM会尽量控制在300字内）
        """
        # 创建重压缩的提示词
        recompress_prompt = [
            SystemMessage(content=(
                "你是一个对话摘要优化助手。请将以下两个摘要合并为一个简洁的摘要。\n\n"
                "要求：\n"
                "1. 合并两个摘要的所有关键信息\n"
                "2. 去除重复和冗余的内容\n"
                "3. 按时间顺序组织信息\n"
                "4. 摘要不超过300字\n"
                "5. 保留所有重要的事实、决策和用户偏好\n"
                "6. 使用第三人称描述"
            )),
            HumanMessage(content=(
                f"请合并以下两个摘要：\n\n"
                f"【摘要一】\n{old_summary}\n\n"
                f"【摘要二】\n{new_summary}\n\n"
                f"请生成一个合并后的摘要（不超过300字）："
            )),
        ]
        
        # 使用摘要器的LLM进行重压缩
        response = self._summarizer._llm.invoke(recompress_prompt)
        return response.content
    
    def _count_chinese_chars(self, text: str) -> int:
        """计算文本中的中文字符数"""
        return sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    
    def _serialize_message_for_hash(self, msg) -> dict:
        """序列化消息用于哈希计算
        
        Args:
            msg: 消息对象
            
        Returns:
            dict: 序列化后的消息字典
        """
        if hasattr(msg, 'content'):
            return {
                'type': msg.__class__.__name__,
                'content': str(msg.content)
            }
        return {
            'type': msg.__class__.__name__,
            'content': ''
        }
    
    def _calculate_message_hash(self, msg) -> str:
        """计算单个消息的哈希值
        
        Args:
            msg: 消息对象
            
        Returns:
            str: 16位哈希值
        """
        msg_dict = self._serialize_message_for_hash(msg)
        msg_json = json.dumps(msg_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(msg_json.encode('utf-8')).hexdigest()[:16]

    def _build_memory_context(self, summary: str, relevant_memories: list[str]) -> str:
        """组装记忆上下文字符串

        Args:
            summary: 对话摘要
            relevant_memories: 语义检索到的相关历史片段

        Returns:
            str: 格式化的记忆上下文，注入到 system prompt 中
        """
        parts = []

        if summary:
            parts.append(
                "【对话摘要 — 早期对话的压缩记录】\n"
                f"{summary}"
            )

        if relevant_memories:
            memories_text = "\n---\n".join(relevant_memories)
            parts.append(
                "【语义记忆 — 与当前问题相关的历史对话片段】\n"
                f"{memories_text}"
            )

        return "\n\n".join(parts)


# ============================================================
# 全局实例（服务启动时预加载）
# ============================================================
print("🧠 正在初始化对话记忆管理器...")
conversation_memory_manager = ConversationMemoryManager()
print("🧠 对话记忆管理器初始化完成 ✅")


def get_memory_manager() -> ConversationMemoryManager:
    """获取全局对话记忆管理器实例

    Args:
        无参数

    Returns:
        ConversationMemoryManager: 全局记忆管理器单例
    """
    return conversation_memory_manager

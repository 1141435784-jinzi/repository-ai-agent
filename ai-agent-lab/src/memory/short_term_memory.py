"""
=== 短期对话记忆增强模块 — 滑动窗口 + 摘要压缩 + 语义检索 ===

【企业级 Agent 的三层记忆架构】

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

【Token 管理策略】
当召回的对话内容过多，叠加当前对话直接超出大模型上下文窗口时，采用以下策略：

1. 动态窗口调整：根据剩余上下文空间自动调整窗口大小
   - 空间充足（> 4000 token）：使用标准窗口（10轮）
   - 空间一般（2000-4000 token）：使用半窗口（5轮）
   - 空间紧张（< 2000 token）：使用最小窗口（3轮）

2. 多维滑动窗口判断：同时考虑对话轮次和 Token 数量
   - 先按对话轮次计算窗口位置
   - 再检查按轮次裁剪后的 token 数是否超出限制
   - 如果超出，继续往前裁剪直到满足 token 限制
   - 解决场景：只有 2 轮对话但每轮内容很长（超出上下文窗口）
   - 确保即使对话轮次很少，但内容很长时也能正确处理

3. 记忆上下文裁剪：智能选择最相关的记忆片段
   - 优先保留摘要（通常包含最重要信息）
   - 保留语义检索结果中最相关的片段
   - 按段落边界裁剪，保持语义完整

4. 逐级降级策略：
   - Level 0（无压缩）：对话轮数 ≤ 窗口大小，直接使用
   - Level 1（标准压缩）：正常处理，摘要 + 语义检索
   - Level 2（深度压缩）：记忆上下文裁剪到 1500 token
   - Level 3（极端压缩）：窗口缩小到 3 轮 + 只保留关键点

【与长期记忆的职责划分】
- 短期记忆：按 thread_id 隔离，管理当前会话的对话历史
- 长期记忆：按 user_id 隔离，管理跨会话的用户画像和历史知识
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional, Any

import chromadb
from chromadb.config import Settings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config import (
    CHROMA_DB_DIR,
    MEMORY_WINDOW_SIZE,
    MEMORY_MIN_WINDOW_SIZE,
    MEMORY_SEMANTIC_TOP_K,
    MEMORY_MAX_CONTEXT_TOKENS,
    MEMORY_RESERVED_TOKENS,
    MEMORY_MAX_MEMORY_CONTEXT_TOKENS,
    MEMORY_COMPACT_CONTEXT_TOKENS,
)
from src.rag.embedding import get_embeddings
from src.llm.gateway import get_llm

logger = logging.getLogger(__name__)

# ============================================================
# 配置常量（从 settings.py 导入）
# ============================================================
WINDOW_SIZE: int = MEMORY_WINDOW_SIZE
MIN_WINDOW_SIZE: int = MEMORY_MIN_WINDOW_SIZE
SEMANTIC_TOP_K: int = MEMORY_SEMANTIC_TOP_K
MEMORY_COLLECTION_NAME: str = "short_term_memory"
MAX_CONTEXT_TOKENS: int = MEMORY_MAX_CONTEXT_TOKENS
RESERVED_TOKENS: int = MEMORY_RESERVED_TOKENS
MAX_MEMORY_CONTEXT_TOKENS: int = MEMORY_MAX_MEMORY_CONTEXT_TOKENS
COMPACT_MEMORY_CONTEXT_TOKENS: int = MEMORY_COMPACT_CONTEXT_TOKENS


# ============================================================
# 语义记忆存储（基于 ChromaDB 向量检索）
# ============================================================

class SemanticMemoryStore:
    """语义记忆存储 — 将对话历史向量化，支持语义检索

    【核心思想】
    - 每轮对话（问+答）作为一个文档，用 Embedding 模型转为向量
    - 存入向量数据库（ChromaDB），按 thread_id 隔离
    - 用户提问时，用当前问题做语义检索，找到历史中最相关的对话片段
    - 这些片段作为"回忆"注入到 LLM 上下文中

    【企业实战】为什么用独立的 collection？
    - 对话记忆是动态生成的，与静态知识库分离
    - 分开管理方便独立清理、备份、监控
    """

    def __init__(self) -> None:
        self._embedding_model = get_embeddings()
        memory_db_dir = os.path.join(CHROMA_DB_DIR, "short_term_memory")
        self._client = chromadb.PersistentClient(
            path=memory_db_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"短期语义记忆存储初始化完成，当前记录数: {self._collection.count()}")

    def add_conversation_turn(self, thread_id: str, human_msg: str, ai_msg: str) -> None:
        """将一轮对话存入语义记忆

        Args:
            thread_id: 会话 ID，用于隔离不同用户/会话的记忆
            human_msg: 用户消息文本
            ai_msg: AI 回复文本

        Returns:
            None
        """
        doc_text = f"用户: {human_msg}\n助手: {ai_msg}"
        doc_id = f"{thread_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        embedding = self._embedding_model.embed_query(doc_text)

        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "human_msg": human_msg[:200],
            }],
        )

    def search_relevant_memories(self, thread_id: str, query: str, top_k: int = SEMANTIC_TOP_K) -> list[str]:
        """语义检索相关的历史对话片段

        Args:
            thread_id: 会话 ID，只检索当前会话的记忆
            query: 当前用户问题，用于语义匹配
            top_k: 返回的最相关片段数量

        Returns:
            list[str]: 相关的历史对话片段列表，按相似度降序排列
        """
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self._embedding_model.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where={"thread_id": thread_id},
        )

        return results.get("documents", [[]])[0]


# ============================================================
# 对话摘要压缩器
# ============================================================

class ConversationSummarizer:
    """对话摘要压缩器 — 将超出窗口的历史对话压缩为摘要

    【核心思想】
    - 最近 N 轮对话保持原文（滑动窗口），保证即时上下文
    - 窗口之前的历史对话，由 LLM 压缩为一段简洁的摘要
    - 摘要 + 窗口内原文 = 完整的记忆上下文，但 token 数大幅减少

    【企业实战 — token 成本对比】
    假设每轮对话平均 200 token，50 轮对话 = 10000 token
    - 不压缩：每次请求带 10000 token 历史 → 成本高、延迟大
    - 压缩后：摘要 ~500 token + 最近 10 轮 2000 token = 2500 token → 节省 75%
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)

    def summarize(self, messages: list) -> str:
        """将一组对话消息压缩为摘要

        Args:
            messages: 需要压缩的消息列表（HumanMessage / AIMessage 混合）

        Returns:
            str: 压缩后的对话摘要文本
        """
        if not messages:
            return ""

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
# 滑动窗口对话历史存储
# ============================================================

class SlidingWindowChatHistory(BaseChatMessageHistory):
    """基于滑动窗口的对话历史管理器

    【核心思想】
    - 自动维护最近 N 轮对话
    - 超出窗口的消息自动丢弃（由上层决定是否需要摘要压缩）
    - 符合 LangChain BaseChatMessageHistory 接口规范

    【与 SemanticMemoryStore 的区别】
    - 本类：内存中的短期存储，自动裁剪
    - SemanticMemoryStore：持久化的向量存储，支持语义检索
    """

    def __init__(self, thread_id: str, window_size: int = WINDOW_SIZE) -> None:
        self._thread_id = thread_id
        self._window_size = window_size
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._messages.extend(messages)
        self._trim_to_window()

    def _trim_to_window(self) -> None:
        """裁剪消息到窗口大小

        【知识点】裁剪逻辑：
        - 统计 HumanMessage 的数量
        - 如果超过窗口大小，从开头删除直到符合要求
        """
        human_count = sum(1 for m in self._messages if isinstance(m, HumanMessage))
        while human_count > self._window_size:
            if isinstance(self._messages[0], HumanMessage):
                human_count -= 1
            self._messages.pop(0)

    def clear(self) -> None:
        self._messages.clear()


# ============================================================
# 短期记忆管理器（整合三层记忆）
# ============================================================

class ShortTermMemoryManager:
    """短期记忆管理器 — 整合滑动窗口 + 摘要压缩 + 语义检索

    【三层记忆的协作流程】

    用户发送新消息时：
    1. 从 Checkpointer 获取完整对话历史
    2. 如果历史超过 WINDOW_SIZE 轮：
       a. 将窗口外的旧消息压缩为摘要
       b. 用当前问题做语义检索，找到相关的历史片段
    3. 组装最终的记忆上下文：摘要 + 语义检索结果 + 最近 N 轮原文
    4. 将记忆上下文注入到 Agent 的 system prompt 中

    Agent 回复后：
    5. 将本轮对话（问+答）存入语义记忆向量库

    【与长期记忆的协作】
    - 长期记忆通过语义检索召回跨会话的相关片段
    - 这些片段注入到短期记忆的上下文增强中

    【Token 管理策略】
    - 动态窗口调整：根据剩余上下文空间自动调整窗口大小
    - 记忆上下文裁剪：智能选择最相关的记忆片段
    - 逐级降级：空间紧张时逐步压缩上下文
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._sessions: dict[str, SlidingWindowChatHistory] = {}
        self._semantic_store = SemanticMemoryStore()
        self._summarizer = ConversationSummarizer()
        
        # 增量缓存
        self._summary_cache: dict[str, str] = {}
        self._processed_messages_state: dict[str, list] = {}
        
        # Token 估算器（按字符数估算，1 token ≈ 1.5 中文字符）
        self._tokens_per_char = 1.5
        
        logger.info(f"短期记忆管理器初始化完成，窗口大小: {window_size}")

    def _count_tokens(self, text_or_messages) -> int:
        """估算文本或消息列表的 token 数

        Args:
            text_or_messages: 字符串或消息列表

        Returns:
            int: 估算的 token 数
        """
        if isinstance(text_or_messages, str):
            text = text_or_messages
        elif isinstance(text_or_messages, list):
            text = "".join(m.content for m in text_or_messages if hasattr(m, 'content'))
        else:
            return 0
        
        # 按字符数估算（1 token ≈ 1.5 中文字符或 4 英文字符）
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / self._tokens_per_char + other_chars / 4)

    def _calculate_effective_window(self, messages: list) -> int:
        """根据剩余上下文空间动态计算有效窗口大小

        Args:
            messages: 当前消息列表

        Returns:
            int: 有效的窗口大小
        """
        current_tokens = self._count_tokens(messages)
        available_tokens = MAX_CONTEXT_TOKENS - current_tokens - RESERVED_TOKENS
        
        if available_tokens < 2000:
            # 空间紧张，缩小窗口到最小值
            effective_window = MIN_WINDOW_SIZE
            logger.warning(f"上下文空间紧张（剩余 {available_tokens} token），"
                         f"窗口缩小到 {effective_window}")
        elif available_tokens < 4000:
            # 空间一般，缩小一半窗口
            effective_window = max(MIN_WINDOW_SIZE, self._window_size // 2)
            logger.info(f"上下文空间一般（剩余 {available_tokens} token），"
                      f"窗口调整为 {effective_window}")
        else:
            effective_window = self._window_size
        
        return effective_window

    def get_session_history(self, thread_id: str) -> SlidingWindowChatHistory:
        """获取或创建会话的对话历史

        Args:
            thread_id: 会话 ID

        Returns:
            SlidingWindowChatHistory: 会话的对话历史对象
        """
        if thread_id not in self._sessions:
            self._sessions[thread_id] = SlidingWindowChatHistory(
                thread_id=thread_id,
                window_size=self._window_size,
            )
        return self._sessions[thread_id]

    def process_memory(self, messages: list, thread_id: str, current_query: str) -> list:
        """处理记忆 — 核心方法，返回增强后的记忆上下文消息列表

        【处理逻辑】：
        1. 统计对话轮数（只计 HumanMessage）
        2. 根据剩余上下文空间动态计算有效窗口大小
        3. 如果超出有效窗口：
           - 分割为 [旧消息 | 最近 N 轮]
           - 旧消息 → 压缩为摘要（增量缓存）
           - 当前问题 → 语义检索相关历史
           - 组装记忆上下文
        4. 检查 token 数量，必要时逐级降级
        5. 返回合并后的消息列表

        Args:
            messages: 完整的对话历史消息列表（从 Checkpointer 恢复的）
            thread_id: 会话 ID
            current_query: 当前用户问题

        Returns:
            list: 合并后的消息列表，格式为：
                [SystemMessage(历史对话摘要+语义检索结果), ...最近N轮对话原文...]
        """
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        
        # 动态计算有效窗口大小
        effective_window = self._calculate_effective_window(messages)

        if human_count <= effective_window:
            # 对话轮数未超出窗口，直接返回原始消息列表
            return messages

        logger.info(
            f"对话轮数 {human_count} 超出有效窗口 {effective_window}，"
            f"启动摘要压缩 + 语义检索"
        )

        window_messages, old_messages = self._split_messages(messages, effective_window)
        summary = self._get_or_create_summary(thread_id, old_messages)
        relevant_memories = self._semantic_store.search_relevant_memories(
            thread_id=thread_id,
            query=current_query,
        )
        memory_context = self._build_memory_context(summary, relevant_memories)

        # 检查 token 数量，必要时逐级降级，返回合并后的消息列表
        # window_messages: 窗口内的消息列表（最近 N 轮对话原文）
        # memory_context: 记忆上下文字符串（摘要 + 语义检索结果）
        # effective_window: 动态计算的有效窗口大小（考虑了上下文空间后的实际窗口）
        return self._apply_token_constraint(
            window_messages, memory_context, effective_window
        )

    def save_conversation_turn(self, thread_id: str, human_msg: str, ai_msg: str) -> None:
        """保存一轮对话到语义记忆

        Args:
            thread_id: 会话 ID
            human_msg: 用户消息
            ai_msg: AI 回复

        Returns:
            None
        """
        if human_msg and ai_msg:
            self._semantic_store.add_conversation_turn(thread_id, human_msg, ai_msg)

    def _split_messages(self, messages: list, effective_window: Optional[int] = None) -> tuple[list, list]:
        """将消息列表分割为 [最近 N 轮] 和 [旧消息]

        【多维判断策略】
        同时考虑对话轮次和 Token 数量，以更严格的维度为准：
        - 先按对话轮次计算窗口位置
        - 再检查按轮次裁剪后的 token 数是否超出限制
        - 如果超出，继续往前裁剪直到满足 token 限制
        - 即使只有很少的对话轮次，如果每轮内容很长，也能正确处理

        Args:
            messages: 完整消息列表
            effective_window: 有效窗口大小（可选，默认为 self._window_size）

        Returns:
            tuple: (window_messages, old_messages)
        """
        window_size = effective_window if effective_window is not None else self._window_size
        human_indices = [
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        ]

        if not human_indices:
            return messages, []

        # 计算按轮次裁剪后的窗口位置
        if len(human_indices) <= window_size:
            window_start_idx = 0
        else:
            window_start_idx = human_indices[-window_size]

        # 多维判断：同时考虑轮次和 Token 数量
        window_messages = messages[window_start_idx:]
        old_messages = messages[:window_start_idx]

        # 检查 Token 数量是否超出限制
        # 预留空间给记忆上下文和系统提示
        max_window_tokens = MAX_CONTEXT_TOKENS - RESERVED_TOKENS - MAX_MEMORY_CONTEXT_TOKENS
        current_window_tokens = self._count_tokens(window_messages)

        if current_window_tokens > max_window_tokens:
            logger.info(
                f"按轮次裁剪后 Token 数（{current_window_tokens}）超出限制（{max_window_tokens}），"
                f"启动 Token 数量维度裁剪"
            )

            # 从头开始逐个移除消息，直到满足 Token 限制
            removed_messages = []
            while window_messages and current_window_tokens > max_window_tokens:
                removed_msg = window_messages.pop(0)
                removed_messages.append(removed_msg)
                current_window_tokens -= self._count_tokens([removed_msg])

            # 更新 old_messages：原来窗口前的消息 + 被 token 裁剪移除的消息
            old_messages = messages[:window_start_idx] + removed_messages

            logger.info(
                f"Token 数量维度裁剪完成，窗口缩小到 {len(window_messages)} 条消息，"
                f"约 {current_window_tokens} token"
            )

        return window_messages, old_messages

    def _apply_token_constraint(self, window_messages: list, memory_context: str, effective_window: int) -> list:
        """应用 token 约束，必要时逐级降级，返回合并后的消息列表

        【逐级降级策略】：
        Level 1（标准压缩）：正常处理，检查记忆上下文大小
        Level 2（深度压缩）：减少记忆上下文，只保留最重要的内容
        Level 3（极端压缩）：进一步缩小窗口，只保留核心摘要

        Args:
            window_messages: 窗口内的消息列表
            memory_context: 记忆上下文
            effective_window: 当前有效的窗口大小

        Returns:
            list: 合并后的消息列表（记忆上下文 + 最近N轮原文）
        """
        total_tokens = self._count_tokens(window_messages) + self._count_tokens(memory_context)

        if total_tokens > MAX_CONTEXT_TOKENS - RESERVED_TOKENS:
            # Level 2：深度压缩 - 减少记忆上下文
            logger.warning(f"记忆上下文超出限制（{total_tokens} token），启动深度压缩")
            
            memory_context = self._truncate_memory_context(memory_context, MAX_MEMORY_CONTEXT_TOKENS)
            total_tokens = self._count_tokens(window_messages) + self._count_tokens(memory_context)
            
            if total_tokens > MAX_CONTEXT_TOKENS - RESERVED_TOKENS:
                # Level 3：极端压缩 - 进一步缩小窗口（以 effective_window 为基准再减半）
                logger.error(f"仍超出限制（{total_tokens} token），启动极端压缩")
                
                # 使用 effective_window 的一半作为目标窗口，但不小于最小值
                reduced_window = max(MIN_WINDOW_SIZE, effective_window // 2)
                window_messages = self._reduce_window_size(window_messages, reduced_window)
                memory_context = self._extract_key_points(memory_context)
                
        # 将记忆上下文包装成 SystemMessage，插入到消息列表开头
        if memory_context:
            context_message = SystemMessage(content=f"【历史对话记忆】\n{memory_context}")
            return [context_message] + window_messages
        
        return window_messages

    def _truncate_memory_context(self, memory_context: str, max_tokens: int) -> str:
        """裁剪记忆上下文到指定的 token 数

        Args:
            memory_context: 原始记忆上下文
            max_tokens: 最大 token 数

        Returns:
            str: 裁剪后的记忆上下文
        """
        current_tokens = self._count_tokens(memory_context)
        if current_tokens <= max_tokens:
            return memory_context

        # 按比例裁剪
        ratio = max_tokens / current_tokens
        chars_to_keep = int(len(memory_context) * ratio)
        
        # 尽量保持完整的段落
        if chars_to_keep < len(memory_context):
            # 找到最后一个完整的分隔符
            separators = ["\n\n", "\n---\n", "\n【"]
            for sep in separators:
                idx = memory_context.rfind(sep, 0, chars_to_keep)
                if idx > 0:
                    chars_to_keep = idx
            
            memory_context = memory_context[:chars_to_keep].strip()
        
        logger.info(f"记忆上下文已裁剪，原 {current_tokens} token → 现 {self._count_tokens(memory_context)} token")
        return memory_context

    def _reduce_window_size(self, messages: list, target_window: int) -> list:
        """进一步缩小窗口大小

        Args:
            messages: 当前消息列表
            target_window: 目标窗口大小

        Returns:
            list: 缩小后的消息列表
        """
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        
        if len(human_indices) <= target_window:
            return messages
        
        window_start_idx = human_indices[-target_window]
        new_messages = messages[window_start_idx:]
        
        logger.info(f"窗口进一步缩小，原 {len(human_indices)} 轮 → 现 {target_window} 轮")
        return new_messages

    def _extract_key_points(self, memory_context: str) -> str:
        """提取记忆上下文中的关键点

        Args:
            memory_context: 原始记忆上下文

        Returns:
            str: 提取的关键点
        """
        if not memory_context:
            return ""

        # 尝试简单提取关键点
        lines = memory_context.split('\n')
        key_points = []
        
        for line in lines:
            # 提取包含数字、关键术语的行
            if any(char.isdigit() for char in line):
                key_points.append(line)
            elif any(term in line for term in ["摘要", "结论", "重要", "关键", "注意"]):
                key_points.append(line)
        
        if key_points:
            result = "【关键信息】\n" + "\n".join(key_points[:5])  # 最多保留5条
        else:
            # 如果没有明显的关键点，取前200字
            result = "【关键信息】\n" + memory_context[:200] + "..."
        
        return result

    def _get_or_create_summary(self, thread_id: str, old_messages: list) -> str:
        """获取或创建对话摘要（增量缓存策略）

        【增量摘要缓存策略】：
        - 记录每个 thread 已处理的旧消息状态
        - 只对新增加的旧消息进行压缩
        - 增量更新缓存的摘要
        - 用户重复提问100次时，第2-100次直接命中缓存

        Args:
            thread_id: 会话 ID
            old_messages: 需要压缩的旧消息列表

        Returns:
            str: 对话摘要文本
        """
        if not old_messages:
            return ""

        processed_messages = self._processed_messages_state.get(thread_id, [])
        new_old_messages = self._get_new_old_messages(old_messages, processed_messages)
        last_summary = self._summary_cache.get(f"{thread_id}_last", "")

        if not new_old_messages:
            logger.info(f"无新增旧消息，使用缓存的摘要（长度: {len(last_summary)} 字）")
            return last_summary

        logger.info(f"正在压缩 {len(new_old_messages)} 条新增旧消息为摘要...")
        new_summary = self._summarizer.summarize(new_old_messages)

        if last_summary:
            combined_summary = self._merge_summaries(last_summary, new_summary)
        else:
            combined_summary = new_summary

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

        processed_hashes = {
            self._calculate_message_hash(msg) for msg in processed_messages
        }

        return [
            msg for msg in current_old_messages
            if self._calculate_message_hash(msg) not in processed_hashes
        ]

    def _calculate_message_hash(self, msg: BaseMessage) -> str:
        """计算消息的哈希值用于去重"""
        content = msg.content if hasattr(msg, 'content') else str(msg)
        return hashlib.sha256(content.encode()).hexdigest()

    def _merge_summaries(self, old_summary: str, new_summary: str) -> str:
        """合并两个摘要

        Args:
            old_summary: 旧的摘要
            new_summary: 新的摘要

        Returns:
            str: 合并后的摘要
        """
        if not old_summary:
            return new_summary
        if not new_summary:
            return old_summary

        combined = f"{old_summary}\n\n{new_summary}"
        if self._count_chinese_chars(combined) <= 300:
            return combined

        return self._recompress(old_summary, new_summary)

    def _count_chinese_chars(self, text: str) -> int:
        """统计中文字符数"""
        return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

    def _recompress(self, old_summary: str, new_summary: str) -> str:
        """重新压缩合并后的摘要"""
        merge_prompt = [
            SystemMessage(content=(
                "你是一个对话摘要助手。请将两个对话摘要合并为一个简洁的摘要。\n\n"
                "要求：\n"
                "1. 保留所有关键信息\n"
                "2. 摘要不超过 300 字\n"
                "3. 使用第三人称描述\n"
                "4. 按时间顺序组织"
            )),
            HumanMessage(content=f"摘要1:\n{old_summary}\n\n摘要2:\n{new_summary}"),
        ]
        response = self._llm.invoke(merge_prompt)
        return response.content

    def _build_memory_context(self, summary: str, relevant_memories: list[str]) -> str:
        """组装记忆上下文

        Args:
            summary: 对话摘要
            relevant_memories: 语义检索到的相关历史片段

        Returns:
            str: 完整的记忆上下文字符串
        """
        parts = []

        if summary:
            parts.append(f"【历史对话摘要】\n{summary}")

        if relevant_memories:
            memories_text = "\n---\n".join(relevant_memories)
            parts.append(f"【相关历史对话】\n{memories_text}")

        return "\n\n".join(parts) if parts else ""

    def get_session_info(self, thread_id: str) -> dict:
        """获取会话信息

        Args:
            thread_id: 会话 ID

        Returns:
            dict: 会话统计信息
        """
        history = self.get_session_history(thread_id)
        human_count = sum(1 for m in history.messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in history.messages if isinstance(m, AIMessage))
        return {
            "thread_id": thread_id,
            "human_count": human_count,
            "ai_count": ai_count,
            "total_messages": len(history.messages),
            "window_size": self._window_size,
        }


# ============================================================
# 全局单例
# ============================================================

_short_term_memory_manager: Optional[ShortTermMemoryManager] = None


def get_short_term_memory_manager() -> ShortTermMemoryManager:
    """获取短期记忆管理器的全局单例

    Returns:
        ShortTermMemoryManager: 短期记忆管理器实例
    """
    global _short_term_memory_manager
    if _short_term_memory_manager is None:
        _short_term_memory_manager = ShortTermMemoryManager()
    return _short_term_memory_manager

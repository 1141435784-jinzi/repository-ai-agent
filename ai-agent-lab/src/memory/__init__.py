"""
=== 记忆系统 ===

【功能】
1. 短期记忆：基于 LangChain 的三层记忆架构（滑动窗口 + 摘要压缩 + 语义检索）
2. 长期记忆：语义检索 + 用户画像 + Rerank 重排序
3. 状态记忆：基于 Checkpointer 的跨会话状态持久化

【架构】
- 短期记忆 (short_term_memory.py): 三层架构
  - SemanticMemoryStore: 对话历史向量化存储与检索
  - ConversationSummarizer: 对话摘要压缩器
  - SlidingWindowChatHistory: 滑动窗口对话历史
  - ShortTermMemoryManager: 整合三层的管理器
- 长期记忆 (long_term_memory.py): 跨会话记忆
  - SemanticMemoryStore: 语义记忆（跨会话检索）
  - UserProfileStore: 用户画像存储
  - LongTermMemoryManager: 长期记忆管理器
- 状态记忆 (checkpointer.py): MemorySaver / AsyncPostgresSaver

【使用方式】
```python
from src.memory import (
    get_short_term_memory_manager,
    get_long_term_memory_manager,
    get_async_checkpointer,
    close_async_pool,
    ShortTermMemoryManager,
    LongTermMemoryManager,
    SlidingWindowChatHistory,
)

# 短期记忆管理 - 核心方法
short_term_mgr = get_short_term_memory_manager()
result = short_term_mgr.process_memory(messages, thread_id, current_query)
# result = {
#     "trimmed_messages": [...],  # 裁剪后的消息列表（最近 N 轮）
#     "memory_context": "...",    # 记忆上下文（摘要 + 语义检索结果）
#     "needs_trim": True/False    # 是否进行了裁剪
# }

# 保存对话到语义记忆
short_term_mgr.save_conversation_turn(thread_id, human_msg, ai_msg)

# 长期记忆管理
long_term_mgr = get_long_term_memory_manager()
long_term_mgr.save_conversation_turn(user_id, human_msg, ai_msg)
recalled = long_term_mgr.recall_relevant_memories(user_id, query)
profile = long_term_mgr.get_user_profile(user_id)

# 状态检查点
checkpointer = await get_async_checkpointer()
await close_async_pool()
```
"""

from .short_term_memory import (
    SemanticMemoryStore as ShortTermSemanticMemoryStore,
    SlidingWindowChatHistory,
    ShortTermMemoryManager,
    ConversationSummarizer,
    get_short_term_memory_manager,
)
from .long_term_memory import (
    SemanticMemoryStore,
    UserProfileStore,
    LongTermMemoryManager,
    get_long_term_memory_manager,
)
from .checkpointer import get_async_checkpointer, close_async_pool

__all__ = [
    "ShortTermSemanticMemoryStore",
    "SlidingWindowChatHistory",
    "ShortTermMemoryManager",
    "ConversationSummarizer",
    "get_short_term_memory_manager",
    "SemanticMemoryStore",
    "UserProfileStore",
    "LongTermMemoryManager",
    "get_long_term_memory_manager",
    "get_async_checkpointer",
    "close_async_pool",
]

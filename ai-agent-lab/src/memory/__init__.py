"""
=== 记忆系统 ===

【功能】：
1. 对话记忆：管理用户与 Agent 的对话历史
2. 状态记忆：持久化 Agent 状态和检查点
3. 记忆管理器：统一协调记忆系统

【使用方式】：
```python
from src.memory import get_memory_manager, get_async_checkpointer, close_async_pool

# 获取对话记忆管理器
memory_manager = get_memory_manager()

# 获取状态检查点管理器
checkpointer = await get_async_checkpointer()

# 关闭异步连接池（在应用关闭时调用）
await close_async_pool()
```

【模块】：
- conversation.py: 对话记忆管理
- checkpointer.py: 状态记忆和检查点（基于 PostgreSQL AsyncPostgresSaver）
- manager.py: 记忆系统管理器
"""

from .manager import (
    get_memory_manager,
    get_async_checkpointer,
    ConversationMemoryManager,
)
from .checkpointer import close_async_pool

# 导出列表
__all__ = [
    "get_memory_manager",
    "get_async_checkpointer",
    "ConversationMemoryManager",
    "close_async_pool",
]
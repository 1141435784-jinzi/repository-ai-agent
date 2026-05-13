"""
=== 记忆管理器 ===

【功能】：
1. 统一管理对话记忆和状态记忆
2. 提供简化的记忆访问接口
3. 协调不同类型的记忆系统

【架构】：
用户对话 → 对话记忆（短期） → 状态记忆（长期） → 记忆管理器
"""

from .conversation import ConversationMemoryManager
from .checkpointer import get_async_checkpointer

# 全局记忆管理器实例
_memory_manager = None

def get_memory_manager() -> ConversationMemoryManager:
    """获取全局记忆管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationMemoryManager()
    return _memory_manager

# 导出接口
__all__ = [
    "get_memory_manager",
    "get_async_checkpointer",
    "ConversationMemoryManager",
]
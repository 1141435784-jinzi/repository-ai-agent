"""
记忆服务

【功能】：
1. 会话记忆管理
2. 记忆获取
3. 记忆清除

【设计原则】：
1. 异步处理：支持异步操作
2. 持久化支持：支持记忆持久化
3. 清理机制：支持过期记忆清理
"""

import logging
from typing import Optional

from src.memory.manager import get_memory_manager

logger = logging.getLogger(__name__)


class MemoryService:
    """记忆服务类"""

    async def get_memory(self, thread_id: str):
        """
        获取会话记忆
        
        Args:
            thread_id: 会话 ID
            
        Returns:
            dict: 记忆内容
        """
        memory_manager = get_memory_manager()
        try:
            memory = await memory_manager.get_memory(thread_id)
            return memory
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            raise

    async def clear_memory(self, thread_id: str):
        """
        清除会话记忆
        
        Args:
            thread_id: 会话 ID
        """
        memory_manager = get_memory_manager()
        try:
            await memory_manager.clear_memory(thread_id)
        except Exception as e:
            logger.error(f"清除记忆失败: {e}")
            raise

    async def add_memory(self, thread_id: str, key: str, value):
        """
        添加记忆项
        
        Args:
            thread_id: 会话 ID
            key: 记忆键
            value: 记忆值
        """
        memory_manager = get_memory_manager()
        try:
            await memory_manager.add_memory(thread_id, key, value)
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            raise

    async def update_memory(self, thread_id: str, key: str, value):
        """
        更新记忆项
        
        Args:
            thread_id: 会话 ID
            key: 记忆键
            value: 记忆值
        """
        memory_manager = get_memory_manager()
        try:
            await memory_manager.update_memory(thread_id, key, value)
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            raise

    async def delete_memory_item(self, thread_id: str, key: str):
        """
        删除记忆项
        
        Args:
            thread_id: 会话 ID
            key: 记忆键
        """
        memory_manager = get_memory_manager()
        try:
            await memory_manager.delete_memory_item(thread_id, key)
        except Exception as e:
            logger.error(f"删除记忆项失败: {e}")
            raise

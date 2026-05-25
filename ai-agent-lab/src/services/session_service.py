"""
会话服务

【功能】：
1. 会话创建
2. 会话查询
3. 会话删除
4. 会话管理

【设计原则】：
1. 会话隔离：每个会话独立管理
2. 持久化支持：支持会话状态持久化
3. 清理机制：支持过期会话清理
"""

import uuid
from datetime import datetime
from typing import Optional, Dict

from src.memory.manager import get_memory_manager


class SessionService:
    """会话服务类"""

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self, user_id: str) -> str:
        """
        创建新会话
        
        Args:
            user_id: 用户 ID
            
        Returns:
            str: 会话 ID
        """
        thread_id = f"thread_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.sessions[thread_id] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "status": "active"
        }
        return thread_id

    def get_session(self, thread_id: str) -> Optional[dict]:
        """
        获取会话信息（自动验证用户归属）
        
        Args:
            thread_id: 会话 ID（格式：thread_{user_id}_{timestamp}）
            
        Returns:
            dict or None: 会话信息（仅当会话属于正确用户时返回）
        """
        session = self.sessions.get(thread_id)
        if session:
            # 从 thread_id 解析用户 ID 并验证归属
            user_id = self._extract_user_id(thread_id)
            if session.get("user_id") == user_id:
                session["last_active"] = datetime.now().isoformat()
                return session
        return None

    def _extract_user_id(self, thread_id: str) -> str:
        """从 thread_id 中解析用户 ID"""
        parts = thread_id.split('_')
        if len(parts) >= 2:
            return parts[1]
        return ""

    def delete_session(self, thread_id: str, user_id: Optional[str] = None) -> bool:
        """
        删除会话（自动验证用户归属）
        
        Args:
            thread_id: 会话 ID（格式：thread_{user_id}_{timestamp}）
            user_id: 用户 ID（可选，用于向后兼容）
            
        Returns:
            bool: 是否删除成功
        """
        session = self.sessions.get(thread_id)
        if session:
            # 如果未提供 user_id，从 thread_id 自动解析
            actual_user_id = user_id if user_id else self._extract_user_id(thread_id)
            if session.get("user_id") == actual_user_id:
                del self.sessions[thread_id]
                # 同时清除记忆
                asyncio.create_task(self._clear_memory(thread_id))
                return True
        return False

    async def _clear_memory(self, thread_id: str):
        """清除会话记忆"""
        try:
            memory_manager = get_memory_manager()
            await memory_manager.clear_memory(thread_id)
        except Exception:
            pass

    def list_sessions(self) -> list:
        """
        获取所有会话列表
        
        Returns:
            list: 会话列表
        """
        return list(self.sessions.values())

    def cleanup_expired_sessions(self, hours: int = 24) -> int:
        """
        清理过期会话
        
        Args:
            hours: 过期时间（小时）
            
        Returns:
            int: 清理的会话数量
        """
        now = datetime.now()
        expired_count = 0
        
        expired_thread_ids = []
        for thread_id, session in self.sessions.items():
            last_active = datetime.fromisoformat(session["last_active"])
            if (now - last_active).total_seconds() > hours * 3600:
                expired_thread_ids.append(thread_id)
        
        for thread_id in expired_thread_ids:
            self.delete_session(thread_id)
            expired_count += 1
        
        return expired_count

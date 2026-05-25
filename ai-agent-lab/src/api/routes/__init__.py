"""
API 路由模块

【功能】：
1. 按功能划分路由模块
2. 统一路由注册
3. 支持模块化扩展

【模块说明】：
- chat.py: 聊天相关路由（流式聊天、会话管理）
- tools.py: 工具管理路由（MCP状态、工具列表）
- rag.py: RAG相关路由（文档上传、索引管理）
- memory.py: 记忆管理路由（会话记忆操作）
"""

from fastapi import APIRouter

# 导入各路由模块
from .chat import router as chat_router
from .tools import router as tools_router
from .rag import router as rag_router
from .memory import router as memory_router

# 创建主路由器（保持向后兼容，不添加版本前缀）
api_router = APIRouter()

# 注册子路由
api_router.include_router(chat_router, tags=["聊天"])
api_router.include_router(rag_router, tags=["RAG"])
api_router.include_router(memory_router, tags=["记忆"])
api_router.include_router(tools_router, tags=["工具"])

__all__ = [
    "api_router",
    "chat_router",
    "tools_router",
    "rag_router",
    "memory_router",
]

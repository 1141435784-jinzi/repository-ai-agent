"""
记忆路由模块

【功能】：
1. 获取会话记忆
2. 清除会话记忆
3. 记忆管理

【接口列表】：
- GET /memory/{thread_id} - 获取会话记忆
- DELETE /memory/{thread_id} - 清除会话记忆
"""

from fastapi import APIRouter, HTTPException

from src.services.memory_service import MemoryService

router = APIRouter(prefix="/memory")


@router.get("/{thread_id}")
async def get_memory(thread_id: str):
    """获取指定会话的记忆"""
    memory_service = MemoryService()
    try:
        memory = await memory_service.get_memory(thread_id)
        return {"thread_id": thread_id, "memory": memory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{thread_id}")
async def clear_memory(thread_id: str):
    """清除指定会话的记忆"""
    memory_service = MemoryService()
    try:
        await memory_service.clear_memory(thread_id)
        return {"success": True, "message": f"记忆已清除: {thread_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
聊天路由模块

【功能】：
1. 流式聊天接口
2. 会话管理（创建、查询、删除）
3. 聊天历史管理

【接口列表】：
- POST /chat/stream - 流式聊天
- POST /session - 创建新会话
- GET /session/{thread_id} - 获取会话信息
- DELETE /session/{thread_id} - 删除会话
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import DEEPSEEK_API_KEY
from src.services.chat_service import ChatService
from src.services.session_service import SessionService

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., description="用户消息")
    thread_id: Optional[str] = Field(None, description="会话线程 ID")


class ChatResponse(BaseModel):
    """聊天响应体"""
    response: str = Field(..., description="AI 响应")
    thread_id: str = Field(..., description="会话线程 ID")
    sources: list = Field(default_factory=list, description="引用的知识库来源")
    found_in_kb: bool = Field(False, description="是否在知识库中找到相关信息")


class SessionResponse(BaseModel):
    """新建会话响应体"""
    thread_id: str = Field(..., description="新创建的会话 ID")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 - 实时返回AI响应"""
    try:
        chat_service = ChatService()
        return await chat_service.stream_chat(request.message, request.thread_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", response_model=SessionResponse)
async def create_session():
    """创建新会话"""
    session_service = SessionService()
    thread_id = session_service.create_session()
    return SessionResponse(thread_id=thread_id)


@router.get("/session/{thread_id}")
async def get_session(thread_id: str):
    """获取会话信息"""
    session_service = SessionService()
    session = session_service.get_session(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {thread_id}")
    return session


@router.delete("/session/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话"""
    session_service = SessionService()
    success = session_service.delete_session(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"会话不存在: {thread_id}")
    return {"success": True, "message": f"会话已删除: {thread_id}"}

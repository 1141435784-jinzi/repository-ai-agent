"""
=== 聊天相关数据模型 ===
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class Message(BaseModel):
    """单条消息模型"""
    role: str = Field(..., description="消息角色：user, assistant, system, tool")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="消息时间戳")
    tool_call: Optional[Dict[str, Any]] = Field(None, description="工具调用信息")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="工具执行结果")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户输入消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    stream: Optional[bool] = Field(False, description="是否流式响应")
    model: Optional[str] = Field(None, description="指定LLM模型")
    temperature: Optional[float] = Field(0.7, description="温度参数")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str = Field(..., description="助手回复内容")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    tool_used: Optional[List[str]] = Field(None, description="使用的工具列表")
    rag_sources: Optional[List[str]] = Field(None, description="RAG检索来源")


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str = Field(..., description="会话ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    last_active_at: datetime = Field(default_factory=datetime.now, description="最后活跃时间")
    message_count: int = Field(0, description="消息数量")


class StreamResponse(BaseModel):
    """流式响应模型"""
    chunk: str = Field(..., description="响应片段")
    session_id: str = Field(..., description="会话ID")
    done: bool = Field(False, description="是否完成")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    code: Optional[int] = Field(None, description="错误码")
    detail: Optional[str] = Field(None, description="错误详情")
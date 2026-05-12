"""
=== Agent 相关数据模型 ===
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class AgentConfig(BaseModel):
    """Agent 配置模型"""
    name: str = Field(..., description="Agent名称")
    description: str = Field(..., description="Agent描述")
    role: str = Field(..., description="Agent角色")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    max_iterations: int = Field(20, description="最大迭代次数")
    temperature: float = Field(0.7, description="温度参数")


class AgentState(BaseModel):
    """Agent 状态模型"""
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="对话消息")
    trimmed_messages: List[Dict[str, Any]] = Field(default_factory=list, description="裁剪后的消息")
    memory_context: Dict[str, Any] = Field(default_factory=dict, description="记忆上下文")
    route: Optional[str] = Field(None, description="路由结果")
    rag_context: Optional[str] = Field(None, description="RAG检索上下文")
    rag_sources: List[str] = Field(default_factory=list, description="RAG来源")


class ToolCall(BaseModel):
    """工具调用模型"""
    tool_name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    thought: Optional[str] = Field(None, description="调用思考")


class ToolResult(BaseModel):
    """工具执行结果模型"""
    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否成功")
    result: Any = Field(None, description="执行结果")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(0.0, description="执行耗时(秒)")


class RAGContext(BaseModel):
    """RAG 上下文模型"""
    context: str = Field(..., description="检索到的上下文内容")
    sources: List[str] = Field(default_factory=list, description="来源文档列表")
    scores: List[float] = Field(default_factory=list, description="相似度分数")


class MemoryEntry(BaseModel):
    """记忆条目模型"""
    id: str = Field(..., description="条目ID")
    content: str = Field(..., description="记忆内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    relevance_score: float = Field(0.0, description="相关度分数")
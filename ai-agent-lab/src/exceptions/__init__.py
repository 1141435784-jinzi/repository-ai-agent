"""
统一异常处理模块

【功能】：
1. 定义统一的异常类
2. 提供异常处理工具函数
3. 支持异常转换和包装

【设计原则】：
1. 分层设计：按功能模块划分异常类
2. 可追溯性：保留原始异常信息
3. 友好提示：提供用户友好的错误消息
"""

from .base import (
    AgentException,
    AgentError,
    ConfigurationError,
    ValidationError,
    ServiceUnavailableError,
    ResourceNotFoundError,
)

from .llm import (
    LLMException,
    LLMConnectionError,
    LLMTimeoutError,
    LLMQuotaExceededError,
    LLMModelNotFoundError,
)

from .rag import (
    RAGException,
    DocumentNotFoundError,
    DocumentParseError,
    IndexError,
    EmbeddingError,
)

from .tools import (
    ToolException,
    ToolNotFoundError,
    ToolExecutionError,
    MCPConnectionError,
    MCPServerError,
)

# 便捷函数
def handle_exception(e: Exception, message: str = None):
    """
    统一异常处理函数
    
    Args:
        e: 异常对象
        message: 自定义错误消息
        
    Returns:
        AgentException: 包装后的异常
    """
    from .base import AgentException
    
    if isinstance(e, AgentException):
        return e
    
    if message is None:
        message = str(e)
    
    return AgentException(message, original_exception=e)


__all__ = [
    # 基础异常
    "AgentException",
    "AgentError",
    "ConfigurationError",
    "ValidationError",
    "ServiceUnavailableError",
    "ResourceNotFoundError",
    
    # LLM 异常
    "LLMException",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMQuotaExceededError",
    "LLMModelNotFoundError",
    
    # RAG 异常
    "RAGException",
    "DocumentNotFoundError",
    "DocumentParseError",
    "IndexError",
    "EmbeddingError",
    
    # 工具异常
    "ToolException",
    "ToolNotFoundError",
    "ToolExecutionError",
    "MCPConnectionError",
    "MCPServerError",
    
    # 工具函数
    "handle_exception",
]

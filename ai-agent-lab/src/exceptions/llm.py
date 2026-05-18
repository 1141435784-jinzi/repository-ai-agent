"""
LLM 相关异常类

【功能】：
1. LLM 连接异常
2. LLM 超时异常
3. LLM 配额超限异常
4. LLM 模型未找到异常
"""

from .base import AgentError


class LLMException(AgentError):
    """LLM 基础异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "LLM_ERROR", original_exception)


class LLMConnectionError(LLMException):
    """LLM 连接错误"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, original_exception)
        self.error_code = "LLM_CONNECTION_ERROR"


class LLMTimeoutError(LLMException):
    """LLM 超时错误"""
    
    def __init__(self, message: str = "LLM 请求超时", original_exception: Exception = None):
        super().__init__(message, original_exception)
        self.error_code = "LLM_TIMEOUT"


class LLMQuotaExceededError(LLMException):
    """LLM 配额超限错误"""
    
    def __init__(self, message: str = "LLM 配额已用尽", original_exception: Exception = None):
        super().__init__(message, original_exception)
        self.error_code = "LLM_QUOTA_EXCEEDED"


class LLMModelNotFoundError(LLMException):
    """LLM 模型未找到错误"""
    
    def __init__(self, model_name: str, original_exception: Exception = None):
        message = f"未找到指定的 LLM 模型: {model_name}"
        super().__init__(message, original_exception)
        self.error_code = "LLM_MODEL_NOT_FOUND"
        self.model_name = model_name

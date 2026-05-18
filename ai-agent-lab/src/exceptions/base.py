"""
基础异常类

【设计原则】：
1. 继承关系清晰
2. 保留原始异常信息
3. 支持错误码和错误消息
"""

class AgentError(Exception):
    """Agent 系统基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AGENT_ERROR"
        self.original_exception = original_exception
    
    def __str__(self):
        if self.original_exception:
            return f"{self.message} (Original: {self.original_exception})"
        return self.message


class AgentException(AgentError):
    """Agent 异常（通用）"""
    
    def __init__(self, message: str, error_code: str = "AGENT_EXCEPTION", original_exception: Exception = None):
        super().__init__(message, error_code, original_exception)


class ConfigurationError(AgentError):
    """配置错误异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "CONFIGURATION_ERROR", original_exception)


class ValidationError(AgentError):
    """数据验证错误异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "VALIDATION_ERROR", original_exception)


class ServiceUnavailableError(AgentError):
    """服务不可用异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "SERVICE_UNAVAILABLE", original_exception)


class ResourceNotFoundError(AgentError):
    """资源未找到异常"""
    
    def __init__(self, resource_name: str, resource_id: str = None):
        message = f"资源未找到: {resource_name}"
        if resource_id:
            message += f" (ID: {resource_id})"
        super().__init__(message, "RESOURCE_NOT_FOUND")
        self.resource_name = resource_name
        self.resource_id = resource_id

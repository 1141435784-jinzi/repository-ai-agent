"""
工具相关异常类

【功能】：
1. 工具基础异常
2. 工具未找到异常
3. 工具执行异常
4. MCP 连接异常
5. MCP 服务器异常
"""

from .base import AgentError


class ToolException(AgentError):
    """工具基础异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "TOOL_ERROR", original_exception)


class ToolNotFoundError(ToolException):
    """工具未找到异常"""
    
    def __init__(self, tool_name: str, tool_type: str = None):
        message = f"工具未找到: {tool_name}"
        if tool_type:
            message += f" (类型: {tool_type})"
        super().__init__(message)
        self.error_code = "TOOL_NOT_FOUND"
        self.tool_name = tool_name
        self.tool_type = tool_type


class ToolExecutionError(ToolException):
    """工具执行异常"""
    
    def __init__(self, tool_name: str, original_exception: Exception = None):
        message = f"工具执行失败: {tool_name}"
        super().__init__(message, original_exception)
        self.error_code = "TOOL_EXECUTION_ERROR"
        self.tool_name = tool_name


class MCPConnectionError(ToolException):
    """MCP 连接异常"""
    
    def __init__(self, server_name: str, original_exception: Exception = None):
        message = f"MCP 服务器连接失败: {server_name}"
        super().__init__(message, original_exception)
        self.error_code = "MCP_CONNECTION_ERROR"
        self.server_name = server_name


class MCPServerError(ToolException):
    """MCP 服务器异常"""
    
    def __init__(self, server_name: str, message: str = None):
        if message:
            full_message = f"MCP 服务器错误 [{server_name}]: {message}"
        else:
            full_message = f"MCP 服务器错误: {server_name}"
        super().__init__(full_message)
        self.error_code = "MCP_SERVER_ERROR"
        self.server_name = server_name

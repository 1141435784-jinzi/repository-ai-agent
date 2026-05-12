"""
=== MCP 工具系统（基于官方库，使用schema自动验证）===

【功能】：
1. 基于官方 mcp 库的动态工具发现
2. 使用 MCP 工具的 inputSchema 自动验证参数
3. 自动集成到 LangChain 工具系统
4. 配置驱动的 MCP 服务器管理
5. 提供统一的工具调用接口

【使用方式】：
```python
# 获取所有可用的 LangChain 工具
from src.tools import get_all_tools

async def main():
    tools = await get_all_tools()
    print(f"共有 {len(tools)} 个工具可用")
    
# 直接调用 MCP 工具（参数自动验证）
from src.tools import call_tool

async def get_weather():
    # 参数会根据 MCP 工具的 schema 自动验证
    result = await call_tool("mcp.weather_cn.query_weather_cn", city="深圳")
    print(result)
    
# 管理 MCP 服务器
from src.tools import get_mcp_manager, is_mcp_available

async def manage_servers():
    manager = await get_mcp_manager()
    servers = await manager.list_servers()
    print(f"MCP 可用: {await is_mcp_available()}")
    print(f"服务器列表: {servers}")
```

【架构】：
MCP 服务器 → MCP 客户端管理器 → LangChain 工具 → Agent

【特点】：
1. 使用官方 mcp 库（版本 1.27.0）
2. 基于 MCP 工具的 inputSchema 自动验证参数，无硬编码
3. 动态创建 Pydantic 模型进行参数验证
4. 异步设计，高性能
5. 支持动态工具发现
6. 简化的工具调用接口
"""

# 导出 MCP 集成功能（基于官方库，使用schema自动验证）
from .mcp_client import (
    # MCP 客户端管理器接口
    get_mcp_client_manager as get_mcp_manager,
    close_mcp_client_manager as close_mcp_manager,
    is_mcp_available,
    
    # MCP 工具接口
    get_available_mcp_tools,
    call_mcp_tool,
    get_langchain_tools_from_mcp,
    
    # 工具管理函数
    refresh_tools,
    get_metrics,
    get_tool_info,
    
    # 类型定义
    MCPServerConfig,
)

# 简化接口 - 直接使用 MCP 客户端功能
get_all_tools = get_langchain_tools_from_mcp

# 导入 time 模块用于 get_metrics
import time

# 导出列表
__all__ = [
    # 主接口
    "get_all_tools",
    "call_mcp_tool",
    
    # 管理接口
    "get_mcp_manager",
    "close_mcp_manager",
    "is_mcp_available",
    "refresh_tools",
    "get_tool_info",
    "get_metrics",
    
    # MCP 工具接口
    "get_available_mcp_tools",
    "get_langchain_tools_from_mcp",
    
    # 类型定义
    "MCPServerConfig",
]

# 文档字符串
__doc__ = """
MCP 工具系统（简化版）

这个模块提供了基于官方 mcp 库的工具发现和调用功能。
所有工具都通过 MCP 服务器动态发现，无需硬编码。

核心功能：
1. 使用官方 mcp 库实现完整的 MCP 协议支持
2. 配置驱动的 MCP 服务器管理
3. 自动创建 LangChain 工具
4. 简化的工具调用接口

MCP 工具会自动被发现和注册，新增 MCP 服务器时只需要：
1. 安装 MCP 服务器包
2. 在 mcp_config.json 中配置服务器
3. 系统会自动发现和注册工具
"""

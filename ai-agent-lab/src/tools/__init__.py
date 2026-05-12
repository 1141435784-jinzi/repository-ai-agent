"""
=== 工具系统 ===

【功能】：
1. 统一管理所有类型的工具
2. 提供统一的工具发现和调用接口
3. 支持多种工具类型：MCP、API、本地工具

【架构】：
工具系统 → 工具管理器 → 工具类型
    ├── MCP 工具 (mcp/)     - 基于 MCP 协议的动态工具
    ├── API 工具 (api/)     - RESTful API 工具
    └── 本地工具 (local/)   - 本地函数和脚本工具

【使用方式】：
```python
# 获取所有可用的工具
from src.tools import get_all_tools

async def main():
    tools = await get_all_tools()
    print(f"共有 {len(tools)} 个工具可用")
    
# 按类型获取工具
from src.tools import get_mcp_tools, get_api_tools, get_local_tools

async def get_tools_by_type():
    mcp_tools = await get_mcp_tools()
    api_tools = await get_api_tools()
    local_tools = await get_local_tools()
    
# 调用特定工具
from src.tools import call_tool

async def use_tool():
    # 调用 MCP 工具
    result = await call_tool("mcp.weather_cn.query_weather_cn", city="深圳")
    
    # 调用 API 工具
    result = await call_tool("api.github.get_user", username="octocat")
    
    # 调用本地工具
    result = await call_tool("local.calculator.add", a=5, b=3)
```

【设计原则】：
1. 统一接口：所有工具通过相同接口访问
2. 类型安全：工具参数和返回值类型安全
3. 易于扩展：新增工具类型只需添加对应模块
4. 配置驱动：工具配置集中管理
"""

import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

# 导入各类型工具模块
from .mcp import (
    # MCP 客户端管理器接口
    get_mcp_manager,
    close_mcp_manager,
    is_mcp_available,
    
    # MCP 工具接口
    get_available_mcp_tools as _get_mcp_tools,
    call_mcp_tool,
    get_langchain_tools_from_mcp as _get_mcp_langchain_tools,
    
    # 工具管理函数
    refresh_tools as refresh_mcp_tools,
    get_metrics as get_mcp_metrics,
    get_tool_info as get_mcp_tool_info,
    
    # 类型定义
    MCPServerConfig,
)

# 导入新的动态工具管理器
from .tool_manager import (
    DynamicToolManager,
    tool_manager,
    get_tool_system_stats,
    refresh_tool_system,
    get_tool_info,
)

# 保持向后兼容的 ToolManager 别名
ToolManager = DynamicToolManager

# 使用新的动态工具管理器
async def get_all_tools() -> List[BaseTool]:
    """获取所有工具（简化接口）"""
    return await tool_manager.get_tools()

async def get_mcp_tools() -> List[BaseTool]:
    """获取 MCP 工具"""
    # 直接调用 MCP 模块的函数，避免循环
    from .mcp import get_langchain_tools_from_mcp
    return await get_langchain_tools_from_mcp()

async def get_api_tools() -> List[BaseTool]:
    """获取 API 工具"""
    # 直接调用 API 模块的函数，避免循环
    from .api import create_all_api_tools
    return create_all_api_tools()

async def get_local_tools() -> List[BaseTool]:
    """获取本地工具"""
    # 直接调用本地模块的函数，避免循环
    from .local import create_calculator_tools
    return create_calculator_tools()

async def refresh_tools():
    """刷新所有工具"""
    await tool_manager.refresh_tools()

async def call_tool(tool_name: str, **kwargs) -> Any:
    """调用工具（统一接口）"""
    # 解析工具类型和名称
    if "." not in tool_name:
        raise ValueError(f"工具名称格式错误，应为type.tool_name: {tool_name}")
    
    tool_type, actual_tool_name = tool_name.split(".", 1)
    
    if tool_type == "mcp":
        # 调用 MCP 工具
        from .mcp import call_mcp_tool
        return await call_mcp_tool(actual_tool_name, **kwargs)
    elif tool_type == "api":
        # 调用 API 工具
        from .api import call_api_tool
        return await call_api_tool(actual_tool_name, **kwargs)
    elif tool_type == "local":
        # 调用本地工具
        from .local import call_local_tool
        return await call_local_tool(actual_tool_name, **kwargs)
    else:
        raise ValueError(f"未知的工具类型: {tool_type}")

# 导出列表
__all__ = [
    # 工具管理器
    "ToolManager",
    "DynamicToolManager",
    "tool_manager",
    
    # 工具管理 API
    "get_tool_system_stats",
    "refresh_tool_system",
    "get_tool_info",
    
    # 工具获取接口
    "get_all_tools",
    "get_mcp_tools",
    "get_api_tools",
    "get_local_tools",
    
    # 工具操作接口
    "refresh_tools",
    "call_tool",
    
    # MCP 相关（保持兼容性）
    "get_mcp_manager",
    "close_mcp_manager",
    "is_mcp_available",
    "get_mcp_tool_info",
    "get_mcp_metrics",
    "MCPServerConfig",
]
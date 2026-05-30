"""
工具服务

【功能】：
1. MCP 服务管理
2. 工具列表查询
3. 工具执行

【设计原则】：
1. 统一接口：所有工具通过相同接口访问
2. 类型安全：工具参数和返回值类型安全
3. 易于扩展：新增工具类型只需添加对应模块
"""

import logging
from typing import Optional, List

from src.tools import (
    get_mcp_manager,
    is_mcp_available,
    get_all_tools,
    call_tool,
    tool_api,
)

logger = logging.getLogger(__name__)


class ToolService:
    """工具服务类"""

    def __init__(self):
        self.mcp_manager = None

    async def _ensure_mcp_manager(self):
        """确保 MCP 管理器已初始化"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = await get_mcp_manager()
            except Exception as e:
                logger.warning(f"MCP 管理器初始化失败: {e}")

    def get_mcp_status(self):
        """
        获取 MCP 服务状态
        
        Returns:
            dict: MCP 状态信息
        """
        if self.mcp_manager is None:
            return {
                "available": False,
                "message": "MCP 管理器未初始化"
            }

        try:
            return {
                "available": True,
                "message": "MCP 管理器运行中",
                "server_count": len(self.mcp_manager._servers) if hasattr(self.mcp_manager, '_servers') else 0
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"MCP 状态检查失败: {str(e)}"
            }

    def list_mcp_servers(self):
        """
        获取已注册的 MCP 服务器列表
        
        Returns:
            dict: 服务器列表
        """
        if self.mcp_manager is None:
            return {"servers": [], "total": 0}

        try:
            servers = self.mcp_manager.list_servers() if hasattr(self.mcp_manager, 'list_servers') else []
            return {"servers": servers, "total": len(servers)}
        except Exception as e:
            logger.error(f"获取MCP服务器列表失败: {e}")
            raise

    async def execute_mcp_tool(self, server_name: str, tool_name: str, arguments: dict):
        """
        执行 MCP 工具
        
        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            Any: 工具执行结果
        """
        await self._ensure_mcp_manager()
        
        if self.mcp_manager is None:
            raise Exception("MCP 管理器未初始化")

        try:
            result = await self.mcp_manager.execute_tool(server_name, tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"MCP工具执行失败: {e}")
            raise

    async def get_all_tools(self):
        """
        获取所有可用工具列表
        
        Returns:
            list: 工具列表
        """
        try:
            tools = await get_all_tools()
            return {"tools": [{"name": t.name, "description": t.description} for t in tools], "total": len(tools)}
        except Exception as e:
            logger.error(f"获取工具列表失败: {e}")
            raise

    async def get_tools_by_type(self, tool_type: str):
        """
        获取指定类型的工具列表
        
        Args:
            tool_type: 工具类型（mcp, api, local）
            
        Returns:
            list: 工具列表
        """
        if tool_type not in ["mcp", "api", "local"]:
            raise ValueError(f"未知的工具类型: {tool_type}")

        try:
            tools = tool_api.get_tools_by_category(tool_type)
            return [{"name": t.name, "description": t.description} for t in tools]
        except Exception as e:
            logger.error(f"获取{tool_type}工具列表失败: {e}")
            raise

    async def execute_tool(self, tool_name: str, **kwargs):
        """
        执行工具（统一接口）
        
        Args:
            tool_name: 工具名称（格式：type.name）
            **kwargs: 工具参数
            
        Returns:
            Any: 工具执行结果
        """
        return await call_tool(tool_name, **kwargs)

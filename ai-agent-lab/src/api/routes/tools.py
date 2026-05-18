"""
工具路由模块

【功能】：
1. MCP 服务状态管理
2. 工具列表查询
3. 工具执行

【接口列表】：
- GET /tools/mcp/status - 获取 MCP 服务状态
- GET /tools/mcp/servers - 获取已注册的 MCP 服务器列表
- POST /tools/mcp/execute - 执行 MCP 工具
- GET /tools/list - 获取所有可用工具列表
- GET /tools/{tool_type} - 获取指定类型的工具列表
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.services.tool_service import ToolService

router = APIRouter(prefix="/tools")


class ToolExecuteRequest(BaseModel):
    """工具执行请求体"""
    tool_name: str = Field(..., description="工具名称")
    arguments: Optional[dict] = Field(default={}, description="工具参数")


@router.get("/mcp/status")
async def get_mcp_status():
    """获取 MCP 服务状态"""
    tool_service = ToolService()
    return tool_service.get_mcp_status()


@router.get("/mcp/servers")
async def list_mcp_servers():
    """获取已注册的 MCP 服务器列表"""
    tool_service = ToolService()
    return tool_service.list_mcp_servers()


@router.post("/mcp/execute")
async def execute_mcp_tool(server_name: str, tool_name: str, arguments: dict = None):
    """执行 MCP 工具"""
    tool_service = ToolService()
    try:
        result = await tool_service.execute_mcp_tool(server_name, tool_name, arguments or {})
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_all_tools():
    """获取所有可用工具列表"""
    tool_service = ToolService()
    return await tool_service.get_all_tools()


@router.get("/{tool_type}")
async def list_tools_by_type(tool_type: str):
    """获取指定类型的工具列表"""
    tool_service = ToolService()
    try:
        tools = await tool_service.get_tools_by_type(tool_type)
        return {"tools": tools, "type": tool_type}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

"""
=== MCP 客户端（基于官方 langchain-mcp-adapters）===
使用官方转换函数｜支持多服务器｜自动参数验证｜异步支持
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Type

# 使用官方 langchain-mcp-adapters 库进行工具转换
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

# 官方 MCP SDK
import mcp
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

logger = logging.getLogger(__name__)

# 全局缓存
_mcp_tools: Dict[str, BaseTool] = {}
_last_load_time: float = 0
CACHE_TTL: int = 300


class MCPServerConfig:
    """MCP 服务器配置"""
    def __init__(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None, disabled: bool = False):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.disabled = disabled


async def _load_config(config_path: Optional[str] = None) -> Dict[str, MCPServerConfig]:
    """加载 MCP 服务器配置"""
    config_path = config_path or os.path.join(os.path.dirname(__file__), "mcp_config.json")
    servers: Dict[str, MCPServerConfig] = {}
    
    try:
        if not os.path.exists(config_path):
            logger.warning(f"MCP 配置不存在: {config_path}")
            return servers
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for srv_name, cfg in config.get("mcpServers", {}).items():
            servers[srv_name] = MCPServerConfig(
                name=srv_name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                disabled=cfg.get("disabled", False)
            )
    except Exception as e:
        logger.error(f"加载 MCP 配置失败: {e}")
    
    return servers


async def _create_tools_for_server(cfg: MCPServerConfig) -> List[BaseTool]:
    """为单个服务器创建工具（使用官方适配器）"""
    tools: List[BaseTool] = []
    
    try:
        env = {**os.environ, **cfg.env}
        params = mcp.StdioServerParameters(command=cfg.command, args=cfg.args, env=env)

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                mcp_tools = await sess.list_tools()
                
                for mcp_tool in mcp_tools.tools:
                    # 使用官方适配器转换为 LangChain 工具
                    langchain_tool = convert_mcp_tool_to_langchain_tool(mcp_tool)
                    
                    # 添加服务器前缀
                    langchain_tool.name = f"{cfg.name}.{langchain_tool.name}"
                    tools.append(langchain_tool)
        
        logger.info(f"服务器 {cfg.name} 加载完成，工具数：{len(tools)}")
    except Exception as e:
        logger.error(f"连接 MCP {cfg.name} 失败: {e}")
    
    return tools


async def _load_mcp_tools() -> Dict[str, BaseTool]:
    """加载所有 MCP 工具（带缓存）"""
    global _mcp_tools, _last_load_time
    
    current_time = time.time()
    if _mcp_tools and (current_time - _last_load_time) < CACHE_TTL:
        return _mcp_tools
    
    _mcp_tools = {}
    servers = await _load_config()
    
    for name, cfg in servers.items():
        if not cfg.disabled:
            tools = await _create_tools_for_server(cfg)
            for tool in tools:
                _mcp_tools[tool.name] = tool
    
    _last_load_time = current_time
    logger.info(f"MCP 工具加载完成，总工具数：{len(_mcp_tools)}")
    
    return _mcp_tools


async def get_langchain_tools_from_mcp() -> List[BaseTool]:
    """获取所有 MCP 工具并转换为 LangChain 工具（使用官方适配器）"""
    tools = await _load_mcp_tools()
    return list(tools.values())


async def call_mcp_tool(tool_name: str, **kwargs) -> str:
    """调用 MCP 工具"""
    tools = await _load_mcp_tools()
    
    if tool_name not in tools:
        return f"工具未找到: {tool_name}"
    
    try:
        tool = tools[tool_name]
        result = await tool.arun(**kwargs)
        return str(result) if result else "无返回内容"
    except Exception as e:
        logger.error(f"MCP 工具调用失败 {tool_name}: {e}")
        return f"调用失败: {str(e)[:200]}"


async def get_available_mcp_tools() -> List[Dict[str, str]]:
    """获取所有可用的 MCP 工具列表"""
    tools = await _load_mcp_tools()
    
    return [
        {
            "name": name,
            "description": tool.description or "无描述"
        }
        for name, tool in tools.items()
    ]


async def is_mcp_available() -> bool:
    """检查 MCP 是否可用"""
    try:
        tools = await _load_mcp_tools()
        return len(tools) > 0
    except Exception:
        return False


def refresh_tools():
    """刷新 MCP 工具缓存"""
    global _mcp_tools, _last_load_time
    _mcp_tools = {}
    _last_load_time = 0
    logger.info("MCP 工具缓存已刷新")


def get_metrics() -> Dict[str, Any]:
    """获取 MCP 系统指标"""
    return {
        "tool_count": len(_mcp_tools),
        "last_load_time": _last_load_time,
        "cache_ttl": CACHE_TTL,
        "is_available": len(_mcp_tools) > 0
    }


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """获取工具详细信息"""
    if tool_name in _mcp_tools:
        tool = _mcp_tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "args_schema": str(tool.args_schema) if hasattr(tool, 'args_schema') else None
        }
    return None


# 兼容旧接口
async def get_mcp_client_manager():
    """获取 MCP 客户端管理器（兼容旧接口）"""
    await _load_mcp_tools()
    return _mcp_tools


async def close_mcp_client_manager():
    """关闭 MCP 客户端管理器（兼容旧接口）"""
    refresh_tools()

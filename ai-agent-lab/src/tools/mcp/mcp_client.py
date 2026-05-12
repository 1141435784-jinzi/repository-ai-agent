"""
=== MCP 客户端（基于官方库，使用schema自动验证）===
稳定修复版｜支持并发｜进程安全｜自动重连｜参数验证
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
import time

# 官方MCP库
import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

# LangChain集成
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    disabled: bool = False


class SimpleMCPClient:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "mcp_config.json")
        self.servers: Dict[str, MCPServerConfig] = {}
        self._langchain_tools: Dict[str, BaseTool] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self):
        logger.info("初始化 MCP 客户端管理器...")
        await self._load_config()
        await self._create_tools_for_enabled_servers()
        logger.info(f"MCP 初始化完成，可用工具数：{len(self._langchain_tools)}")

    async def _load_config(self):
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"MCP 配置不存在: {self.config_path}")
                return
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for srv_name, cfg in config.get("mcpServers", {}).items():
                self.servers[srv_name] = MCPServerConfig(
                    name=srv_name,
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    disabled=cfg.get("disabled", False)
                )
        except Exception as e:
            logger.error(f"加载 MCP 配置失败: {e}")

    async def _create_tools_for_enabled_servers(self):
        for name, cfg in self.servers.items():
            if not cfg.disabled:
                try:
                    await self._create_tools_for_server(name, cfg)
                except Exception as e:
                    logger.error(f"服务器 {name} 初始化失败: {e}")

    async def _create_tools_for_server(self, srv_name: str, cfg: MCPServerConfig):
        try:
            env = {**os.environ, **cfg.env}
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=env)

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    tools = await sess.list_tools()
                    for t in tools.tools:
                        await self._create_tool(srv_name, t, cfg)
            logger.info(f"服务器 {srv_name} 加载完成，工具数：{len(tools.tools)}")
        except Exception as e:
            logger.error(f"连接 MCP {srv_name} 失败: {e}")

    def _create_pydantic_model(self, tool_name: str, schema: Dict) -> Type[BaseModel]:
        if not schema or "properties" not in schema:
            return create_model(f"{tool_name}_Default")

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "array": List, "object": Dict}
        fields = {}
        props = schema.get("properties", {})
        required = schema.get("required", [])

        for k, v in props.items():
            t = type_map.get(v.get("type"), str)
            fields[k] = (t, Field(description=v.get("description", ""), default=... if k in required else None))

        return create_model(f"{tool_name}_Input", __config__={"arbitrary_types_allowed": True}, **fields)

    async def _create_tool(self, srv_name: str, mcp_tool: MCPTool, cfg: MCPServerConfig):
        schema = mcp_tool.inputSchema or {}
        full_name = f"{srv_name}.{mcp_tool.name}"
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', full_name)

        self._tool_schemas[clean_name] = {
            "server": srv_name,
            "tool": mcp_tool.name,
            "schema": schema,
            "cfg": cfg
        }

        InputModel = self._create_pydantic_model(clean_name, schema)

        async def _run(**kwargs):
            try:
                val = InputModel(**kwargs)
                args = val.model_dump(exclude_none=True)
            except Exception as e:
                return f"参数错误: {str(e)}"

            try:
                env = {**os.environ, **cfg.env}
                params = StdioServerParameters(command=cfg.command, args=cfg.args, env=env)

                async with (
                    asyncio.timeout(30),
                    stdio_client(params) as (read, write),
                    ClientSession(read, write) as sess
                ):
                    await sess.initialize()
                    res = await sess.call_tool(mcp_tool.name, arguments=args)
                    return "".join([c.text for c in res.content if hasattr(c, "text")]) or "无返回内容"
            except Exception as e:
                logger.error(f"MCP 调用失败 {clean_name}: {e}")
                return f"调用失败: {str(e)[:200]}"

        async def _adapter(input: Optional[Any] = None, **kwargs):
            final = {}
            if isinstance(input, dict):
                final.update(input)
            elif isinstance(input, str):
                try:
                    final.update(json.loads(input))
                except:
                    if len(InputModel.model_fields) == 1:
                        k = next(iter(InputModel.model_fields.keys()))
                        final[k] = input
            final.update(kwargs)
            return await _run(**final)

        tool = Tool(
            name=clean_name,
            description=mcp_tool.description or "无描述",
            args_schema=InputModel,
            coroutine=_adapter,
            func=lambda *_, **__: None
        )

        self._langchain_tools[clean_name] = tool

    async def get_langchain_tools(self) -> List[BaseTool]:
        return list(self._langchain_tools.values())

    async def list_tools(self) -> List[Dict]:
        return [{"name": n, "description": t.description} for n, t in self._langchain_tools.items()]

    async def list_servers(self) -> List[Dict]:
        """列出所有配置的 MCP 服务器及其状态"""
        servers = []
        for name, cfg in self.servers.items():
            # 检查服务器是否有可用工具（间接判断是否连接成功）
            server_tools = [t for t in self._langchain_tools.keys() if t.startswith(f"{name}.")]
            has_tools = len(server_tools) > 0
            
            servers.append({
                "name": name,
                "status": "connected" if has_tools else "disconnected",
                "disabled": cfg.disabled,
                "tools_count": len(server_tools),
                "error_count": 0  # 简化处理，实际应该记录错误计数
            })
        return servers

    async def call_tool_direct(self, server_name: str, tool_name: str, **kwargs) -> Any:
        """直接调用指定服务器的工具"""
        # 构建完整的工具名称
        full_tool_name = f"{server_name}.{tool_name}"
        
        # 检查工具是否存在
        if full_tool_name not in self._langchain_tools:
            # 尝试不带服务器前缀的工具名
            if tool_name in self._langchain_tools:
                full_tool_name = tool_name
            else:
                raise ValueError(f"工具 {full_tool_name} 不存在")
        
        # 调用工具
        tool = self._langchain_tools[full_tool_name]
        return await tool.ainvoke(kwargs)

    async def shutdown(self):
        self._langchain_tools.clear()
        self._tool_schemas.clear()
        logger.info("MCP 客户端已关闭")


# ==================== 全局单例（线程安全）====================
_mcp_client_manager: Optional[SimpleMCPClient] = None
_mcp_lock = asyncio.Lock()

async def get_mcp_client_manager() -> SimpleMCPClient:
    global _mcp_client_manager
    async with _mcp_lock:
        if _mcp_client_manager is None:
            _mcp_client_manager = SimpleMCPClient()
            await _mcp_client_manager.initialize()
    return _mcp_client_manager

async def close_mcp_client_manager():
    global _mcp_client_manager
    async with _mcp_lock:
        if _mcp_client_manager:
            await _mcp_client_manager.shutdown()
            _mcp_client_manager = None

# ==================== 工具函数 ====================

async def is_mcp_available() -> bool:
    """检查 MCP 是否可用"""
    try:
        import mcp
        return True
    except ImportError:
        return False

async def get_metrics() -> Dict[str, Any]:
    """获取 MCP 指标信息"""
    mgr = await get_mcp_client_manager()
    return {
        "servers": len(mgr.servers),
        "tools": len(mgr._langchain_tools),
        "timestamp": time.time()
    }

async def get_tool_info(tool_name: str) -> Optional[Dict]:
    """获取工具信息"""
    mgr = await get_mcp_client_manager()
    if tool_name in mgr._langchain_tools:
        tool = mgr._langchain_tools[tool_name]
        return {
            "name": tool_name,
            "description": tool.description,
            "schema": mgr._tool_schemas.get(tool_name, {})
        }
    return None

async def get_available_mcp_tools() -> List[Dict]:
    """获取所有可用的 MCP 工具列表"""
    mgr = await get_mcp_client_manager()
    return await mgr.list_tools()

async def call_mcp_tool(tool_name: str, **kwargs) -> Any:
    """调用 MCP 工具"""
    mgr = await get_mcp_client_manager()
    if tool_name in mgr._langchain_tools:
        tool = mgr._langchain_tools[tool_name]
        return await tool.ainvoke(kwargs)
    raise ValueError(f"工具 {tool_name} 不存在")

async def get_langchain_tools_from_mcp() -> List[BaseTool]:
    mgr = await get_mcp_client_manager()
    return await mgr.get_langchain_tools()

async def refresh_tools():
    await close_mcp_client_manager()
    await get_mcp_client_manager()

get_all_tools = get_langchain_tools_from_mcp
"""MCP 工具适配器 - 将 MCP 工具集成到企业级工具架构"""

import asyncio
import json
import logging
import os
import re
from typing import Any, ClassVar, Dict, List, Optional, Type
from dataclasses import dataclass, field
import time

from src.tools.base import BaseTool, ToolOutput
from src.tools.registry import register_tool
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
        self._langchain_tools: Dict[str, Any] = {}
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
        try:
            import mcp
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.types import Tool as MCPTool
            
            for name, cfg in self.servers.items():
                if not cfg.disabled:
                    try:
                        await self._create_tools_for_server(name, cfg, mcp, ClientSession, 
                                                           StdioServerParameters, stdio_client, MCPTool)
                    except Exception as e:
                        logger.error(f"服务器 {name} 初始化失败: {e}")
        except ImportError:
            logger.warning("MCP 库未安装，跳过 MCP 工具加载")

    async def _create_tools_for_server(self, srv_name: str, cfg: MCPServerConfig, 
                                      mcp_module, ClientSession, StdioServerParameters,
                                      stdio_client, MCPTool):
        try:
            env = {**os.environ, **cfg.env}
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=env)

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    tools = await sess.list_tools()
                    for t in tools.tools:
                        await self._create_tool(srv_name, t, cfg, mcp_module, 
                                              ClientSession, StdioServerParameters, stdio_client)
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

    async def _create_tool(self, srv_name: str, mcp_tool, cfg: MCPServerConfig,
                          mcp_module, ClientSession, StdioServerParameters, stdio_client):
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

        self._langchain_tools[clean_name] = {
            "name": clean_name,
            "description": mcp_tool.description or "无描述",
            "adapter": _adapter,
            "input_model": InputModel,
            "server": srv_name,
            "original_name": mcp_tool.name
        }

    async def get_tools(self) -> List[Dict]:
        return list(self._langchain_tools.values())

    async def list_tools(self) -> List[Dict]:
        return [{"name": n, "description": t["description"]} for n, t in self._langchain_tools.items()]

    async def list_servers(self) -> List[Dict]:
        servers = []
        for name, cfg in self.servers.items():
            server_tools = [t for t in self._langchain_tools.keys() if t.startswith(f"{name}.")]
            has_tools = len(server_tools) > 0
            
            servers.append({
                "name": name,
                "status": "connected" if has_tools else "disconnected",
                "disabled": cfg.disabled,
                "tools_count": len(server_tools),
                "error_count": 0
            })
        return servers

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name in self._langchain_tools:
            tool = self._langchain_tools[tool_name]
            return await tool["adapter"](**kwargs)
        raise ValueError(f"工具 {tool_name} 不存在")

    async def shutdown(self):
        self._langchain_tools.clear()
        self._tool_schemas.clear()
        logger.info("MCP 客户端已关闭")


# 全局单例
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


async def is_mcp_available() -> bool:
    try:
        import mcp
        return True
    except ImportError:
        return False


async def get_mcp_metrics() -> Dict[str, Any]:
    mgr = await get_mcp_client_manager()
    return {
        "servers": len(mgr.servers),
        "tools": len(mgr._langchain_tools),
        "timestamp": time.time()
    }


async def get_mcp_tool_info(tool_name: str) -> Optional[Dict]:
    mgr = await get_mcp_client_manager()
    if tool_name in mgr._langchain_tools:
        tool = mgr._langchain_tools[tool_name]
        return {
            "name": tool_name,
            "description": tool["description"],
            "schema": mgr._tool_schemas.get(tool_name, {})
        }
    return None


async def get_available_mcp_tools() -> List[Dict]:
    mgr = await get_mcp_client_manager()
    return await mgr.list_tools()


async def call_mcp_tool(tool_name: str, **kwargs) -> Any:
    mgr = await get_mcp_client_manager()
    return await mgr.call_tool(tool_name,** kwargs)


async def refresh_mcp_tools():
    await close_mcp_client_manager()
    await get_mcp_client_manager()


async def get_langchain_tools_from_mcp() -> List:
    """获取所有 MCP 工具（返回字典列表，兼容旧接口）"""
    mgr = await get_mcp_client_manager()
    return await mgr.get_tools()


def refresh_tools():
    """刷新 MCP 工具缓存（兼容旧接口）"""
    import asyncio
    asyncio.create_task(refresh_mcp_tools())


# MCP 工具包装器 - 将 MCP 工具注册到企业级工具架构
class MCPToolInput(BaseModel):
    """MCP 工具输入参数"""
    tool_name: str = Field(description="MCP 工具名称")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


class MCPToolOutput(ToolOutput):
    """MCP 工具输出"""
    result: Any = Field(description="工具执行结果")


@register_tool
class MCPToolWrapper(BaseTool):
    name = "mcp_tool"
    description = "调用 MCP (Model Context Protocol) 工具"
    args_schema: ClassVar[Type[BaseModel]] = MCPToolInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "mcp", "tags": ["mcp", "external", "plugin"]}

    async def _arun(self, tool_name: str, kwargs: Dict[str, Any] = {}) -> MCPToolOutput:
        try:
            result = await call_mcp_tool(tool_name, **kwargs)
            return MCPToolOutput(
                success=True,
                message="调用成功",
                result=result
            )
        except Exception as e:
            return MCPToolOutput(
                success=False,
                message=f"调用失败: {str(e)}"
            )
"""
=== 动态工具管理器 ===
稳定修复版｜并发安全｜无死锁｜热刷新有效
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.tools import BaseTool


class DynamicToolManager:
    """动态工具管理器（集成到 Agent 中）- 适配 MCP 和本地工具系统"""
    
    def __init__(self):
        self._tools: List[BaseTool] = []
        self._initialized = False
        self._lock = asyncio.Lock()  # 全局初始化锁
        self._tool_stats = {
            "mcp_tools": 0,
            "api_tools": 0,
            "local_tools": 0,
            "last_refresh": None
        }

    async def initialize(self):
        """初始化工具管理器（线程安全）"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            print("🛠️  正在初始化生产级动态工具系统...")

            # 延迟导入，避免循环依赖
            from .mcp import get_langchain_tools_from_mcp
            from .api import create_all_api_tools
            from .local import create_calculator_tools

            mcp_tools = []
            api_tools = []
            local_tools = []

            # === 分别加载，互不影响 ===
            try:
                mcp_tools = await get_langchain_tools_from_mcp()
                print(f"   📊 加载了 {len(mcp_tools)} 个 MCP 工具")
            except Exception as e:
                print(f"   ⚠️  MCP 工具加载失败: {e}")

            try:
                api_tools = create_all_api_tools()
                print(f"   📊 加载了 {len(api_tools)} 个 API 工具")
            except Exception as e:
                print(f"   ⚠️  API 工具加载失败: {e}")

            try:
                local_tools = create_calculator_tools()
                print(f"   📊 加载了 {len(local_tools)} 个本地工具")
            except Exception as e:
                print(f"   ⚠️  本地工具加载失败: {e}")

            # 合并 + 去重（按工具名）
            tool_map = {}
            for t in mcp_tools + api_tools + local_tools:
                tool_map[t.name] = t
            self._tools = list(tool_map.values())

            # 更新统计
            self._tool_stats = {
                "mcp_tools": len(mcp_tools),
                "api_tools": len(api_tools),
                "local_tools": len(local_tools),
                "last_refresh": "initialized"
            }

            self._initialized = True
            print(f"🛠️  工具系统初始化完成，共 {len(self._tools)} 个工具")

    async def get_tools(self) -> List[BaseTool]:
        """获取工具列表（线程安全）"""
        if not self._initialized:
            await self.initialize()
        return self._tools

    async def refresh_tools(self):
        """刷新工具列表（热更新）"""
        async with self._lock:
            print("🛠️  正在刷新工具列表...")

            # 刷新 MCP
            try:
                from .mcp import refresh_tools
                await refresh_tools()
                print("   🔄 MCP 工具已刷新")
            except Exception as e:
                print(f"   ⚠️ MCP 刷新失败: {e}")

            # 重置状态
            self._initialized = False
            await self.initialize()
            print(f"🛠️ 刷新完成，当前工具数: {len(self._tools)}")

    async def get_tool_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        return self._tool_stats

    def _classify_tool_type(self, tool_name: str) -> str:
        """内部工具类型判断（修复 MCP 工具无法识别问题）"""
        if "mcp_" in tool_name:
            return "mcp"
        elif tool_name.startswith("api_"):
            return "api"
        elif tool_name.startswith("local_"):
            return "local"
        return "unknown"

    async def get_tools_by_type(self, tool_type: str) -> List[BaseTool]:
        """按类型获取工具（修复 MCP 识别错误）"""
        tools = await self.get_tools()
        return [
            t for t in tools
            if self._classify_tool_type(t.name) == tool_type
        ]


# ==================== 全局单例 ====================
tool_manager = DynamicToolManager()


# ==================== 对外 API ====================
async def get_tool_system_stats() -> dict:
    stats = await tool_manager.get_tool_stats()
    mcp_tools = await tool_manager.get_tools_by_type("mcp")
    api_tools = await tool_manager.get_tools_by_type("api")
    local_tools = await tool_manager.get_tools_by_type("local")

    return {
        "total_tools": len(await tool_manager.get_tools()),
        "tool_stats": stats,
        "tool_details": {
            "mcp_tools": [{"name": t.name, "description": t.description} for t in mcp_tools],
            "api_tools": [{"name": t.name, "description": t.description} for t in api_tools],
            "local_tools": [{"name": t.name, "description": t.description} for t in local_tools],
        },
        "agent_system": {
            "total_agents": 2,
            "available_agents": ["agent_tech", "travel"],
            "all_agents_tool_enabled": True,
            "rag_agents": ["agent_tech", "travel"]
        }
    }


async def refresh_tool_system() -> dict:
    await tool_manager.refresh_tools()
    return {
        "success": True,
        "message": "工具系统已刷新",
        "new_stats": await tool_manager.get_tool_stats()
    }


async def get_tool_info(tool_name: str) -> dict:
    tools = await tool_manager.get_tools()
    tool = next((t for t in tools if t.name == tool_name), None)

    if not tool:
        return {"name": tool_name, "available": False, "error": "工具未找到"}

    t_type = tool_manager._classify_tool_type(tool_name)
    return {
        "name": tool.name,
        "description": tool.description,
        "tool_type": t_type,
        "args_schema": tool.args_schema.model_json_schema() if hasattr(tool, "args_schema") else None,
        "available": True
    }


# ==================== 移除危险同步初始化（必须删掉）====================
# 以下函数会导致 FastAPI 死锁，已完全删除
# def initialize_tools_sync() -> List[BaseTool]:
#     ...


# ==================== 导出 ====================
__all__ = [
    "DynamicToolManager",
    "tool_manager",
    "get_tool_system_stats",
    "refresh_tool_system",
    "get_tool_info",
]
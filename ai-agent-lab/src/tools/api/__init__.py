"""
=== 工具开放接口层 ===

为 Agent 提供统一的工具访问接口，简化工具调用流程。

【功能】：
1. 统一的工具获取接口
2. 统一的工具调用接口
3. LangChain 工具转换
4. 批量工具执行
5. 工具信息查询

【设计原则】：
1. 简化接口：提供最简单的调用方式
2. 兼容性：支持多种 Agent 框架
3. 类型安全：使用 Pydantic 模型
4. 易用性：提供直观的 API
"""

from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolMetadata
from src.tools.registry import tool_registry
from src.tools.executor import tool_executor, ExecutionResult


class ToolInfo(BaseModel):
    """工具信息模型"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    category: str = Field(description="工具类别")
    tags: List[str] = Field(default_factory=list, description="工具标签")
    rate_limit: int = Field(default=10, description="限流限制")
    timeout: int = Field(default=30, description="超时时间")


class ToolAPI:
    """
    工具开放接口
    
    为 Agent 提供统一的工具访问入口
    """
    
    def get_tools(self) -> List[Type[BaseTool]]:
        """获取所有注册的工具类"""
        return tool_registry.get_all_tools()
    
    def get_tool_instances(self) -> List[BaseTool]:
        """获取所有工具实例"""
        return tool_registry.get_all_tool_instances()
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """获取工具信息"""
        tool_class = tool_registry.get_tool(tool_name)
        if tool_class:
            metadata = tool_class.metadata
            return ToolInfo(
                name=tool_class.name,
                description=tool_class.description,
                category=metadata.category,
                tags=metadata.tags,
                rate_limit=metadata.rate_limit,
                timeout=metadata.timeout
            )
        return None
    
    def get_all_tool_info(self) -> List[ToolInfo]:
        """获取所有工具信息"""
        infos = []
        for tool_class in tool_registry.get_all_tools():
            metadata = tool_class.metadata
            infos.append(ToolInfo(
                name=tool_class.name,
                description=tool_class.description,
                category=metadata.category,
                tags=metadata.tags,
                rate_limit=metadata.rate_limit,
                timeout=metadata.timeout
            ))
        return infos
    
    async def call_tool(self, tool_name: str, **kwargs) -> ExecutionResult:
        """调用工具"""
        return await tool_executor.execute(tool_name, **kwargs)
    
    async def batch_call_tools(self, tool_calls: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """批量调用工具"""
        tasks = []
        for call in tool_calls:
            tool_name = call.get("tool_name")
            params = call.get("params", {})
            tasks.append(tool_executor.execute(tool_name, **params))
        
        return await asyncio.gather(*tasks)
    
    def to_langchain_tools(self) -> List[LangChainBaseTool]:
        """转换为 LangChain 工具格式（包含注册工具和 MCP 工具）"""
        langchain_tools = []
        
        # 1. 添加注册到 tool_registry 的工具
        for tool_class in tool_registry.get_all_tools():
            # 创建工具实例
            tool_instance = tool_class()
            
            # 获取输入输出模型
            input_model = tool_class.input_schema
            output_model = tool_class.output_schema
            
            # 创建包装函数
            async def _wrapped_call(tool_instance=tool_instance, **kwargs):
                result = await tool_instance.execute(**kwargs)
                if hasattr(result, 'model_dump'):
                    return result.model_dump()
                return result
            
            # 创建 StructuredTool
            langchain_tool = StructuredTool.from_function(
                name=tool_class.name,
                func=_wrapped_call,
                description=tool_class.description,
                args_schema=input_model,
                coroutine=_wrapped_call
            )
            
            langchain_tools.append(langchain_tool)
        
        # 2. 添加 MCP 工具（动态发现）
        try:
            from src.tools.mcp import get_langchain_tools_from_mcp
            import asyncio
            mcp_tools = asyncio.run(get_langchain_tools_from_mcp())
            langchain_tools.extend(mcp_tools)
        except Exception as e:
            # MCP 未配置或未安装时跳过
            pass
        
        return langchain_tools
    
    def get_tools_by_category(self, category: str) -> List[Type[BaseTool]]:
        """按类别获取工具"""
        return tool_registry.get_tools_by_category(category)
    
    def search_tools(self, keyword: str) -> List[Type[BaseTool]]:
        """搜索工具"""
        return tool_registry.search_tools(keyword)


# 全局工具 API 实例
tool_api = ToolAPI()


# ==================== 便捷函数 ====================

def get_all_tools() -> List[Type[BaseTool]]:
    """获取所有工具类（便捷函数）"""
    return tool_api.get_tools()


def get_tool_info(tool_name: str) -> Optional[ToolInfo]:
    """获取工具信息（便捷函数）"""
    return tool_api.get_tool_info(tool_name)


def to_langchain_tools() -> List[LangChainBaseTool]:
    """转换为 LangChain 工具（便捷函数）"""
    return tool_api.to_langchain_tools()


# 导入 asyncio 用于异步调用
import asyncio

# ==================== 导出列表 ====================

__all__ = [
    "ToolAPI",
    "tool_api",
    "ToolInfo",
    "get_all_tools",
    "get_tool_info",
    "to_langchain_tools",
]

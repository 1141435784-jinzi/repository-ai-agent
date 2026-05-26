"""
=== 工具开放接口层 ===

为 Agent 提供统一的工具访问接口，简化工具调用流程。

【功能】：
1. 统一的工具获取接口（区分工具和技能）
2. 统一的工具调用接口
3. LangChain 工具转换
4. 批量工具执行
5. 工具信息查询

【设计原则】：
1. 简化接口：提供最简单的调用方式
2. 兼容性：支持多种 Agent 框架
3. 类型安全：使用 Pydantic 模型
4. 易用性：提供直观的 API
5. 清晰区分：工具（静态注册）和技能（动态加载）分开管理
"""

from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool as LangChainBaseTool, BaseTool
from pydantic import BaseModel, Field

from src.tools.base import ToolMetadata
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


# ==================== MCP 工具缓存 ====================

_mcp_tools_cache: Optional[List[LangChainBaseTool]] = None
_mcp_tools_cache_time: float = 0
MCP_CACHE_TTL: int = 300


async def _get_mcp_tools_cached() -> List[LangChainBaseTool]:
    """获取 MCP 工具（带缓存，避免重复初始化）"""
    global _mcp_tools_cache, _mcp_tools_cache_time
    
    import time
    current_time = time.time()
    
    if _mcp_tools_cache is not None and (current_time - _mcp_tools_cache_time) < MCP_CACHE_TTL:
        return _mcp_tools_cache
    
    try:
        from src.tools.mcp import get_langchain_tools_from_mcp
        _mcp_tools_cache = await get_langchain_tools_from_mcp()
        _mcp_tools_cache_time = current_time
    except ImportError:
        _mcp_tools_cache = []
    except Exception as e:
        print(f"⚠️ 获取 MCP 工具失败: {e}")
        _mcp_tools_cache = []
    
    return _mcp_tools_cache


def refresh_mcp_cache():
    """刷新 MCP 工具缓存"""
    global _mcp_tools_cache, _mcp_tools_cache_time
    _mcp_tools_cache = None
    _mcp_tools_cache_time = 0


class SkillInfo(BaseModel):
    """技能信息模型"""
    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述")
    version: str = Field(default="1.0.0", description="技能版本")
    author: Optional[str] = Field(default=None, description="作者")
    tags: List[str] = Field(default_factory=list, description="技能标签")


class ToolAPI:
    """
    工具开放接口
    
    为 Agent 提供统一的工具访问入口
    """
    
    def get_tools(self) -> List[Type[BaseTool]]:
        """获取所有注册的工具类（仅静态工具）"""
        return tool_registry.get_all_tools()
    
    def get_skills(self) -> List[Dict[str, Any]]:
        """获取所有技能（仅动态技能）"""
        try:
            from src.tools.skills import SkillManager
            
            skill_manager = SkillManager()
            skill_manager.initialize()
            return skill_manager.list_skills_dict()
        except ImportError:
            return []
        except Exception as e:
            print(f"⚠️ 获取技能失败: {e}")
            return []
    
    def get_tool_info_list(self) -> List[ToolInfo]:
        """获取所有工具信息列表（仅静态工具）"""
        infos = []
        for tool_class in tool_registry.get_all_tools():
            metadata = getattr(tool_class, 'metadata', None)
            infos.append(ToolInfo(
                name=tool_class.name,
                description=tool_class.description,
                category=getattr(metadata, 'category', 'general') if metadata else 'general',
                tags=getattr(metadata, 'tags', []) if metadata else [],
                rate_limit=getattr(metadata, 'rate_limit', 10) if metadata else 10,
                timeout=getattr(metadata, 'timeout', 30) if metadata else 30
            ))
        return infos
    
    def get_skill_info_list(self) -> List[SkillInfo]:
        """获取所有技能信息列表（仅动态技能）"""
        try:
            from src.tools.skills import SkillManager
            
            skill_manager = SkillManager()
            skill_manager.initialize()
            skills = skill_manager.list_skills()
            
            return [
                SkillInfo(
                    name=skill.name,
                    description=skill.description,
                    version=skill.version,
                    author=skill.author,
                    tags=skill.tags
                )
                for skill in skills
            ]
        except ImportError:
            return []
        except Exception as e:
            print(f"⚠️ 获取技能信息失败: {e}")
            return []
    
    def get_tool_instances(self) -> List[BaseTool]:
        """获取所有工具实例"""
        return tool_registry.get_all_tool_instances()
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """获取工具信息"""
        tool_class = tool_registry.get_tool(tool_name)
        if tool_class:
            metadata = getattr(tool_class, 'metadata', None)
            return ToolInfo(
                name=tool_class.name,
                description=tool_class.description,
                category=getattr(metadata, 'category', 'general') if metadata else 'general',
                tags=getattr(metadata, 'tags', []) if metadata else [],
                rate_limit=getattr(metadata, 'rate_limit', 10) if metadata else 10,
                timeout=getattr(metadata, 'timeout', 30) if metadata else 30
            )
        return None
    
    def get_all_tool_info(self) -> List[ToolInfo]:
        """获取所有工具信息"""
        infos = []
        for tool_class in tool_registry.get_all_tools():
            metadata = getattr(tool_class, 'metadata', None)
            infos.append(ToolInfo(
                name=tool_class.name,
                description=tool_class.description,
                category=getattr(metadata, 'category', 'general') if metadata else 'general',
                tags=getattr(metadata, 'tags', []) if metadata else [],
                rate_limit=getattr(metadata, 'rate_limit', 10) if metadata else 10,
                timeout=getattr(metadata, 'timeout', 30) if metadata else 30
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
    
    async def to_langchain_tools(self) -> List[LangChainBaseTool]:
        """转换为 LangChain 工具格式（异步版本，推荐使用）"""
        langchain_tools = []
        
        for tool_class in tool_registry.get_all_tools():
            tool_instance = tool_class()
            langchain_tools.append(tool_instance)
        
        try:
            mcp_tools = await _get_mcp_tools_cached()
            if mcp_tools:
                langchain_tools.extend(mcp_tools)
                print(f"🔧 已加载 {len(mcp_tools)} 个 MCP 工具")
        except Exception as e:
            print(f"⚠️ MCP 工具加载失败: {e}")
        
        try:
            from src.tools.skills import SkillManager

            skill_manager = SkillManager()
            skill_manager.initialize()
            skill_tools = skill_manager.to_langchain_tools()
            
            valid_skill_tools = [
                tool for tool in skill_tools
                if hasattr(tool, 'name') and hasattr(tool, 'description')
            ]
            
            if valid_skill_tools:
                langchain_tools.extend(valid_skill_tools)
                print(f"🎯 已加载 {len(valid_skill_tools)} 个技能")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ 技能加载失败: {e}")
        
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
    """获取所有工具类（便捷函数，仅静态工具）"""
    return tool_api.get_tools()


def get_tool_info(tool_name: str) -> Optional[ToolInfo]:
    """获取工具信息（便捷函数）"""
    return tool_api.get_tool_info(tool_name)


def get_all_skills() -> List[Dict[str, Any]]:
    """获取所有技能（便捷函数，仅动态技能）"""
    return tool_api.get_skills()


def get_tool_info_list() -> List[ToolInfo]:
    """获取所有工具信息列表（便捷函数，仅静态工具）"""
    return tool_api.get_tool_info_list()


def get_skill_info_list() -> List[SkillInfo]:
    """获取所有技能信息列表（便捷函数，仅动态技能）"""
    return tool_api.get_skill_info_list()


async def to_langchain_tools() -> List[LangChainBaseTool]:
    """转换为 LangChain 工具（便捷函数，异步版本，推荐使用）"""
    return await tool_api.to_langchain_tools()


def refresh_mcp_tools_cache():
    """刷新 MCP 工具缓存（便捷函数）"""
    refresh_mcp_cache()


# 导入 asyncio 用于异步调用
import asyncio

# ==================== 导出列表 ====================

__all__ = [
    "ToolAPI",
    "tool_api",
    "ToolInfo",
    "SkillInfo",
    "get_all_tools",
    "get_all_skills",
    "get_tool_info",
    "get_tool_info_list",
    "get_skill_info_list",
    "to_langchain_tools",
    "refresh_mcp_tools_cache",
]

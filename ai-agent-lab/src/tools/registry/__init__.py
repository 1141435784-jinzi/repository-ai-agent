"""
=== 第二层：工具注册中心层 ===

企业级工具注册中心，负责工具的自动扫描、动态加载和统一管理。

【功能】：
1. 自动扫描指定目录下的工具类
2. 动态加载工具模块
3. 工具注册和注销
4. 工具查询和分类管理
5. 支持按名称、类别、标签搜索工具

【设计原则】：
1. 单例模式：全局唯一的注册中心
2. 线程安全：支持并发访问
3. 热加载：支持运行时动态添加工具
4. 元数据驱动：基于工具元数据进行管理
"""

import asyncio
import importlib
import inspect
import os
from typing import Dict, List, Type, Optional, Any, Callable
from abc import ABC
import logging

from langchain_core.tools import BaseTool
from src.tools.base import ToolMetadata

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册中心
    
    负责管理所有工具的注册、发现和查询。
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tools: Dict[str, Type[BaseTool]] = {}  # 工具名称 -> 工具类
        self._tools_by_category: Dict[str, List[str]] = {}  # 类别 -> 工具名称列表
        self._tools_by_tag: Dict[str, List[str]] = {}  # 标签 -> 工具名称列表
        self._initialized = True
        logger.info("工具注册中心已初始化")
    
    def register_tool(self, tool_class: Type[BaseTool]) -> bool:
        """
        注册工具类
        
        Args:
            tool_class: 工具类（必须继承自 BaseTool）
        
        Returns:
            bool: 是否注册成功
        """
        if not issubclass(tool_class, BaseTool):
            logger.error(f"工具类 {tool_class.__name__} 必须继承自 BaseTool")
            return False
        
        tool_name = tool_class.get_name()
        
        if tool_name in self._tools:
            logger.warning(f"工具 {tool_name} 已存在，将被覆盖")
        
        self._tools[tool_name] = tool_class
        
        # 更新类别索引
        category = tool_class.get_metadata().category
        if category not in self._tools_by_category:
            self._tools_by_category[category] = []
        if tool_name not in self._tools_by_category[category]:
            self._tools_by_category[category].append(tool_name)
        
        # 更新标签索引
        tags = tool_class.get_metadata().tags
        for tag in tags:
            if tag not in self._tools_by_tag:
                self._tools_by_tag[tag] = []
            if tool_name not in self._tools_by_tag[tag]:
                self._tools_by_tag[tag].append(tool_name)
        
        logger.info(f"工具 {tool_name} 注册成功")
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销工具
        
        Args:
            tool_name: 工具名称
        
        Returns:
            bool: 是否注销成功
        """
        if tool_name not in self._tools:
            logger.warning(f"工具 {tool_name} 不存在")
            return False
        
        tool_class = self._tools[tool_name]
        
        # 从类别索引中移除
        category = tool_class.get_metadata().category
        if category in self._tools_by_category:
            self._tools_by_category[category].remove(tool_name)
        
        # 从标签索引中移除
        tags = tool_class.get_metadata().tags
        for tag in tags:
            if tag in self._tools_by_tag:
                self._tools_by_tag[tag].remove(tool_name)
        
        # 从工具列表中移除
        del self._tools[tool_name]
        
        logger.info(f"工具 {tool_name} 注销成功")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[Type[BaseTool]]:
        """
        获取工具类
        
        Args:
            tool_name: 工具名称
        
        Returns:
            Optional[Type[BaseTool]]: 工具类，如果不存在返回 None
        """
        return self._tools.get(tool_name)
    
    def get_tool_instance(self, tool_name: str) -> Optional[BaseTool]:
        """
        获取工具实例
        
        Args:
            tool_name: 工具名称
        
        Returns:
            Optional[BaseTool]: 工具实例，如果不存在返回 None
        """
        tool_class = self.get_tool(tool_name)
        if tool_class:
            return tool_class()
        return None
    
    def get_all_tool_names(self) -> List[str]:
        """
        获取所有工具名称
        
        Returns:
            List[str]: 工具名称列表
        """
        return list(self._tools.keys())
    
    def get_all_tools(self) -> List[Type[BaseTool]]:
        """
        获取所有工具类
        
        Returns:
            List[Type[BaseTool]]: 工具类列表
        """
        return list(self._tools.values())
    
    def get_all_tool_instances(self) -> List[BaseTool]:
        """
        获取所有工具实例
        
        Returns:
            List[BaseTool]: 工具实例列表
        """
        return [tool_class() for tool_class in self._tools.values()]
    
    def get_tools_by_category(self, category: str) -> List[Type[BaseTool]]:
        """
        按类别获取工具类
        
        Args:
            category: 工具类别
        
        Returns:
            List[Type[BaseTool]]: 工具类列表
        """
        tool_names = self._tools_by_category.get(category, [])
        return [self._tools[name] for name in tool_names]
    
    def get_tools_by_tag(self, tag: str) -> List[Type[BaseTool]]:
        """
        按标签获取工具类
        
        Args:
            tag: 标签名称
        
        Returns:
            List[Type[BaseTool]]: 工具类列表
        """
        tool_names = self._tools_by_tag.get(tag, [])
        return [self._tools[name] for name in tool_names]
    
    def get_categories(self) -> List[str]:
        """
        获取所有工具类别
        
        Returns:
            List[str]: 类别列表
        """
        return list(self._tools_by_category.keys())
    
    def get_tags(self) -> List[str]:
        """
        获取所有标签
        
        Returns:
            List[str]: 标签列表
        """
        return list(self._tools_by_tag.keys())
    
    def search_tools(self, keyword: str) -> List[Type[BaseTool]]:
        """
        搜索工具
        
        Args:
            keyword: 搜索关键词（匹配名称、描述、类别、标签）
        
        Returns:
            List[Type[BaseTool]]: 匹配的工具类列表
        """
        keyword = keyword.lower()
        matched = []
        
        for tool_class in self._tools.values():
            name = tool_class.get_name().lower()
            desc = tool_class.get_description().lower()
            category = tool_class.get_metadata().category.lower()
            tags = [tag.lower() for tag in tool_class.get_metadata().tags]
            
            if (keyword in name or 
                keyword in desc or 
                keyword in category or 
                any(keyword in tag for tag in tags)):
                matched.append(tool_class)
        
        return matched
    
    async def scan_and_register(self, module_path: str) -> int:
        """
        扫描指定模块路径下的所有工具类并注册
        
        Args:
            module_path: 模块路径（如 "src.tools.implementations"）
        
        Returns:
            int: 成功注册的工具数量
        """
        count = 0
        
        try:
            module = importlib.import_module(module_path)
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool:
                    if await self.register_tool(obj):
                        count += 1
            
            logger.info(f"扫描模块 {module_path}，注册了 {count} 个工具")
        except Exception as e:
            logger.error(f"扫描模块 {module_path} 失败: {e}")
        
        return count
    
    async def scan_directory(self, directory: str) -> int:
        """
        扫描指定目录下的所有 Python 文件中的工具类并注册
        
        Args:
            directory: 目录路径
        
        Returns:
            int: 成功注册的工具数量
        """
        count = 0
        
        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                module_path = os.path.basename(directory) + "." + module_name
                
                try:
                    module = importlib.import_module(module_path)
                    
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool:
                            if await self.register_tool(obj):
                                count += 1
                except Exception as e:
                    logger.error(f"扫描文件 {filename} 失败: {e}")
        
        logger.info(f"扫描目录 {directory}，注册了 {count} 个工具")
        return count
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具详细信息
        
        Args:
            tool_name: 工具名称
        
        Returns:
            Optional[Dict[str, Any]]: 工具信息字典
        """
        tool_class = self.get_tool(tool_name)
        if not tool_class:
            return None
        
        metadata = tool_class.get_metadata()
        return {
            "name": tool_class.get_name(),
            "description": tool_class.get_description(),
            "metadata": metadata.model_dump(),
            "input_schema": tool_class.get_input_schema().model_json_schema(),
            "output_schema": tool_class.get_output_schema().model_json_schema(),
            "function_calling_schema": tool_class.to_function_calling_schema()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取注册中心统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_tools": len(self._tools),
            "categories": self.get_categories(),
            "tags": self.get_tags(),
            "tools_by_category": {
                cat: len(tools) for cat, tools in self._tools_by_category.items()
            }
        }


# 全局单例实例
tool_registry = ToolRegistry()


# 便捷装饰器
def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    装饰器：自动注册工具类
    
    Usage:
        @register_tool
        class MyTool(BaseTool):
            ...
    """
    # 直接同步注册（无异步操作）
    tool_registry.register_tool(cls)
    return cls


# 导出列表
__all__ = [
    "ToolRegistry",
    "tool_registry",
    "register_tool",
]

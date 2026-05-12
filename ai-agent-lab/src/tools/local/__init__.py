"""
=== 本地工具模块 ===

【功能】：
1. 封装本地 Python 函数为 LangChain 工具
2. 提供常用的本地计算和数据处理工具
3. 支持自定义本地工具扩展

【设计原则】：
1. 简单直接：本地函数直接封装
2. 类型安全：函数签名类型提示
3. 易于测试：纯函数，无外部依赖
4. 模块化：按功能分类组织

【使用方式】：
```python
from src.tools.local import get_local_tools, call_local_tool

# 获取所有本地工具
tools = await get_local_tools()

# 调用特定本地工具
result = await call_local_tool("calculator.add", a=5, b=3)
result = await call_local_tool("text.process", text="Hello World", operation="uppercase")
```

【工具分类】：
1. calculator/: 数学计算工具
2. text/: 文本处理工具
3. datetime/: 日期时间工具
4. file/: 文件操作工具
5. custom/: 自定义工具
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from .calculator import create_calculator_tools

# 本地工具注册表
_local_tools: Optional[List[BaseTool]] = None

async def get_local_tools() -> List[BaseTool]:
    """获取所有本地工具"""
    global _local_tools
    if _local_tools is None:
        _local_tools = []
        
        # 添加计算器工具
        calculator_tools = create_calculator_tools()
        _local_tools.extend(calculator_tools)
        
        # TODO: 添加其他类型的本地工具
    
    return _local_tools

async def call_local_tool(tool_name: str, **kwargs) -> Any:
    """调用本地工具"""
    tools = await get_local_tools()
    
    # 查找工具
    for tool in tools:
        if tool.name == tool_name:
            try:
                return await tool.ainvoke(kwargs)
            except Exception as e:
                # 如果参数传递有问题，尝试其他方式
                if "Too many arguments to single-input tool" in str(e):
                    # 尝试使用 StructuredTool 的方式
                    from langchain_core.tools import StructuredTool
                    if isinstance(tool, StructuredTool):
                        return await tool.ainvoke(kwargs)
                    else:
                        # 对于普通 Tool，需要将 kwargs 包装成字典
                        return await tool.ainvoke({"input": str(kwargs)})
                else:
                    raise e
    
    raise ValueError(f"本地工具未找到: {tool_name}")

# 导出列表
__all__ = [
    "get_local_tools",
    "call_local_tool",
    "create_calculator_tools",
]
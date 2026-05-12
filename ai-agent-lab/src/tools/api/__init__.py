"""
=== API 工具模块 ===

【功能】：
1. 封装 RESTful API 为 LangChain 工具
2. 支持认证、请求重试、错误处理
3. 提供统一的 API 工具接口
4. 提供简单好用的免费 API 工具示例

【设计原则】：
1. 简单易用：参数简单，返回结果清晰
2. 完全免费：无需 API key 或使用免费 tier
3. 类型安全：使用 Pydantic 模型验证参数
4. 错误处理：友好的错误提示

【使用方式】：
```python
from src.tools.api import get_api_tools, call_api_tool

# 获取所有 API 工具
tools = await get_api_tools()

# 调用特定 API 工具
result = await call_api_tool("api_ip_info")
result = await call_api_tool("api_exchange_rate", base_currency="USD", target_currency="CNY", amount=100)
result = await call_api_tool("api_random_quote", limit=3)
```

【包含的免费 API 工具】：
1. IP 信息查询 (ipapi.co) - 无需 API key
2. 汇率查询 (exchangerate-api.com) - 无需 API key
3. 随机名言 (quotable.io) - 无需 API key
4. 公共 API 列表 (public-apis) - 无需 API key
5. 占位符图片 (placeholder.com) - 无需 API key
"""

import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from .free_apis import create_free_api_tools
from .payment_api import (
    create_payment,
    process_payment,
    query_payment,
    refund_payment,
    get_payment_methods,
    CreatePaymentInput,
    ProcessPaymentInput,
    QueryPaymentInput,
    RefundPaymentInput,
    PaymentMethod,
    PaymentStatus
)

from langchain_core.tools import StructuredTool


def create_payment_tools() -> List[BaseTool]:
    """创建支付 API 工具列表"""
    
    # 创建包装函数
    async def wrapped_create_payment(**kwargs):
        return await create_payment(**kwargs)
    
    async def wrapped_process_payment(**kwargs):
        return await process_payment(**kwargs)
    
    async def wrapped_query_payment(**kwargs):
        return await query_payment(**kwargs)
    
    async def wrapped_refund_payment(**kwargs):
        return await refund_payment(**kwargs)
    
    async def wrapped_get_payment_methods(**kwargs):
        return await get_payment_methods(**kwargs)
    
    tools = [
        # 1. 创建支付订单
        StructuredTool.from_function(
            name="api_create_payment",
            func=wrapped_create_payment,
            description="创建支付订单（支持微信和支付宝）。此为模拟支付，不进行真实支付操作。",
            args_schema=CreatePaymentInput,
            coroutine=wrapped_create_payment
        ),
        
        # 2. 处理支付
        StructuredTool.from_function(
            name="api_process_payment",
            func=wrapped_process_payment,
            description="处理支付订单（模拟支付流程）。模拟支付成功或失败。",
            args_schema=ProcessPaymentInput,
            coroutine=wrapped_process_payment
        ),
        
        # 3. 查询支付状态
        StructuredTool.from_function(
            name="api_query_payment",
            func=wrapped_query_payment,
            description="查询支付订单状态。包括支付状态、金额、支付时间等信息。",
            args_schema=QueryPaymentInput,
            coroutine=wrapped_query_payment
        ),
        
        # 4. 退款
        StructuredTool.from_function(
            name="api_refund_payment",
            func=wrapped_refund_payment,
            description="退款操作（模拟退款流程）。只有支付成功的订单才能退款。",
            args_schema=RefundPaymentInput,
            coroutine=wrapped_refund_payment
        ),
        
        # 5. 获取支付方式
        StructuredTool.from_function(
            name="api_get_payment_methods",
            func=wrapped_get_payment_methods,
            description="获取支持的支付方式列表。包括微信支付和支付宝的详细信息。",
            args_schema=None,  # 这个工具不需要参数
            coroutine=wrapped_get_payment_methods
        ),
    ]
    
    return tools


def create_all_api_tools() -> List[BaseTool]:
    """创建所有 API 工具（包括免费工具和支付工具）"""
    free_tools = create_free_api_tools()
    payment_tools = create_payment_tools()
    
    all_tools = free_tools + payment_tools
    print(f"📦 创建了 {len(free_tools)} 个免费工具 + {len(payment_tools)} 个支付工具 = {len(all_tools)} 个 API 工具")
    
    return all_tools


# API 工具注册表
_api_tools: Optional[List[BaseTool]] = None

async def get_api_tools() -> List[BaseTool]:
    """获取所有 API 工具
    
    Returns:
        List[BaseTool]: API 工具列表
    """
    global _api_tools
    if _api_tools is None:
        _api_tools = create_all_api_tools()
        print(f"🔄 加载了 {len(_api_tools)} 个 API 工具")
    return _api_tools

async def call_api_tool(tool_name: str, **kwargs) -> Any:
    """调用 API 工具
    
    Args:
        tool_name: 工具名称，格式为 "api_工具名"
        **kwargs: 工具参数
        
    Returns:
        Any: 工具执行结果
    """
    tools = await get_api_tools()
    
    # 查找工具
    for tool in tools:
        if tool.name == tool_name:
            try:
                # 异步执行工具
                result = await tool.ainvoke(kwargs)
                return result
            except Exception as e:
                return {
                    "error": True,
                    "message": f"工具执行失败: {str(e)}",
                    "tool_name": tool_name
                }
    
    return {
        "error": True,
        "message": f"API 工具未找到: {tool_name}",
        "available_tools": [t.name for t in tools]
    }

async def get_api_tool_info(tool_name: str) -> Dict[str, Any]:
    """获取 API 工具详细信息
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Dict[str, Any]: 工具信息
    """
    tools = await get_api_tools()
    
    for tool in tools:
        if tool.name == tool_name:
            return {
                "name": tool.name,
                "description": tool.description,
                "args_schema": str(tool.args_schema) if hasattr(tool, 'args_schema') else None,
                "is_free": True,
                "requires_api_key": False
            }
    
    return {
        "error": True,
        "message": f"API 工具未找到: {tool_name}"
    }

async def test_all_api_tools() -> Dict[str, Any]:
    """测试所有 API 工具
    
    Returns:
        Dict[str, Any]: 测试结果
    """
    tools = await get_api_tools()
    results = {}
    
    for tool in tools:
        try:
            # 使用默认参数测试每个工具
            if tool.name == "api_ip_info":
                result = await tool.ainvoke({})
            elif tool.name == "api_exchange_rate":
                result = await tool.ainvoke({"base_currency": "USD", "target_currency": "CNY", "amount": 1})
            elif tool.name == "api_random_quote":
                result = await tool.ainvoke({"limit": 1})
            elif tool.name == "api_public_apis":
                result = await tool.ainvoke({"limit": 2})
            elif tool.name == "api_weather_cn":
                result = await tool.ainvoke({"city": "北京"})
            elif tool.name == "api_placeholder_image":
                result = await tool.ainvoke({"width": 100, "height": 100, "text": "test"})
            else:
                result = {"status": "skipped", "reason": "no test parameters"}
            
            results[tool.name] = {
                "success": "error" not in result or not result["error"],
                "result": result if isinstance(result, dict) else str(result)[:100] + "..."
            }
        except Exception as e:
            results[tool.name] = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "total_tools": len(tools),
        "tested_tools": len(results),
        "successful_tests": sum(1 for r in results.values() if r.get("success", False)),
        "results": results
    }


# 导出列表
__all__ = [
    # 主接口
    "get_api_tools",
    "call_api_tool",
    "get_api_tool_info",
    "test_all_api_tools",
    
    # 工具创建函数
    "create_free_api_tools",
    "create_payment_tools",
    "create_all_api_tools",
    
    # 支付工具函数
    "create_payment",
    "process_payment",
    "query_payment",
    "refund_payment",
    "get_payment_methods",
    
    # 支付类型定义
    "CreatePaymentInput",
    "ProcessPaymentInput",
    "QueryPaymentInput",
    "RefundPaymentInput",
    "PaymentMethod",
    "PaymentStatus",
]
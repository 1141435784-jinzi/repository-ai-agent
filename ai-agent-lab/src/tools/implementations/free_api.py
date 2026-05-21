"""免费 API 工具实现"""

import aiohttp
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.tools.base import AsyncTool, ToolMetadata, ToolOutput
from src.tools.registry import register_tool


class IPInfoInput(BaseModel):
    """IP 信息查询输入参数"""
    ip_address: Optional[str] = Field(
        default=None,
        description="IP 地址，如果不提供则查询本机 IP"
    )


class IPInfoOutput(ToolOutput):
    """IP 信息查询输出"""
    ip: str = Field(description="IP 地址")
    city: str = Field(description="城市")
    region: str = Field(description="地区")
    country: str = Field(description="国家")
    isp: str = Field(description="ISP 服务商")
    latitude: float = Field(description="纬度")
    longitude: float = Field(description="经度")
    timezone: str = Field(description="时区")
    currency: str = Field(description="货币")
    languages: str = Field(description="语言")


@register_tool
class IPInfoTool(AsyncTool[IPInfoInput, IPInfoOutput]):
    name = "ip_info"
    description = "获取 IP 地址的详细信息，包括地理位置、ISP、时区等。无需 API key。"
    input_schema = IPInfoInput
    output_schema = IPInfoOutput
    metadata = ToolMetadata(
        name="ip_info",
        description="IP 地址信息查询工具",
        category="api",
        tags=["ip", "network", "geolocation"],
        rate_limit=10,
        timeout=30
    )

    async def async_execute(self, ip_address: Optional[str] = None) -> IPInfoOutput:
        url = f"https://ipapi.co/{ip_address}/json/" if ip_address else "https://ipapi.co/json/"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return IPInfoOutput(
                        success=True,
                        message="查询成功",
                        ip=result.get("ip", "未知"),
                        city=result.get("city", "未知"),
                        region=result.get("region", "未知"),
                        country=result.get("country_name", "未知"),
                        isp=result.get("org", "未知"),
                        latitude=result.get("latitude", 0.0),
                        longitude=result.get("longitude", 0.0),
                        timezone=result.get("timezone", "未知"),
                        currency=result.get("currency", "未知"),
                        languages=result.get("languages", "未知")
                    )
                else:
                    return IPInfoOutput(
                        success=False,
                        message=f"HTTP {response.status}: {response.reason}"
                    )


class ExchangeRateInput(BaseModel):
    """汇率查询输入参数"""
    base_currency: str = Field(default="USD", description="基础货币代码，如 USD、EUR、CNY")
    target_currency: str = Field(description="目标货币代码，如 USD、EUR、CNY、JPY")
    amount: Optional[float] = Field(default=1.0, description="金额，默认为 1")


class ExchangeRateOutput(ToolOutput):
    """汇率查询输出"""
    base_currency: str = Field(description="基础货币")
    target_currency: str = Field(description="目标货币")
    exchange_rate: float = Field(description="汇率")
    amount: float = Field(description="原始金额")
    converted_amount: float = Field(description="转换后金额")
    last_updated: str = Field(description="最后更新时间")


@register_tool
class ExchangeRateTool(AsyncTool[ExchangeRateInput, ExchangeRateOutput]):
    name = "exchange_rate"
    description = "查询货币汇率并计算兑换金额。支持多种货币。无需 API key。"
    input_schema = ExchangeRateInput
    output_schema = ExchangeRateOutput
    metadata = ToolMetadata(
        name="exchange_rate",
        description="货币汇率查询工具",
        category="api",
        tags=["currency", "finance", "exchange"],
        rate_limit=10,
        timeout=30
    )

    async def async_execute(self, base_currency: str = "USD", target_currency: str = "CNY", 
                           amount: float = 1.0) -> ExchangeRateOutput:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency.upper()}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    rates = result.get("rates", {})
                    rate = rates.get(target_currency.upper())
                    
                    if rate is None:
                        return ExchangeRateOutput(
                            success=False,
                            message=f"未找到货币 {target_currency} 的汇率"
                        )
                    
                    return ExchangeRateOutput(
                        success=True,
                        message="查询成功",
                        base_currency=base_currency.upper(),
                        target_currency=target_currency.upper(),
                        exchange_rate=rate,
                        amount=amount,
                        converted_amount=amount * rate,
                        last_updated=result.get("date", "未知")
                    )
                else:
                    return ExchangeRateOutput(
                        success=False,
                        message=f"HTTP {response.status}: {response.reason}"
                    )


class QuoteInput(BaseModel):
    """名言查询输入参数"""
    author: Optional[str] = Field(default=None, description="作者名称，如果不提供则随机获取")
    tags: Optional[str] = Field(default=None, description="标签，如 motivation、success、life")
    limit: Optional[int] = Field(default=1, description="返回的名言数量，最大 20")


class QuoteOutput(ToolOutput):
    """名言查询输出"""
    quotes: List[Dict[str, Any]] = Field(description="名言列表")


@register_tool
class RandomQuoteTool(AsyncTool[QuoteInput, QuoteOutput]):
    name = "random_quote"
    description = "获取随机名言，可按作者或标签筛选。无需 API key。"
    input_schema = QuoteInput
    output_schema = QuoteOutput
    metadata = ToolMetadata(
        name="random_quote",
        description="随机名言获取工具",
        category="api",
        tags=["quote", "inspiration", "text"],
        rate_limit=20,
        timeout=30
    )

    async def async_execute(self, author: Optional[str] = None, tags: Optional[str] = None, 
                          limit: int = 1) -> QuoteOutput:
        base_url = "https://api.quotable.io"
        
        if author:
            url = f"{base_url}/quotes"
            params = {"author": author, "limit": min(limit, 20)}
        elif tags:
            url = f"{base_url}/quotes"
            params = {"tags": tags, "limit": min(limit, 20)}
        else:
            url = f"{base_url}/quotes/random"
            params = {"limit": min(limit, 20)}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    quotes = result if isinstance(result, list) else result.get("results", [])
                    
                    formatted_quotes = []
                    for quote in quotes[:limit]:
                        formatted_quotes.append({
                            "content": quote.get("content", ""),
                            "author": quote.get("author", "未知"),
                            "tags": quote.get("tags", []),
                            "length": quote.get("length", 0)
                        })
                    
                    return QuoteOutput(
                        success=True,
                        message="查询成功",
                        quotes=formatted_quotes
                    )
                else:
                    return QuoteOutput(
                        success=False,
                        message=f"HTTP {response.status}: {response.reason}"
                    )


class PublicAPIInput(BaseModel):
    """公共 API 查询输入参数"""
    category: Optional[str] = Field(default=None, description="API 类别，如 business、animals、weather")
    search: Optional[str] = Field(default=None, description="搜索关键词")
    limit: Optional[int] = Field(default=10, description="返回的 API 数量，最大 50")


class PublicAPIOutput(ToolOutput):
    """公共 API 查询输出"""
    apis: List[Dict[str, Any]] = Field(description="API 列表")


@register_tool
class PublicAPIsTool(AsyncTool[PublicAPIInput, PublicAPIOutput]):
    name = "public_apis"
    description = "搜索和浏览公共 API 列表，可按类别或关键词筛选。无需 API key。"
    input_schema = PublicAPIInput
    output_schema = PublicAPIOutput
    metadata = ToolMetadata(
        name="public_apis",
        description="公共 API 列表查询工具",
        category="api",
        tags=["api", "directory", "resources"],
        rate_limit=10,
        timeout=30
    )

    async def async_execute(self, category: Optional[str] = None, search: Optional[str] = None, 
                          limit: int = 10) -> PublicAPIOutput:
        url = "https://api.publicapis.org/entries"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    entries = result.get("entries", [])
                    
                    filtered_entries = []
                    for entry in entries:
                        match = True
                        if category and category.lower() not in entry.get("Category", "").lower():
                            match = False
                        if search and search.lower() not in entry.get("API", "").lower() and \
                           search.lower() not in entry.get("Description", "").lower():
                            match = False
                        if match:
                            filtered_entries.append(entry)
                        if len(filtered_entries) >= limit:
                            break
                    
                    formatted_apis = []
                    for api in filtered_entries:
                        formatted_apis.append({
                            "name": api.get("API", "未知"),
                            "description": api.get("Description", "无描述"),
                            "category": api.get("Category", "未知"),
                            "url": api.get("Link", ""),
                            "auth": api.get("Auth", "无"),
                            "https": api.get("HTTPS", False),
                            "cors": api.get("Cors", "未知")
                        })
                    
                    return PublicAPIOutput(
                        success=True,
                        message="查询成功",
                        apis=formatted_apis
                    )
                else:
                    return PublicAPIOutput(
                        success=False,
                        message=f"HTTP {response.status}: {response.reason}"
                    )


class PlaceholderImageInput(BaseModel):
    """占位符图片输入参数"""
    width: Optional[int] = Field(default=300, description="图片宽度，最大 2000")
    height: Optional[int] = Field(default=200, description="图片高度，最大 2000")
    text: Optional[str] = Field(default=None, description="图片上的文字")
    background_color: Optional[str] = Field(default="cccccc", description="背景颜色（十六进制，不带#）")
    text_color: Optional[str] = Field(default="000000", description="文字颜色（十六进制，不带#）")


class PlaceholderImageOutput(ToolOutput):
    """占位符图片输出"""
    image_url: str = Field(description="图片 URL")
    width: int = Field(description="图片宽度")
    height: int = Field(description="图片高度")
    text: Optional[str] = Field(description="图片文字")


@register_tool
class PlaceholderImageTool(AsyncTool[PlaceholderImageInput, PlaceholderImageOutput]):
    name = "placeholder_image"
    description = "生成占位符图片 URL，可自定义尺寸、颜色和文字。无需 API key。"
    input_schema = PlaceholderImageInput
    output_schema = PlaceholderImageOutput
    metadata = ToolMetadata(
        name="placeholder_image",
        description="占位符图片生成工具",
        category="api",
        tags=["image", "placeholder", "design"],
        rate_limit=50,
        timeout=10
    )

    async def async_execute(self, width: int = 300, height: int = 200, text: Optional[str] = None,
                          background_color: str = "cccccc", text_color: str = "000000") -> PlaceholderImageOutput:
        width = min(width, 2000)
        height = min(height, 2000)
        
        base_url = f"https://via.placeholder.com/{width}x{height}"
        params = []
        
        if background_color:
            params.append(f"bg={background_color}")
        if text_color:
            params.append(f"text={text_color}")
        if text:
            import urllib.parse
            encoded_text = urllib.parse.quote(text)
            params.append(f"text={encoded_text}")
        
        url = f"{base_url}/{'/'.join(params)}" if params else base_url
        
        return PlaceholderImageOutput(
            success=True,
            message="生成成功",
            image_url=url,
            width=width,
            height=height,
            text=text
        )

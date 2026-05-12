"""
=== 免费 API 工具示例 ===

【功能】：
提供简单好用的免费 API 工具，无需 API key 或使用免费 tier

【包含的 API】：
1. IP 信息查询 (ipapi.co) - 无需 API key
2. 汇率查询 (exchangerate-api.com) - 无需 API key
3. 随机名言 (quotable.io) - 无需 API key
4. 公共 API 列表 (public-apis) - 无需 API key
5. 占位符图片 (placeholder.com) - 无需 API key
6. 国内天气查询 (Open-Meteo) - 无需 API key，支持中文城市名

【设计原则】：
1. 完全免费：无需 API key 或使用免费 tier
2. 简单易用：参数简单，返回结果清晰
3. 错误处理：友好的错误提示
4. 类型安全：使用 Pydantic 模型验证参数
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel, Field
import json
from datetime import datetime


# ============================================================
# 基础工具类
# ============================================================

class FreeAPITool:
    """免费 API 工具基类"""
    
    @staticmethod
    async def make_request(url: str, method: str = "GET", params: Optional[Dict] = None, 
                          headers: Optional[Dict] = None) -> Dict[str, Any]:
        """发送 HTTP 请求
        
        Args:
            url: 请求 URL
            method: HTTP 方法 (GET, POST)
            params: 查询参数
            headers: 请求头
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, params=params, headers=headers) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/json' in content_type:
                            return await response.json()
                        else:
                            text = await response.text()
                            return {"text": text, "content_type": content_type}
                    else:
                        return {
                            "error": True,
                            "status_code": response.status,
                            "message": f"HTTP {response.status}: {response.reason}"
                        }
        except Exception as e:
            return {
                "error": True,
                "message": f"请求失败: {str(e)}"
            }


# ============================================================
# 1. IP 信息查询工具
# ============================================================

class IPInfoInput(BaseModel):
    """IP 信息查询输入参数"""
    ip_address: Optional[str] = Field(
        default=None,
        description="IP 地址，如果不提供则查询本机 IP"
    )

async def get_ip_info(ip_address: Optional[str] = None) -> Dict[str, Any]:
    """获取 IP 地址信息
    
    使用 ipapi.co 免费 API，无需 API key
    限制：每月 1000 次请求
    
    Args:
        ip_address: IP 地址，如果不提供则查询本机 IP
        
    Returns:
        Dict[str, Any]: IP 信息
    """
    url = "https://ipapi.co"
    if ip_address:
        url = f"https://ipapi.co/{ip_address}/json/"
    else:
        url = "https://ipapi.co/json/"
    
    result = await FreeAPITool.make_request(url)
    
    if "error" in result and result["error"]:
        return result
    
    # 提取关键信息
    simplified_result = {
        "ip": result.get("ip", "未知"),
        "city": result.get("city", "未知"),
        "region": result.get("region", "未知"),
        "country": result.get("country_name", "未知"),
        "isp": result.get("org", "未知"),
        "latitude": result.get("latitude", "未知"),
        "longitude": result.get("longitude", "未知"),
        "timezone": result.get("timezone", "未知"),
        "currency": result.get("currency", "未知"),
        "languages": result.get("languages", "未知")
    }
    
    return simplified_result


# ============================================================
# 2. 汇率查询工具
# ============================================================

class ExchangeRateInput(BaseModel):
    """汇率查询输入参数"""
    base_currency: str = Field(
        default="USD",
        description="基础货币代码，如 USD、EUR、CNY"
    )
    target_currency: str = Field(
        description="目标货币代码，如 USD、EUR、CNY、JPY"
    )
    amount: Optional[float] = Field(
        default=1.0,
        description="金额，默认为 1"
    )

async def get_exchange_rate(base_currency: str = "USD", target_currency: str = "CNY", 
                           amount: float = 1.0) -> Dict[str, Any]:
    """获取汇率信息
    
    使用 exchangerate-api.com 免费 API，无需 API key
    限制：每月 1500 次请求
    
    Args:
        base_currency: 基础货币代码
        target_currency: 目标货币代码
        amount: 金额
        
    Returns:
        Dict[str, Any]: 汇率信息
    """
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency.upper()}"
    
    result = await FreeAPITool.make_request(url)
    
    if "error" in result and result["error"]:
        return result
    
    if "rates" not in result:
        return {"error": True, "message": "API 返回格式错误"}
    
    rates = result.get("rates", {})
    rate = rates.get(target_currency.upper())
    
    if rate is None:
        return {
            "error": True,
            "message": f"未找到货币 {target_currency} 的汇率"
        }
    
    converted_amount = amount * rate
    
    return {
        "base_currency": base_currency.upper(),
        "target_currency": target_currency.upper(),
        "exchange_rate": rate,
        "amount": amount,
        "converted_amount": converted_amount,
        "last_updated": result.get("date", "未知"),
        "available_currencies": list(rates.keys())[:10]  # 只显示前10种货币
    }


# ============================================================
# 3. 随机名言工具
# ============================================================

class QuoteInput(BaseModel):
    """名言查询输入参数"""
    author: Optional[str] = Field(
        default=None,
        description="作者名称，如果不提供则随机获取"
    )
    tags: Optional[str] = Field(
        default=None,
        description="标签，如 motivation、success、life"
    )
    limit: Optional[int] = Field(
        default=1,
        description="返回的名言数量，最大 20"
    )

async def get_random_quote(author: Optional[str] = None, tags: Optional[str] = None, 
                          limit: int = 1) -> Dict[str, Any]:
    """获取随机名言
    
    使用 quotable.io 免费 API，无需 API key
    无限制
    
    Args:
        author: 作者名称
        tags: 标签
        limit: 返回数量
        
    Returns:
        Dict[str, Any]: 名言信息
    """
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
    
    result = await FreeAPITool.make_request(url, params=params)
    
    if "error" in result and result["error"]:
        return result
    
    # 处理返回结果
    quotes = result if isinstance(result, list) else result.get("results", [])
    
    if not quotes:
        return {"message": "未找到匹配的名言"}
    
    formatted_quotes = []
    for quote in quotes[:limit]:
        formatted_quotes.append({
            "content": quote.get("content", ""),
            "author": quote.get("author", "未知"),
            "tags": quote.get("tags", []),
            "length": quote.get("length", 0)
        })
    
    return {
        "count": len(formatted_quotes),
        "quotes": formatted_quotes
    }


# ============================================================
# 4. 公共 API 列表工具
# ============================================================

class PublicAPIInput(BaseModel):
    """公共 API 查询输入参数"""
    category: Optional[str] = Field(
        default=None,
        description="API 类别，如 business、animals、weather"
    )
    search: Optional[str] = Field(
        default=None,
        description="搜索关键词"
    )
    limit: Optional[int] = Field(
        default=10,
        description="返回的 API 数量，最大 50"
    )

async def get_public_apis(category: Optional[str] = None, search: Optional[str] = None, 
                         limit: int = 10) -> Dict[str, Any]:
    """获取公共 API 列表
    
    使用 public-apis 项目的 API，无需 API key
    无限制
    
    Args:
        category: API 类别
        search: 搜索关键词
        limit: 返回数量
        
    Returns:
        Dict[str, Any]: API 列表
    """
    url = "https://api.publicapis.org/entries"
    
    result = await FreeAPITool.make_request(url)
    
    if "error" in result and result["error"]:
        return result
    
    entries = result.get("entries", [])
    
    # 过滤结果
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
    
    # 格式化结果
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
    
    return {
        "count": len(formatted_apis),
        "total": result.get("count", 0),
        "apis": formatted_apis
    }


# ============================================================
# 5. 国内天气查询工具
# ============================================================

# 天气代码到中文描述的映射
WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大部分晴天",
    2: "部分多云",
    3: "多云",
    45: "雾",
    48: "雾凇",
    51: "小雨",
    53: "中雨",
    55: "大雨",
    56: "冻雨",
    57: "强冻雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴有冰雹",
    99: "强雷暴伴有冰雹"
}

class WeatherCNInput(BaseModel):
    """国内天气查询输入参数"""
    city: str = Field(
        description="城市名称，支持中文城市名，如：北京、上海、广州、深圳"
    )

async def get_weather_cn(city: str) -> Dict[str, Any]:
    """查询中国城市天气
    
    使用 Open-Meteo 免费天气 API，无需 API key
    支持中文城市名称，提供实时天气、未来6小时预报和未来3天预报
    
    Args:
        city: 城市名称，如：北京、上海、广州、深圳
        
    Returns:
        Dict[str, Any]: 天气信息
    """
    # Open-Meteo API 配置
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    
    try:
        # 1. 先通过地理编码API获取城市坐标
        geocoding_params = {
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json"
        }
        
        geocoding_result = await FreeAPITool.make_request(geocoding_url, params=geocoding_params)
        
        if "error" in geocoding_result and geocoding_result["error"]:
            return {"error": True, "message": f"地理编码失败: {geocoding_result.get('message', '未知错误')}"}
        
        if not geocoding_result.get("results"):
            return {"error": True, "message": f"未找到城市: {city}"}
        
        location = geocoding_result["results"][0]
        city_name = location["name"]
        latitude = location["latitude"]
        longitude = location["longitude"]
        timezone = location.get("timezone", "Asia/Shanghai")
        
        # 2. 查询天气数据
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
            "hourly": "temperature_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": 3
        }
        
        weather_result = await FreeAPITool.make_request(weather_url, params=weather_params)
        
        if "error" in weather_result and weather_result["error"]:
            return {"error": True, "message": f"天气查询失败: {weather_result.get('message', '未知错误')}"}
        
        current = weather_result.get("current", {})
        hourly = weather_result.get("hourly", {})
        daily = weather_result.get("daily", {})
        
        # 获取当前天气信息
        temperature = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        weather_code = current.get("weather_code", 0)
        wind_speed = current.get("wind_speed_10m", "N/A")
        wind_direction = current.get("wind_direction_10m", "N/A")
        
        # 转换天气代码为中文描述
        weather_desc = WEATHER_CODE_MAP.get(weather_code, "未知天气")
        
        # 获取未来几小时的温度趋势
        hourly_temps = hourly.get("temperature_2m", [])
        hourly_codes = hourly.get("weather_code", [])
        
        # 获取未来3天预报
        daily_codes = daily.get("weather_code", [])
        daily_max = daily.get("temperature_2m_max", [])
        daily_min = daily.get("temperature_2m_min", [])
        
        result = {
            "success": True,
            "city": city_name,
            "temperature": f"{temperature}°C" if temperature != "N/A" else "N/A",
            "weather": weather_desc,
            "humidity": f"{humidity}%" if humidity != "N/A" else "N/A",
            "wind_speed": f"{wind_speed} km/h" if wind_speed != "N/A" else "N/A",
            "wind_direction": f"{wind_direction}°" if wind_direction != "N/A" else "N/A",
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hourly_forecast": list(zip(hourly_temps[:6], hourly_codes[:6])),  # 未来6小时
            "daily_forecast": list(zip(daily_codes[:3], daily_max[:3], daily_min[:3]))  # 未来3天
        }
        
        # 添加生活提示
        try:
            temp_str = result['temperature'].replace('°C', '').strip()
            temp = float(temp_str) if temp_str != 'N/A' else 20
        except:
            temp = 20
        
        weather_desc = result['weather']
        
        tips = []
        if temp > 30:
            tips.append("天气炎热，注意防暑降温，多喝水")
        elif temp < 10:
            tips.append("天气寒冷，注意保暖，添加衣物")
        
        if "雨" in weather_desc:
            tips.append("有降雨，出门请带伞")
        elif "雪" in weather_desc:
            tips.append("有降雪，注意防滑，小心出行")
        elif "雷" in weather_desc:
            tips.append("有雷暴，避免户外活动，注意安全")
        elif "雾" in weather_desc:
            tips.append("有雾，能见度低，注意交通安全")
        
        if not tips:
            tips.append("天气适宜，适合户外活动")
        
        result["tips"] = tips
        
        return result
        
    except Exception as e:
        return {"error": True, "message": f"天气查询异常: {str(e)}"}


# ============================================================
# 6. 占位符图片工具
# ============================================================

class PlaceholderImageInput(BaseModel):
    """占位符图片输入参数"""
    width: Optional[int] = Field(
        default=300,
        description="图片宽度，最大 2000"
    )
    height: Optional[int] = Field(
        default=200,
        description="图片高度，最大 2000"
    )
    text: Optional[str] = Field(
        default=None,
        description="图片上的文字"
    )
    background_color: Optional[str] = Field(
        default="cccccc",
        description="背景颜色（十六进制，不带#）"
    )
    text_color: Optional[str] = Field(
        default="000000",
        description="文字颜色（十六进制，不带#）"
    )

async def get_placeholder_image(width: int = 300, height: int = 200, text: Optional[str] = None,
                              background_color: str = "cccccc", text_color: str = "000000") -> Dict[str, Any]:
    """获取占位符图片 URL
    
    使用 placeholder.com 服务，无需 API key
    无限制
    
    Args:
        width: 图片宽度
        height: 图片高度
        text: 图片上的文字
        background_color: 背景颜色
        text_color: 文字颜色
        
    Returns:
        Dict[str, Any]: 图片信息
    """
    # 限制最大尺寸
    width = min(width, 2000)
    height = min(height, 2000)
    
    # 构建 URL
    base_url = f"https://via.placeholder.com/{width}x{height}"
    
    params = []
    if background_color:
        params.append(f"bg={background_color}")
    if text_color:
        params.append(f"text={text_color}")
    
    if text:
        # 对文本进行 URL 编码
        import urllib.parse
        encoded_text = urllib.parse.quote(text)
        params.append(f"text={encoded_text}")
    
    if params:
        url = f"{base_url}/{'/'.join(params)}"
    else:
        url = base_url
    
    return {
        "image_url": url,
        "width": width,
        "height": height,
        "text": text,
        "background_color": f"#{background_color}" if background_color else "默认",
        "text_color": f"#{text_color}" if text_color else "默认",
        "usage": "可直接在 HTML 中使用: <img src='{image_url}' alt='placeholder'>"
    }


# ============================================================
# 工具创建函数
# ============================================================

def create_free_api_tools() -> List[BaseTool]:
    """创建免费 API 工具列表"""
    
    from langchain_core.tools import StructuredTool
    
    # 创建包装函数，确保返回结果而不是协程
    async def wrapped_get_ip_info(**kwargs):
        return await get_ip_info(**kwargs)
    
    async def wrapped_get_exchange_rate(**kwargs):
        return await get_exchange_rate(**kwargs)
    
    async def wrapped_get_random_quote(**kwargs):
        return await get_random_quote(**kwargs)
    
    async def wrapped_get_public_apis(**kwargs):
        return await get_public_apis(**kwargs)
    
    async def wrapped_get_placeholder_image(**kwargs):
        return await get_placeholder_image(**kwargs)
    
    async def wrapped_get_weather_cn(**kwargs):
        return await get_weather_cn(**kwargs)
    
    tools = [
        # 1. IP 信息查询工具
        StructuredTool.from_function(
            name="api_ip_info",
            func=wrapped_get_ip_info,
            description="获取 IP 地址的详细信息，包括地理位置、ISP、时区等。无需 API key。",
            args_schema=IPInfoInput,
            coroutine=wrapped_get_ip_info
        ),
        
        # 2. 汇率查询工具
        StructuredTool.from_function(
            name="api_exchange_rate",
            func=wrapped_get_exchange_rate,
            description="查询货币汇率并计算兑换金额。支持多种货币。无需 API key。",
            args_schema=ExchangeRateInput,
            coroutine=wrapped_get_exchange_rate
        ),
        
        # 3. 随机名言工具
        StructuredTool.from_function(
            name="api_random_quote",
            func=wrapped_get_random_quote,
            description="获取随机名言，可按作者或标签筛选。无需 API key。",
            args_schema=QuoteInput,
            coroutine=wrapped_get_random_quote
        ),
        
        # 4. 公共 API 列表工具
        StructuredTool.from_function(
            name="api_public_apis",
            func=wrapped_get_public_apis,
            description="搜索和浏览公共 API 列表，可按类别或关键词筛选。无需 API key。",
            args_schema=PublicAPIInput,
            coroutine=wrapped_get_public_apis
        ),
        
        # 5. 国内天气查询工具
        StructuredTool.from_function(
            name="api_weather_cn",
            func=wrapped_get_weather_cn,
            description="查询中国城市天气，支持中文城市名（如：北京、上海、广州）。提供实时天气、未来6小时预报和未来3天预报。无需 API key。",
            args_schema=WeatherCNInput,
            coroutine=wrapped_get_weather_cn
        ),
        
        # 6. 占位符图片工具
        StructuredTool.from_function(
            name="api_placeholder_image",
            func=wrapped_get_placeholder_image,
            description="生成占位符图片 URL，可自定义尺寸、颜色和文字。无需 API key。",
            args_schema=PlaceholderImageInput,
            coroutine=wrapped_get_placeholder_image
        ),
    ]
    
    return tools


# ============================================================
# 测试函数
# ============================================================

async def test_free_apis():
    """测试免费 API 工具"""
    print("🧪 测试免费 API 工具...")
    
    # 测试 IP 信息查询
    print("\n1. 测试 IP 信息查询:")
    ip_info = await get_ip_info()
    print(f"   IP: {ip_info.get('ip', '未知')}")
    print(f"   位置: {ip_info.get('city', '未知')}, {ip_info.get('country', '未知')}")
    
    # 测试汇率查询
    print("\n2. 测试汇率查询:")
    exchange_rate = await get_exchange_rate("USD", "CNY", 100)
    print(f"   汇率: 1 USD = {exchange_rate.get('exchange_rate', '未知')} CNY")
    print(f"   100 USD = {exchange_rate.get('converted_amount', '未知')} CNY")
    
    # 测试随机名言
    print("\n3. 测试随机名言:")
    quote = await get_random_quote(limit=1)
    if quote.get('quotes'):
        first_quote = quote['quotes'][0]
        print(f"   名言: {first_quote.get('content', '未知')}")
        print(f"   作者: {first_quote.get('author', '未知')}")
    
    # 测试公共 API 列表
    print("\n4. 测试公共 API 列表:")
    apis = await get_public_apis(category="weather", limit=3)
    print(f"   找到 {apis.get('count', 0)} 个天气相关 API")
    for api in apis.get('apis', [])[:2]:
        print(f"   - {api.get('name', '未知')}: {api.get('description', '无描述')[:50]}...")
    
    # 测试国内天气查询
    print("\n5. 测试国内天气查询:")
    weather = await get_weather_cn("北京")
    if weather.get('success'):
        print(f"   城市: {weather.get('city', '未知')}")
        print(f"   温度: {weather.get('temperature', '未知')}")
        print(f"   天气: {weather.get('weather', '未知')}")
        print(f"   湿度: {weather.get('humidity', '未知')}")
        print(f"   风速: {weather.get('wind_speed', '未知')}")
        if weather.get('tips'):
            print(f"   生活提示: {', '.join(weather['tips'])}")
    else:
        print(f"   ❌ 查询失败: {weather.get('message', '未知错误')}")
    
    # 测试占位符图片
    print("\n6. 测试占位符图片:")
    image = await get_placeholder_image(400, 300, "测试图片")
    print(f"   图片 URL: {image.get('image_url', '未知')}")
    print(f"   尺寸: {image.get('width', '未知')}x{image.get('height', '未知')}")
    
    print("\n✅ 免费 API 工具测试完成！")


if __name__ == "__main__":
    asyncio.run(test_free_apis())
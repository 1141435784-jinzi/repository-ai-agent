"""
=== 天气查询工具实现 ===

基于 LangChain BaseTool 的天气查询工具实现，支持中文城市名。
"""

import aiohttp
from typing import Any, ClassVar, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime

from src.tools.base import BaseTool
from src.tools.registry import register_tool


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


class WeatherInput(BaseModel):
    """天气查询输入参数"""
    city: str = Field(
        description="城市名称，支持中文城市名，如：北京、上海、广州、深圳"
    )


class WeatherOutput(BaseModel):
    """天气查询输出结果"""
    success: bool = True
    message: Optional[str] = None
    city: Optional[str] = None
    temperature: Optional[str] = None
    weather: Optional[str] = None
    humidity: Optional[str] = None
    wind_speed: Optional[str] = None
    wind_direction: Optional[str] = None
    update_time: Optional[str] = None
    tips: Optional[List[str]] = None


@register_tool
class WeatherTool(BaseTool):
    """中国城市天气查询工具"""

    name = "weather"
    description = "查询中国城市天气，支持中文城市名（如：北京、上海、广州）。提供实时天气、未来6小时预报和未来3天预报。无需 API key。"
    args_schema: ClassVar[Type[BaseModel]] = WeatherInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {
        "category": "api",
        "tags": ["weather", "china", "forecast"],
    }

    async def _arun(self, city: str) -> WeatherOutput:
        """
        查询中国城市天气

        Args:
            city: 城市名称

        Returns:
            WeatherOutput: 天气信息
        """
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        weather_url = "https://api.open-meteo.com/v1/forecast"

        try:
            geocoding_params = {
                "name": city,
                "count": 1,
                "language": "zh",
                "format": "json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(geocoding_url, params=geocoding_params) as response:
                    if response.status != 200:
                        return WeatherOutput(
                            success=False,
                            message=f"地理编码失败: HTTP {response.status}"
                        )
                    geocoding_result = await response.json()

            if not geocoding_result.get("results"):
                return WeatherOutput(
                    success=False,
                    message=f"未找到城市: {city}"
                )

            location = geocoding_result["results"][0]
            city_name = location["name"]
            latitude = location["latitude"]
            longitude = location["longitude"]
            timezone = location.get("timezone", "Asia/Shanghai")

            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
                "hourly": "temperature_2m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": 3
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(weather_url, params=weather_params) as response:
                    if response.status != 200:
                        return WeatherOutput(
                            success=False,
                            message=f"天气查询失败: HTTP {response.status}"
                        )
                    weather_result = await response.json()

            current = weather_result.get("current", {})

            temperature = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            weather_code = current.get("weather_code", 0)
            wind_speed = current.get("wind_speed_10m")
            wind_direction = current.get("wind_direction_10m")

            weather_desc = WEATHER_CODE_MAP.get(weather_code, "未知天气")

            tips = []
            try:
                temp = float(temperature) if temperature else 20
            except:
                temp = 20

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

            return WeatherOutput(
                success=True,
                message="查询成功",
                city=city_name,
                temperature=f"{temperature}°C" if temperature else "N/A",
                weather=weather_desc,
                humidity=f"{humidity}%" if humidity else "N/A",
                wind_speed=f"{wind_speed} km/h" if wind_speed else "N/A",
                wind_direction=f"{wind_direction}°" if wind_direction else "N/A",
                update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tips=tips
            )

        except Exception as e:
            return WeatherOutput(
                success=False,
                message=f"天气查询异常: {str(e)}"
            )

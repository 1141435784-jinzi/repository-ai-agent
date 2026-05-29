"""
=== 时间查询工具实现 ===

基于 LangChain BaseTool 的时间查询工具，支持获取当前系统时间和时区信息。
"""

import datetime
import time
from typing import Any, ClassVar, Dict, Optional, Type, List

from pydantic import BaseModel, Field

from src.tools.base import BaseTool
from src.tools.registry import register_tool


# 时区映射表
TIMEZONE_MAP = {
    "Asia/Shanghai": {"name": "中国标准时间", "offset": "+08:00", "region": "中国（北京、上海、广州、深圳等）"},
    "Asia/Tokyo": {"name": "日本标准时间", "offset": "+09:00", "region": "日本"},
    "Asia/Seoul": {"name": "韩国标准时间", "offset": "+09:00", "region": "韩国"},
    "Asia/Hong_Kong": {"name": "香港时间", "offset": "+08:00", "region": "中国香港"},
    "Asia/Taipei": {"name": "台北时间", "offset": "+08:00", "region": "中国台湾"},
    "UTC": {"name": "协调世界时", "offset": "+00:00", "region": "国际标准时间"},
    "America/New_York": {"name": "美国东部时间", "offset": "-04:00/-05:00", "region": "美国东部（纽约、华盛顿等）"},
    "America/Los_Angeles": {"name": "美国太平洋时间", "offset": "-07:00/-08:00", "region": "美国西部（洛杉矶、旧金山等）"},
    "Europe/London": {"name": "英国夏令时间", "offset": "+01:00/+00:00", "region": "英国"},
    "Europe/Paris": {"name": "中欧时间", "offset": "+01:00/+02:00", "region": "法国、德国等"},
}


class DatetimeInput(BaseModel):
    """时间查询输入参数"""
    timezone: Optional[str] = Field(
        default=None,
        description="时区标识，如：Asia/Shanghai、America/New_York、Europe/London。不指定则使用系统默认时区。"
    )


class DatetimeOutput(BaseModel):
    """时间查询输出结果"""
    success: bool = True
    message: Optional[str] = None
    current_time: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None
    timezone_name: Optional[str] = None
    utc_offset: Optional[str] = None
    region: Optional[str] = None
    day_of_week: Optional[str] = None
    week_number: Optional[int] = None
    is_daylight_saving: Optional[bool] = None
    timestamp: Optional[int] = None
    formatted_time: Optional[str] = None


@register_tool
class DatetimeTool(BaseTool):
    """系统时间查询工具"""

    name = "datetime"
    description = "获取当前系统时间和时区信息。支持查询指定时区的时间，如北京、纽约、伦敦等。可获取当前日期、时间、星期几、周数等信息。"
    args_schema: ClassVar[Type[BaseModel]] = DatetimeInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {
        "category": "system",
        "tags": ["datetime", "time", "clock", "timezone"],
    }

    def _run(self, timezone: Optional[str] = None) -> DatetimeOutput:
        """
        查询当前时间

        Args:
            timezone: 时区标识，可选，如：Asia/Shanghai

        Returns:
            DatetimeOutput: 时间信息
        """
        try:
            # 获取当前时间戳
            timestamp = int(time.time())

            # 根据时区获取时间
            if timezone:
                try:
                    # 使用指定时区
                    tz_info = datetime.timezone(datetime.timedelta(hours=8))  # 默认东八区
                    # 尝试获取时区信息
                    tz_data = TIMEZONE_MAP.get(timezone)
                    if tz_data:
                        # 解析时区偏移
                        offset_str = tz_data["offset"].split("/")[0] if "/" in tz_data["offset"] else tz_data["offset"]
                        hours = int(offset_str[:3])
                        tz_info = datetime.timezone(datetime.timedelta(hours=hours))
                except Exception:
                    # 如果时区解析失败，使用系统默认时区
                    tz_info = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            else:
                # 使用系统默认时区
                tz_info = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo

            # 获取当前时间
            now = datetime.datetime.now(tz_info)

            # 获取时区信息
            tz_name = now.strftime("%Z") or "未知时区"
            tz_offset = now.strftime("%z")
            
            # 获取时区区域信息
            region_info = "未知地区"
            for tz_key, tz_data in TIMEZONE_MAP.items():
                if timezone and tz_key.lower() == timezone.lower():
                    region_info = tz_data["region"]
                    tz_name = tz_data["name"]
                    break

            # 获取星期几
            day_of_week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

            # 获取周数
            week_number = now.isocalendar()[1]

            # 判断是否为夏令时
            is_daylight_saving = now.dst() is not None and now.dst() != datetime.timedelta(0)

            # 格式化输出
            formatted_time = now.strftime("%Y年%m月%d日 %H时%M分%S秒")

            return DatetimeOutput(
                success=True,
                message="查询成功",
                current_time=now.isoformat(),
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H:%M:%S"),
                timezone=timezone or str(tz_info),
                timezone_name=tz_name,
                utc_offset=tz_offset,
                region=region_info,
                day_of_week=day_of_week,
                week_number=week_number,
                is_daylight_saving=is_daylight_saving,
                timestamp=timestamp,
                formatted_time=formatted_time
            )

        except Exception as e:
            return DatetimeOutput(
                success=False,
                message=f"时间查询异常: {str(e)}"
            )

    async def _arun(self, timezone: Optional[str] = None) -> DatetimeOutput:
        """
        异步查询当前时间

        Args:
            timezone: 时区标识，可选

        Returns:
            DatetimeOutput: 时间信息
        """
        return self._run(timezone=timezone)

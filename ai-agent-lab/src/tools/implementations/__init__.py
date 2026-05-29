"""
=== 工具实现层 ===

包含所有具体的工具实现类，这些工具都继承自 BaseTool。

【工具列表】：
1. CalculatorTool - 数学计算工具
2. WeatherTool - 天气查询工具
3. DatetimeTool - 时间查询工具
4. IPInfoTool - IP 信息查询工具
5. ExchangeRateTool - 汇率查询工具
6. RandomQuoteTool - 随机名言工具
7. PublicAPIsTool - 公共 API 列表工具
8. PlaceholderImageTool - 占位符图片工具
9. FlightBookingTool - 机票预订工具
10. CancelFlightTool - 取消机票预订工具
11. QueryFlightTool - 查询机票预订工具
12. TrainBookingTool - 高铁票预订工具
13. CancelTrainTool - 取消高铁票预订工具
14. QueryTrainTool - 查询高铁票预订工具
15. MCPToolWrapper - MCP 工具包装器
"""

# 导入所有工具实现
from .calculator import CalculatorTool
from .weather import WeatherTool
from .datetime import DatetimeTool
from .free_api import (
    IPInfoTool,
    ExchangeRateTool,
    RandomQuoteTool,
    PublicAPIsTool,
    PlaceholderImageTool,
)
from .ticket_booking import (
    FlightBookingTool,
    CancelFlightTool,
    QueryFlightTool,
    TrainBookingTool,
    CancelTrainTool,
    QueryTrainTool,
)
from .mcp_adapter import MCPToolWrapper

# 导出列表
__all__ = [
    # 实用工具
    "CalculatorTool",
    "WeatherTool",
    "DatetimeTool",
    
    # 免费 API 工具
    "IPInfoTool",
    "ExchangeRateTool",
    "RandomQuoteTool",
    "PublicAPIsTool",
    "PlaceholderImageTool",
    
    # 票务工具
    "FlightBookingTool",
    "CancelFlightTool",
    "QueryFlightTool",
    "TrainBookingTool",
    "CancelTrainTool",
    "QueryTrainTool",
    
    # MCP 工具
    "MCPToolWrapper",
]

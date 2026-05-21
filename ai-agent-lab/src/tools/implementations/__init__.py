"""
=== 工具实现层 ===

包含所有具体的工具实现类，这些工具都继承自 BaseTool。

【工具列表】：
1. CalculatorTool - 数学计算工具
2. WeatherTool - 天气查询工具
3. IPInfoTool - IP 信息查询工具
4. ExchangeRateTool - 汇率查询工具
5. RandomQuoteTool - 随机名言工具
6. PublicAPIsTool - 公共 API 列表工具
7. PlaceholderImageTool - 占位符图片工具
8. CreatePaymentTool - 创建支付订单工具
9. ProcessPaymentTool - 处理支付工具
10. QueryPaymentTool - 查询支付状态工具
11. RefundPaymentTool - 退款工具
12. FlightBookingTool - 机票预订工具
13. CancelFlightTool - 取消机票预订工具
14. QueryFlightTool - 查询机票预订工具
15. TrainBookingTool - 高铁票预订工具
16. CancelTrainTool - 取消高铁票预订工具
17. QueryTrainTool - 查询高铁票预订工具
18. MCPToolWrapper - MCP 工具包装器
"""

# 导入所有工具实现
from .calculator import CalculatorTool
from .weather import WeatherTool
from .free_api import (
    IPInfoTool,
    ExchangeRateTool,
    RandomQuoteTool,
    PublicAPIsTool,
    PlaceholderImageTool,
)
from .payment import (
    CreatePaymentTool,
    ProcessPaymentTool,
    QueryPaymentTool,
    RefundPaymentTool,
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
    
    # 免费 API 工具
    "IPInfoTool",
    "ExchangeRateTool",
    "RandomQuoteTool",
    "PublicAPIsTool",
    "PlaceholderImageTool",
    
    # 支付工具
    "CreatePaymentTool",
    "ProcessPaymentTool",
    "QueryPaymentTool",
    "RefundPaymentTool",
    
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

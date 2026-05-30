"""
=== 企业级 Agent 工具模块 ===

基于大厂（字节、阿里、腾讯、OpenAI 企业版）通用架构设计的工具系统。

【架构分层】：
┌─────────────────────────────────────────────────────────────┐
│  5. 工具开放接口层 (ToolAPI)                                │
│     └── 为 Agent 提供统一的工具访问接口                      │
├─────────────────────────────────────────────────────────────┤
│  4. 工具中间件层 (Middleware)                               │
│     └── AOP 切面：日志、监控、重试、校验、脱敏、审计          │
├─────────────────────────────────────────────────────────────┤
│  3. 工具执行器层 (ToolExecutor)                             │
│     └── 统一调用、日志、监控、超时、限流、权限控制            │
├─────────────────────────────────────────────────────────────┤
│  2. 工具注册中心层 (ToolRegistry)                           │
│     └── 自动扫描、动态加载、统一管理                          │
├─────────────────────────────────────────────────────────────┤
│  1. 工具基类层 (BaseTool)                                   │
│     └── 所有工具继承的抽象基类，统一接口标准                  │
└─────────────────────────────────────────────────────────────┘

【核心设计原则】：
1. 统一接口：所有工具通过相同接口访问
2. 类型安全：使用 Pydantic 进行参数验证
3. 易于扩展：新增工具只需继承 BaseTool
4. 配置驱动：工具配置集中管理
5. 可观测性：完整的日志和监控支持
6. 安全性：权限控制和数据脱敏

【使用方式】：
python
# Agent 使用方式（最简）
from src.tools import tool_api, to_langchain_tools

# 获取所有工具
tools = tool_api.get_tools()

# 调用工具
result = await tool_api.call_tool("calculator", expression="2 + 3 * 4")

# 转换为 LangChain 工具
langchain_tools = to_langchain_tools()

# 创建 Agent
agent = create_agent(llm, langchain_tools)


【目录结构】：
src/tools/
├── __init__.py           # 统一导出接口
├── base/                 # 工具基类层
│   └── __init__.py       # BaseTool, SyncTool, AsyncTool
├── registry/             # 工具注册中心层
│   └── __init__.py       # ToolRegistry
├── executor/             # 工具执行器层
│   └── __init__.py       # ToolExecutor
├── middleware/           # 工具中间件层
│   └── __init__.py       # 日志、监控、校验、脱敏、审计
├── api/                  # 工具开放接口层
│   └── __init__.py       # ToolAPI
├── mcp/                  # MCP 工具适配器
│   └── __init__.py       # MCP 客户端管理
└── implementations/      # 工具实现层
    ├── __init__.py
    ├── calculator.py     # 数学计算工具
    ├── weather.py        # 天气查询工具
    ├── free_api.py       # 免费 API 工具（IP、汇率、名言等）
    ├── ticket_booking.py # 票务预订工具（机票、高铁）
    └── mcp_adapter.py    # MCP 工具适配器
"""

# ==================== 统一导出接口 ====================

# 1. 工具基类层
from src.tools.base import (
    BaseTool,
    SyncTool,
    AsyncTool,
    ToolMetadata,
    ToolInput,
    ToolOutput,
)

# 2. 工具注册中心层
from src.tools.registry import (
    ToolRegistry,
    tool_registry,
    register_tool,
)

# 3. 工具执行器层
from src.tools.executor import (
    ToolExecutor,
    ExecutionContext,
    ExecutionResult,
    RateLimiter,
    tool_executor,
    call_tool,
    call_tool_with_retry,
)

# 4. 工具中间件层
from src.tools.middleware import (
    BaseMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
    DataMaskingMiddleware,
    AuditMiddleware,
)

# 5. 工具开放接口层
from src.tools.api import (
    ToolAPI,
    tool_api,
    ToolInfo,
    SkillInfo,
    get_all_tools,
    get_all_skills,
    get_tool_info,
    get_tool_info_list,
    get_skill_info_list,
    to_langchain_tools,
)

# 6. 工具实现层
from src.tools.implementations import (
    CalculatorTool,
    WeatherTool,
    IPInfoTool,
    ExchangeRateTool,
    RandomQuoteTool,
    PublicAPIsTool,
    PlaceholderImageTool,
    FlightBookingTool,
    CancelFlightTool,
    QueryFlightTool,
    TrainBookingTool,
    CancelTrainTool,
    QueryTrainTool,
    MCPToolWrapper,
)

# 7. MCP 工具适配器
from src.tools.implementations.mcp_adapter import (
    get_mcp_client_manager as get_mcp_manager,
    close_mcp_client_manager as close_mcp_manager,
    is_mcp_available,
    get_available_mcp_tools,
    call_mcp_tool,
    refresh_mcp_tools,
    get_mcp_metrics as get_mcp_metrics,
    get_mcp_tool_info as get_mcp_tool_info,
    get_langchain_tools_from_mcp,
    refresh_tools,
    MCPServerConfig,
)

# ==================== 向后兼容接口 ====================

# 保持与原有接口的兼容性
async def get_tools():
    """获取所有工具（向后兼容）"""
    return tool_api.get_tool_instances()

# ==================== 导出列表 ====================

__all__ = [
    # 工具基类层
    "BaseTool",
    "SyncTool",
    "AsyncTool",
    "ToolMetadata",
    "ToolInput",
    "ToolOutput",
    
    # 工具注册中心层
    "ToolRegistry",
    "tool_registry",
    "register_tool",
    
    # 工具执行器层
    "ToolExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "RateLimiter",
    "tool_executor",
    "call_tool",
    "call_tool_with_retry",
    
    # 工具中间件层
    "BaseMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "ValidationMiddleware",
    "DataMaskingMiddleware",
    "AuditMiddleware",
    
    # 工具开放接口层
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
    
    # 工具实现层 - 实用工具
    "CalculatorTool",
    "WeatherTool",
    
    # 工具实现层 - 免费 API
    "IPInfoTool",
    "ExchangeRateTool",
    "RandomQuoteTool",
    "PublicAPIsTool",
    "PlaceholderImageTool",
    
    # 工具实现层 - 票务工具
    "FlightBookingTool",
    "CancelFlightTool",
    "QueryFlightTool",
    "TrainBookingTool",
    "CancelTrainTool",
    "QueryTrainTool",
    
    # 工具实现层 - MCP
    "MCPToolWrapper",
    
    # MCP 工具适配器
    "get_mcp_manager",
    "close_mcp_manager",
    "is_mcp_available",
    "get_available_mcp_tools",
    "call_mcp_tool",
    "refresh_mcp_tools",
    "get_mcp_metrics",
    "get_mcp_tool_info",
    "MCPServerConfig",
    "get_langchain_tools_from_mcp",
    "refresh_tools",
    
    # 向后兼容接口
    "get_tools",
]

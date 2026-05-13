"""
=== FastAPI 接口层 — 将 Agent 封装为 HTTP 服务 ===

【知识点】为什么要用 FastAPI 封装 Agent？
在企业级场景中，Agent 不是一个命令行程序，而是一个后端服务：
- 前端（Web/App/小程序）通过 HTTP 请求与 Agent 交互
- 多个客户端可以同时访问，每个客户端有独立的会话（thread_id）
- 支持水平扩展：多个 API 实例 + 共享存储 = 高并发

【现实例子】就像企业级 Agent 中的客服系统：
- 用户在网页上打开聊天窗口 → 前端发 HTTP 请求到这个 API
- API 通过 thread_id 找到该用户的会话历史 → 调用 Agent → 返回回复
- 用户关闭浏览器明天再来 → 同一个 thread_id → 接着聊

【接口设计】
POST /chat          — 发送消息并获取 Agent 回复（核心接口）
POST /chat/stream   — 发送消息并以 SSE 流式返回回复（提升体验）
POST /session/new   — 创建新会话，返回 thread_id
GET  /health        — 健康检查（运维监控用）
"""

import src.config  # 必须第一个 import，确保离线模式 patch 生效

import os
import uuid
import logging
import time
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from src.agents import get_async_agent
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, MAX_ITERATIONS
from src.memory import get_async_checkpointer, close_async_pool
from src.prompts import sanitize_input, sanitize_output
from src.llm.gateway import get_call_stats
from src.utils.logger import setup_logging, get_logger, WorkflowLogger

# ============================================================
# 日志配置
# ============================================================
setup_logging(log_dir="log", level=logging.INFO)
logger = get_logger(__name__)
workflow_logger = WorkflowLogger(logger)

# Prometheus 指标集成
try:
    from prometheus_metrics import (
        metrics_endpoint,
        record_api_request,
        record_user_session,
        record_agent_call,
        PROMETHEUS_AVAILABLE
    )
    PROMETHEUS_ENABLED = PROMETHEUS_AVAILABLE
except ImportError:
    # 如果 prometheus_metrics 模块不存在，使用空实现
    PROMETHEUS_ENABLED = False
    
    def metrics_endpoint():
        from fastapi import Response
        return Response("Prometheus metrics not available", status_code=501)
    
    def record_api_request(*args, **kwargs):
        pass
    
    def record_user_session(*args, **kwargs):
        pass
    
    def record_agent_call(*args, **kwargs):
        pass


# ============================================================
# 请求/响应模型（Pydantic）
# ============================================================
# 【知识点】用 Pydantic BaseModel 定义接口的输入输出
# FastAPI 会自动做参数校验、生成 OpenAPI 文档（Swagger UI）


class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., min_length=1, max_length=10000, description="用户消息内容")
    thread_id: str = Field(..., description="会话 ID，用于区分不同用户/会话")
    model: str = Field(default="deepseek", description="指定 LLM Provider（deepseek / zhipu），为空时使用默认")


class ChatResponse(BaseModel):
    """聊天响应体"""
    reply: str = Field(..., description="Agent 的回复内容")
    thread_id: str = Field(..., description="当前会话 ID")


class SessionResponse(BaseModel):
    """新建会话响应体"""
    thread_id: str = Field(..., description="新创建的会话 ID")


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期钩子

    【知识点】FastAPI 的 lifespan 机制：
    - yield 之前的代码在应用启动时执行（初始化资源）
    - yield 之后的代码在应用关闭时执行（清理资源）

    【生产实践】PostgreSQL 连接池生命周期管理：
    - 启动时：异步 Agent 初始化会自动创建异步连接池
    - 关闭时：必须显式关闭连接池，释放数据库连接资源
    - 不关闭会导致：连接泄漏 → 数据库 max_connections 耗尽 → 新请求全部失败

    Args:
        app: FastAPI 应用实例，由框架自动传入

    Returns:
        AsyncGenerator: 异步生成器，yield 前为启动逻辑，yield 后为关闭逻辑
    """
    

    
    # 启动时检查（Fail Fast：缺少关键配置直接阻止启动）
    from src.config import DEEPSEEK_API_KEY, LLM_ZHIPU_API_KEY
    if not DEEPSEEK_API_KEY and not LLM_ZHIPU_API_KEY:
        error_msg = "未设置任何 LLM API Key（DEEPSEEK_API_KEY / LLM_ZHIPU_API_KEY），服务无法启动"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("Agent API 开始启动...")

    # 【生产实践】预热异步 Agent（含 PostgreSQL 连接池初始化）
    # 确保第一个用户请求不需要等待连接池创建
    logger.info("正在预热异步 Agent 及 PostgreSQL 连接池...")
    await get_async_agent()
    logger.info("异步 Agent 及 PostgreSQL 连接池预热完成")
    
    # 【生产实践】初始化 MCP 管理器
    # 确保 MCP 工具在服务启动时就可用
    logger.info("正在初始化 MCP 管理器...")
    try:
        from src.tools import get_mcp_manager, is_mcp_available
        
        mcp_manager = await get_mcp_manager()
        mcp_available = await is_mcp_available()
        
        if mcp_available:
            servers = await mcp_manager.list_servers()
            connected_servers = [s for s in servers if s["status"] == "connected"]
            tools = await mcp_manager.list_tools()
            
            logger.info(f"MCP 管理器初始化成功")
            logger.info(f"  已连接服务器: {len(connected_servers)}/{len(servers)}")
            logger.info(f"  可用工具: {len(tools)} 个")
            
            # 显示已连接的服务器
            for server in connected_servers:
                logger.info(f"    - {server['name']}: {server['tools_count']} 个工具")
            
            # 特别显示 weather_cn 工具状态
            weather_tools = [t for t in tools if "weather_cn" in t.get("name", "")]
            if weather_tools:
                logger.info(f"  ✅ weather_cn 工具可用: {len(weather_tools)} 个")
                for tool in weather_tools:
                    logger.info(f"    - {tool['name']}: {tool.get('description', '无描述')[:50]}...")
            else:
                logger.warning("  ⚠️ weather_cn 工具未找到")
        else:
            logger.warning("MCP 管理器初始化失败或没有可用的 MCP 服务器")
            
    except Exception as e:
        logger.warning(f"MCP 管理器初始化失败: {e}")
        logger.warning("MCP 功能可能不可用，但服务将继续运行")
    
    logger.info("Agent API 启动成功。")
    logger.info("=" * 60)
    logger.info("服务已就绪，可通过以下地址访问:")
    logger.info("  - API 文档: http://localhost:8000/docs")
    logger.info("  - 健康检查: http://localhost:8000/health")
    logger.info("  - MCP 状态: http://localhost:8000/mcp/status")
    logger.info("=" * 60)
    
    # 检查 Prometheus 是否可用
    if PROMETHEUS_ENABLED:
        logger.info("Prometheus 监控已启用")
    else:
        logger.warning("Prometheus 监控未启用，请安装 prometheus-client")

    yield

    # 【生产实践】优雅关闭 — 释放所有资源
    logger.info("Agent API 正在关闭，释放所有资源...")
    
    # 1. 关闭 MCP 管理器
    try:
        from src.tools import close_mcp_manager
        logger.info("正在关闭 MCP 管理器...")
        await close_mcp_manager()
        logger.info("MCP ���理器已关闭")
    except Exception as e:
        logger.warning(f"关闭 MCP 管理器失败: {e}")
    
    # 2. 释放 PostgreSQL 异步连接池
    logger.info("正在释放 PostgreSQL 连接池资源...")
    await close_async_pool()
    logger.info("PostgreSQL 连接池资源已释放")
    
    logger.info("Agent API 已完全关闭")
    logger.info("=" * 60)


# ============================================================
# 创建 FastAPI 应用
# ============================================================
app = FastAPI(
    title="Agent Lab API",
    description="企业级 ReAct Agent 服务 — 基于 LangGraph + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# 【知识点】CORS 中间件 — 允许前端跨域访问
# 开发阶段允许所有来源，生产环境应限制为具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境改为具体域名列表
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 请求日志和监控中间件
@app.middleware("http")
async def logging_and_prometheus_middleware(request: Request, call_next):
    """记录 API 请求日志和 Prometheus 指标的中间件"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        
        # 记录 API 请求日志
        if response.status_code < 400:
            logger.info(f"API请求: {request.method} {request.url.path} - {response.status_code} ({elapsed:.3f}s)")
        elif response.status_code < 500:
            logger.warning(f"客户端错误: {request.method} {request.url.path} - {response.status_code} ({elapsed:.3f}s)")
        else:
            logger.error(f"服务器错误: {request.method} {request.url.path} - {response.status_code} ({elapsed:.3f}s)")
        
        # 记录 Prometheus 指标
        if PROMETHEUS_ENABLED:
            record_api_request(
                method=request.method,
                endpoint=request.url.path,
                duration=elapsed,
                status_code=response.status_code
            )
        
        return response
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        # 记录错误日志
        logger.error(f"API请求异常: {request.method} {request.url.path} - 500 ({elapsed:.3f}s) - {str(e)}")
        
        # 记录 Prometheus 错误指标
        if PROMETHEUS_ENABLED:
            record_api_request(
                method=request.method,
                endpoint=request.url.path,
                duration=elapsed,
                status_code=500  # 内部错误
            )
        
        raise

# 【知识点】静态文件托管
# 将 static 目录挂载到 /static 路径，前端 HTML/CSS/JS 放在这里
# 访问 http://localhost:8000/static/index.html 即可打开聊天界面
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "static")
app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")


# ============================================================
# 工具函数
# ============================================================
def extract_ai_response(result: dict) -> str:
    """从 Agent 执行结果中提取最终的 AI 回复

    Args:
        result: Agent invoke() 返回的状态字典，包含 messages 列表

    Returns:
        str: 最后一条有内容的 AIMessage 的文本；如果没有找到则返回默认提示语
    """
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "抱歉，我没有生成有效的回复。"


# ============================================================
# API 接口
# ============================================================
@app.get("/metrics")
async def metrics():
    """Prometheus metrics 端点
    
    【知识点】Prometheus 监控标准端点：
    - Prometheus 定期从此端点抓取指标数据
    - 返回格式为 Prometheus 文本格式
    - 包含所有四大黄金指标：延迟、流量、错误、饱和度
    
    Args:
        无参数
        
    Returns:
        Response: Prometheus 格式的指标数据
    """
    return await metrics_endpoint()


@app.get("/health")
async def health_check():
    """健康检查接口

    【知识点】健康检查是生产部署的标配：
    - 负载均衡器（Nginx/ALB）定期调用此接口判断实例是否存活
    - K8s 的 liveness/readiness probe 也依赖它

    Args:
        无参数

    Returns:
        dict: 包含 status（服务状态）、model（模型名称）、api_key_configured（API Key 是否已配置）
    """
    return {
        "status": "healthy",
        "model": DEEPSEEK_MODEL,
        "api_key_configured": bool(DEEPSEEK_API_KEY),
        "prometheus_enabled": PROMETHEUS_ENABLED,
    }


@app.get("/llm/stats")
async def llm_stats():
    """LLM Gateway 调用统计

    【知识点】运维监控接口，用于观察：
    - 各 Provider 的调用次数分布
    - 降级触发次数（fallback_calls 过高说明主模型不稳定）
    - 错误率（error_calls / total_calls）

    Args:
        无参数

    Returns:
        dict: 调用统计数据
    """
    stats = get_call_stats()
    return {
        "total_calls": stats.total_calls,
        "success_calls": stats.success_calls,
        "fallback_calls": stats.fallback_calls,
        "error_calls": stats.error_calls,
        "calls_by_provider": stats.calls_by_provider,
    }


@app.get("/mcp/status")
async def mcp_status():
    """MCP 工具状态查询

    【知识点】MCP监控接口，用于观察：
    - 各MCP服务器的可用状态
    - 可用工具列表
    - 工具调用统计

    Args:
        无参数

    Returns:
        dict: MCP状态信息
    """
    try:
        from src.tools import get_mcp_manager
        from src.tools.mcp import get_available_mcp_tools
        
        # 获取MCP管理器（会自动初始化）
        manager = await get_mcp_manager()
        
        # 列出服务器和工具
        servers = await manager.list_servers()
        tools = await get_available_mcp_tools()
        
        # 转换服务器格式以兼容旧接口
        clients = []
        for server in servers:
            clients.append({
                "name": server["name"],
                "available": server["status"] == "connected",
                "status": server["status"],
                "disabled": server["disabled"],
                "tools_count": server["tools_count"],
                "error_count": server["error_count"]
            })
        
        # 统计信息
        total_clients = len(clients)
        available_clients = sum(1 for c in clients if c.get("available", False))
        total_tools = len(tools)
        available_tools = sum(1 for t in tools if t.get("available", False))
        
        # 查找weather_cn工具
        weather_tools = [t for t in tools if "weather_cn" in t.get("name", "")]
        
        return {
            "status": "healthy",
            "clients": {
                "total": total_clients,
                "available": available_clients,
                "list": clients
            },
            "tools": {
                "total": total_tools,
                "available": available_tools,
                "list": tools[:10]  # 只返回前10个工具
            },
            "weather_cn_available": len(weather_tools) > 0,
            "weather_tools": weather_tools
        }
        
    except Exception as e:
        logger.error(f"MCP状态查询失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "clients": {"total": 0, "available": 0, "list": []},
            "tools": {"total": 0, "available": 0, "list": []},
            "weather_cn_available": False,
            "weather_tools": []
        }


@app.post("/mcp/tool/call")
async def mcp_tool_call(request: dict):
    """直接调用MCP工具

    【知识点】调试接口，用于测试MCP工具功能
    - 支持直接调用任意MCP工具
    - 返回原始调用结果
    - 用于开发和调试

    Args:
        request: 包含server, tool, params的字典

    Returns:
        dict: 工具调用结果
    """
    try:
        server = request.get("server")
        tool = request.get("tool")
        params = request.get("params", {})
        
        if not server or not tool:
            raise HTTPException(
                status_code=400,
                detail="缺少必要参数: server 和 tool"
            )
        
        from src.tools import get_mcp_manager
        
        manager = await get_mcp_manager()
        result = await manager.call_tool_direct(server, tool, **params)
        
        return {
            "success": True,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP工具调用失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    """创建新会话

    【知识点】会话管理流程：
    1. 前端首次打开聊天窗口 → 调用此接口获取 thread_id
    2. 后续每次发消息都带上这个 thread_id
    3. Checkpointer 通过 thread_id 自动管理对话历史

    Args:
        无参数

    Returns:
        SessionResponse: 包含新生成的 thread_id（UUID 字符串）
    """
    thread_id = str(uuid.uuid4())
    
    # 记录用户会话指标
    if PROMETHEUS_ENABLED:
        record_user_session(status="created")
        record_user_session(status="active")
    
    return SessionResponse(thread_id=thread_id)


# ==================== 【已屏蔽】同步聊天接口 ====================
# 注：当前只提供流式对话接口 /chat/stream
# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     """发送消息并获取 Agent 回复（同步模式）
# 
#     【知识点】这是最核心的接口，完整流程：
#     1. 前端发送 { message: "你好", thread_id: "xxx" }
#     2. API 将 message 包装为 HumanMessage
#     3. 通过 thread_id 让 Checkpointer 自动加载历史消息
#     4. Agent 执行 ReAct 循环（思考 → 工具调用 → 回复）
#     5. 提取最终回复返回给前端
# 
#     【企业实战】为什么前端只传 message 和 thread_id？
#     - 对话历史由后端 Checkpointer 管理，前端不需要维护
#     - 减少网络传输量（不用每次把完整历史发过来）
#     - 安全：历史数据不暴露给前端
# 
#     Args:
#         request: ChatRequest 请求体，包含 message（用户消息）和 thread_id（会话 ID）
# 
#     Returns:
#         ChatResponse: 包含 reply（Agent 回复文本）和 thread_id（当前会话 ID）
# 
#     Raises:
#         HTTPException(503): LLM API Key 未配置
#         HTTPException(500): Agent 执行过程中出错
#     """
#     if not DEEPSEEK_API_KEY:
#         raise HTTPException(status_code=503, detail="LLM API Key 未配置")
# 
#     try:
#         # 【安全防护】输入校验 — 检测 Prompt 注入风险
#         cleaned_input, is_risky, risk_desc = sanitize_input(request.message)
#         if is_risky:
#             logger.warning(f"Prompt 注入风险 — thread_id={request.thread_id}, {risk_desc}")
# 
#         async_agent = await get_async_agent()
#         start_time = time.time()
#         result = await async_agent.ainvoke(
#             {"messages": [HumanMessage(content=cleaned_input)]},
#             config={
#                 "configurable": {"thread_id": request.thread_id, "model": request.model},
#                 "recursion_limit": MAX_ITERATIONS,
#             },
#         )
#         elapsed = time.time() - start_time
#         reply = extract_ai_response(result)
# 
#         # 【安全防护】输出过滤 — 脱敏 + 泄露检测
#         reply = sanitize_output(reply)
#         
#         # 记录 Agent 调用指标
#         if PROMETHEUS_ENABLED:
#             # 这里需要从 result 中提取 Agent 类型和路由信息
#             # 简化处理：假设为通用聊天类型
#             agent_type = "chat"
#             route = "general"  # 可以从 result 中提取实际路由
#             
#             record_agent_call(
#                 agent_type=agent_type,
#                 route=route,
#                 duration=elapsed,
#                 status="success"
#             )
# 
#         return ChatResponse(reply=reply, thread_id=request.thread_id)
# 
#     except Exception as e:
#         logger.error(f"Agent 执行出错: {e}", exc_info=True)
#         
#         # 记录 Agent 错误指标
#         if PROMETHEUS_ENABLED:
#             record_agent_call(
#                 agent_type="chat",
#                 route="error",
#                 duration=0,  # 执行失败，没有有效耗时
#                 status="error"
#             )
#         
#         raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """发送消息并以 SSE 流式返回 Agent 回复

    【知识点】SSE（Server-Sent Events）流式输出：
    - 用户不用等 Agent 完整执行完才看到回复
    - 像 ChatGPT 一样逐字/逐块显示，体验更好
    - 前端用 EventSource 或 fetch + ReadableStream 接收

    【实现原理】
    LangGraph 的 astream_events() 会在 Agent 执行过程中实时产出事件：
    - on_chat_model_stream：LLM 生成的每个 token
    - on_tool_start / on_tool_end：工具调用开始/结束
    我们过滤出 LLM 的 token 流，逐个推送给前端

    Args:
        request: ChatRequest 请求体，包含 message（用户消息）和 thread_id（会话 ID）

    Returns:
        StreamingResponse: SSE 格式的流式响应，每个 chunk 格式为 "data: 文本内容\\n\\n"，
                           流结束时发送 "data: [DONE]\\n\\n"

    Raises:
        HTTPException(503): LLM API Key 未配置
    """
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail="LLM API Key 未配置")
    
    # 记录工作流开始
    workflow_logger.workflow_start(request.thread_id, request.message)
    
    # 【安全防护】输入校验
    cleaned_input, is_risky, risk_desc = sanitize_input(request.message)
    if is_risky:
        logger.warning(f"Prompt 注入风险 — thread_id={request.thread_id[:8]}, {risk_desc}")

    async def event_generator():
        try:
            async_agent = await get_async_agent()
            full_text = ""
            
            async for event in async_agent.astream_events(
                {"messages": [HumanMessage(content=cleaned_input)]},
                config={
                    "configurable": {"thread_id": request.thread_id, "model": request.model},
                    "recursion_limit": MAX_ITERATIONS,
                },
                version="v2",
            ):
                event_type = event.get("event", "")
                
                # LLM 流式事件内容过滤
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    
                    content = None
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk["content"]
                    
                    if content:
                        if not isinstance(content, str):
                            content = str(content)
                        
                        full_text += content
                        escaped = content.replace("\n", "\\n")
                        yield f"data: {escaped}\n\n"
                
                # 工具调用开始事件
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    args = event.get("data", {}).get("input", {})
                    
                    workflow_logger.tool_execution(request.thread_id, tool_name, args)
                    yield f"data: [TOOL_START]{{\"name\": \"{tool_name}\", \"args\": {json.dumps(args)}}}\n\n"
                
                # 工具调用结束事件
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    
                    if hasattr(output, "content"):
                        result = output.content
                    else:
                        result = str(output)
                    
                    workflow_logger.tool_result(request.thread_id, tool_name, True, str(result)[:50])
                    if result:
                        result_str = str(result).replace("\n", "\\n")
                        yield f"data: [TOOL_RESULT]{{\"name\": \"{tool_name}\", \"result\": \"{result_str}\"}}\n\n"

            # 【安全防护】流式输出完成后，对完整文本做泄露检测
            safe_text = sanitize_output(full_text)
            if safe_text != full_text:
                logger.error(f"🚨 流式输出泄露检测触发")
                yield f"data: [REPLACE]{safe_text}\n\n"

            workflow_logger.workflow_end(request.thread_id)
            yield "data: [DONE]\n\n"

        except Exception as e:
            workflow_logger.error(request.thread_id, "chat_stream", e)
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

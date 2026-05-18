"""
=== FastAPI 服务入口 ===

【知识点】FastAPI 服务架构：
1. 使用 Lifespan 管理应用生命周期（启动/关闭）
2. CORS 中间件配置，支持跨域请求
3. 路由模块化，按功能划分
4. 统一的错误处理和日志记录

【生产实践】企业级 FastAPI 项目结构：
- 主入口文件（server.py）：负责应用创建、生命周期、中间件配置
- 路由模块（routes/）：按功能划分 API 路由
- 服务模块（services/）：业务逻辑处理
- 数据模型（models/）：Pydantic 数据模型
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# 全局变量
# ============================================================
mcp_manager = None
app_instance: Optional[FastAPI] = None


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期钩子

    【知识点】FastAPI 的 lifespan 机制：
    - 在应用启动前执行异步初始化（如连接数据库、加载模型）
    - 在应用关闭时执行异步清理（如关闭连接、释放资源）
    - 使用 asynccontextmanager 确保资源正确释放

    【生产实践】启动检查清单：
    1. 初始化日志系统
    2. 初始化 MCP 管理器（可选）
    3. 启动知识库文件监听服务
    4. 检查外部服务状态（如 Prometheus）
    """
    global mcp_manager

    logger.info("=" * 60)
    logger.info("Agent API 正在启动...")
    logger.info("=" * 60)

    # 1. 初始化 MCP 管理器（可选功能）
    PROMETHEUS_ENABLED = False
    try:
        from src.tools import get_mcp_manager
        from src.metrics import PROMETHEUS_AVAILABLE
        mcp_manager = await get_mcp_manager()
        if mcp_manager:
            logger.info("✅ MCP 管理器初始化成功")
        else:
            logger.warning("⚠️ MCP 管理器初始化失败，功能可能不可用")
    except Exception as e:
        logger.warning(f"MCP 管理器初始化失败: {e}")
        logger.warning("MCP 功能可能不可用，但服务将继续运行")

    # 【生产实践】初始化领域专家系统（加载知识库）
    # 这会创建所有专家的 RAG 引擎并加载知识库内容到向量数据库
    logger.info("正在初始化领域专家系统...")
    try:
        from src.agents import initialize_experts
        await initialize_experts()
        logger.info("✅ 领域专家系统初始化成功")
    except Exception as e:
        logger.warning(f"领域专家系统初始化失败: {e}")
        logger.warning("专家功能可能不可用，但服务将继续运行")

    # 【生产实践】启动知识库文件监听服务
    # 当知识库文件发生变化时，自动触发增量更新
    logger.info("正在启动知识库文件监听服务...")
    try:
        from src.rag import start_all_file_watchers
        start_all_file_watchers()
        logger.info("知识库文件监听服务已启动")
    except Exception as e:
        logger.warning(f"文件监听服务启动失败: {e}")
        logger.warning("文件监听功能不可用，知识库更新需要手动触发")

    logger.info("Agent API 启动成功。")
    logger.info("=" * 60)
    logger.info("服务已就绪，可通过以下地址访问:")
    logger.info("  - API 文档: http://localhost:8000/docs")
    logger.info("  - 健康检查: http://localhost:8000/health")
    logger.info("  - MCP 状态: http://localhost:8000/v1/tools/mcp/status")
    logger.info("=" * 60)

    # 检查 Prometheus 是否可用
    if PROMETHEUS_AVAILABLE:
        logger.info("Prometheus 监控已启用")
    else:
        logger.warning("Prometheus 监控未启用，请安装 prometheus-client")

    yield

    # 【生产实践】优雅关闭 — 释放所有资源
    logger.info("Agent API 正在关闭，释放所有资源...")

    # 1. 停止知识库文件监听服务
    try:
        from src.rag import stop_all_file_watchers
        logger.info("正在停止知识库文件监听服务...")
        stop_all_file_watchers()
        logger.info("知识库文件监听服务已停止")
    except Exception as e:
        logger.warning(f"停止文件监听服务失败: {e}")

    # 2. 关闭 MCP 管理器
    try:
        from src.tools import close_mcp_manager
        logger.info("正在关闭 MCP 管理器...")
        await close_mcp_manager()
        logger.info("MCP 管理器已关闭")
    except Exception as e:
        logger.warning(f"关闭 MCP 管理器失败: {e}")

    # 3. 关闭内存池
    try:
        from src.memory import close_async_pool
        logger.info("正在关闭记忆服务...")
        await close_async_pool()
        logger.info("记忆服务已关闭")
    except Exception as e:
        logger.warning(f"关闭记忆服务失败: {e}")

    logger.info("Agent API 已关闭")


# ============================================================
# 应用创建
# ============================================================
app = FastAPI(
    title="AI Agent API",
    description="基于 LangGraph 的多专家 Agent 系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务配置
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# 注册路由模块
# ============================================================
from src.api.routes import api_router
app.include_router(api_router)


# ============================================================
# 健康检查（保留在主文件中）
# ============================================================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# ============================================================
# LLM 统计接口
# ============================================================
@app.get("/llm/stats")
async def get_llm_stats():
    """获取 LLM 调用统计"""
    from src.llm.gateway import get_call_stats

    try:
        stats = get_call_stats()
        return stats
    except Exception as e:
        logger.error(f"获取LLM统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

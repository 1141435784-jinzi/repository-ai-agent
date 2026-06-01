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

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.metrics import PROMETHEUS_AVAILABLE

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
skill_manager = None
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
    3. 启动知识库文件监听服务
    4. 检查外部服务状态（如 Prometheus）
    """

    logger.info("=" * 60)
    logger.info("Agent API 正在启动...")
    logger.info("=" * 60)

    # 【生产实践】启动知识库文件监听服务
    # 当知识库文件发生变化时，自动触发增量更新
    logger.info("正在启动知识库文件监听服务...")
    try:
        from src.rag import start_all_file_watchers
        start_all_file_watchers()
        logger.info("知识库文件监听服务已启动")
    except Exception as e:
        logger.warning(f"文件监听服务启动失败：{e}")
        logger.warning("文件监听功能不可用，知识库更新需要手动触发")

    logger.info("Agent API 启动成功。")
    logger.info("=" * 60)
    logger.info("服务已就绪，可通过以下地址访问:")
    logger.info("  - API 文档：http://localhost:8000/docs")
    logger.info("  - 健康检查：http://localhost:8000/health")
    logger.info("  - MCP 状态：http://localhost:8000/v1/tools/mcp/status")
    logger.info("=" * 60)

    # 检查 Prometheus 是否可用
    if PROMETHEUS_AVAILABLE:
        logger.info("Prometheus 监控已启用")
    else:
        logger.warning("Prometheus 监控未启用，请安装 prometheus-client")

    # 【进阶方案】启动状态图预热任务（异步执行，不阻塞启动）
    # 预热机制可以在服务启动后后台创建 StateGraph 实例，提升首次请求响应速度
    logger.info("🚀 启动状态图预热任务...")
    
    asyncio.create_task(warm_up_graph())

    # ============================================================
    # 关键：lifespan 上下文管理器的分界点
    # ============================================================
    # yield 之前的代码：应用启动时执行（初始化阶段）
    # yield 本身：暂停 lifespan，将控制权交给 FastAPI 主程序
    # yield 之后的代码：应用关闭时执行（清理阶段）
    #
    # 【重要】不要删除或移动这行 yield！
    # - 它是 asynccontextmanager 的核心，区分启动和关闭逻辑
    # - 它确保资源在应用运行期间保持可用
    # - 它保证关闭时的清理代码一定会执行
    # ============================================================
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
        logger.warning(f"停止文件监听服务失败：{e}")

    # 3. 关闭内存池
    try:
        from src.memory import close_async_pool
        logger.info("正在关闭记忆服务...")
        await close_async_pool()
        logger.info("记忆服务已关闭")
    except Exception as e:
        logger.warning(f"关闭记忆服务失败：{e}")

    logger.info("Agent API 已关闭")


# ============================================================
# Agent 预热机制（简化版）
# ============================================================
async def warm_up_graph():
    """
    状态图预热任务 - 在服务启动后异步创建执行图实例
    
    【功能】：
    1. 延迟 1 秒启动，避免与其他初始化任务竞争资源
    2. 异步执行，不阻塞服务启动
    3. 创建 StateGraph 实例，触发所有依赖组件初始化
    
    【依赖链】
    StateGraph → 领域专家 → RAG 引擎 → Embedding 模型
              → LLM 模型
              → Memory Manager
              → Tool Manager
    """
    start_time = datetime.now()
    
    await asyncio.sleep(1)
    
    try:
        logger.info("  └─ 开始创建状态图实例...")
        
        from src.agents.workflow import get_async_graph
        await get_async_graph()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ 状态图创建完成！耗时：{duration:.2f} 秒")
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.warning(f"⚠️ 状态图创建失败：{str(e)}")
        logger.warning(f"   耗时：{duration:.2f} 秒，首次请求时将自动重试")

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
        logger.error(f"获取 LLM 统计失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ┌─────────────────────────────────────────┐
# │         Python 进程 (主进程)             │
# │  ┌───────────────────────────────────┐  │
# │  │   Uvicorn 服务器                   │  │
# │  │  ┌─────────────────────────────┐  │  │
# │  │  │  Event Loop (事件循环)       │  │  │
# │  │  │  ┌───────────────────────┐  │  │  │
# │  │  │  │  预热任务 (后台)        │  │  │  │
# │  │  │  ├───────────────────────┤  │  │  │
# │  │  │  │  请求 1 (用户对话)      │  │  │  │
# │  │  │  ├───────────────────────┤  │  │  │
# │  │  │  │  请求 2 (用户对话)      │  │  │  │
# │  │  │  ├───────────────────────┤  │  │  │
# │  │  │  │  请求 3 (新建会话)      │  │  │  │
# │  │  │  └───────────────────────┘  │  │  │
# │  │  └─────────────────────────────┘  │  │
# │  └───────────────────────────────────┘  │
# └─────────────────────────────────────────┘
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

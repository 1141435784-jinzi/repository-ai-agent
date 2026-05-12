"""
=== 记忆管理模块 — 基于 PostgreSQL 的生产级持久化记忆 ===

【知识点】记忆（Memory）是 Agent 实现多轮对话的关键
没有记忆的 Agent 每次对话都是"失忆"的，无法理解上下文

【企业实战 — 生产级 PostgreSQL AsyncPostgresSaver】
- AsyncPostgresSaver（异步）：用于 FastAPI 服务（agent.astream_events / ainvoke）
- 使用 psycopg3 异步连接池，支持高并发、连接复用、自动重连
- 本模块统一管理异步连接池的生命周期

【Windows 兼容说明】
Windows 上 uvicorn 单进程模式默认使用 ProactorEventLoop，
psycopg3 的异步模式不兼容 ProactorEventLoop。
解决方案：通过 run_server.py 使用 --reload 或 --workers 模式启动，
uvicorn 在这些模式下会自动切换为 SelectorEventLoop。

【长期：渐进升级计划】
# 阶段1：ChromaDB（当前）
#   优点：简单，成本低
#   监控：建立性能基线

# 阶段2：pgvector（如果需要）
#   触发：PostgreSQL已部署，希望一体化
#   迁移：相对平滑，数据可迁移
#   优势：一体化，减少技术栈
#   技术：PostgreSQL + pgvector扩展
#   适合：已使用PostgreSQL，希望简化架构

# 示例代码：
# CREATE EXTENSION vector;
# CREATE TABLE conversation_memories (
#     id UUID PRIMARY KEY,
#     thread_id TEXT,
#     content TEXT,
#     embedding vector(768),  # 与你的Embedding模型维度匹配
#     metadata JSONB,
#     created_at TIMESTAMP
# );


# 阶段3：独立向量DB（如果必要）
#   触发：明确的高性能需求
#   迁移：需要数据迁移和代码调整
#   优势：专业性能，最佳实践
#   选项：Pinecone、Weaviate、Qdrant、Milvus
#   适合：大规模生产环境，高性能要求

# 示例（Weaviate）：
import weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={"X-OpenAI-Api-Key": "..."}
)


"""

import logging

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import (
    POSTGRES_DSN,
    POSTGRES_POOL_MIN_SIZE,
    POSTGRES_POOL_MAX_SIZE,
)

logger = logging.getLogger(__name__)

# ============================================================
# 异步连接池（用于 FastAPI 服务）
# ============================================================
_async_pool: AsyncConnectionPool | None = None


async def get_async_pool() -> AsyncConnectionPool:
    """获取异步连接池单例

    Args:
        无参数

    Returns:
        AsyncConnectionPool: psycopg3 异步连接池实例
    """
    global _async_pool
    if _async_pool is None:
        logger.info(
            f"正在创建 PostgreSQL 异步连接池 "
            f"(min={POSTGRES_POOL_MIN_SIZE}, max={POSTGRES_POOL_MAX_SIZE})"
        )
        _async_pool = AsyncConnectionPool(
            conninfo=POSTGRES_DSN,
            min_size=POSTGRES_POOL_MIN_SIZE,
            max_size=POSTGRES_POOL_MAX_SIZE,
            open=False,
            kwargs={"autocommit": True},
        )
        await _async_pool.open()
    return _async_pool


async def get_async_checkpointer() -> AsyncPostgresSaver:
    """获取异步 Checkpointer 实例（用于 FastAPI 服务）

    Args:
        无参数

    Returns:
        AsyncPostgresSaver: 基于 PostgreSQL 的异步 Checkpointer 实例
    """
    pool = await get_async_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    logger.info("PostgreSQL 异步 Checkpointer 就绪 ✅")
    return checkpointer


# ============================================================
# 连接池生命周期管理
# ============================================================

async def close_async_pool() -> None:
    """关闭异步连接池（FastAPI 关闭时调用）

    Args:
        无参数

    Returns:
        None
    """
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None
        logger.info("PostgreSQL 异步连接池已关闭")

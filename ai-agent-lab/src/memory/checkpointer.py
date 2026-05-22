"""
=== 记忆管理模块 — 支持多平台的 Checkpointer 实现 ===

【跨平台策略】：
- Windows: 使用 MemorySaver（避免 psycopg3 的 ProactorEventLoop 问题，以及 SQLite 上下文管理器问题）
- Linux/Mac: 使用 AsyncPostgresSaver（生产级 PostgreSQL）

【Windows 兼容说明】
Windows 上 psycopg3 的异步模式与 ProactorEventLoop 不兼容，
AsyncSqliteSaver 需要上下文管理器，使用不便。
MemorySaver 是内存版本的 checkpointer，不需要数据库连接，
对于开发/测试环境来说足够使用（重启后状态会丢失，但功能正常）。

"""

import sys
import os
import logging

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    from langgraph.checkpoint.memory import MemorySaver
else:
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import (
    POSTGRES_DSN,
    POSTGRES_POOL_MIN_SIZE,
    POSTGRES_POOL_MAX_SIZE,
)

# ============================================================
# Checkpointer 实例缓存
# ============================================================
_checkpointer = None

# ============================================================
# Windows: MemorySaver 实现
# ============================================================
if IS_WINDOWS:

    async def get_async_checkpointer():
        """获取 Memory Checkpointer 实例（Windows 专用）

        Args:
            无参数

        Returns:
            MemorySaver: 基于内存的 Checkpointer 实例
        """
        global _checkpointer
        if _checkpointer is None:
            _checkpointer = MemorySaver()
            logger.info("Windows 模式: Memory Checkpointer 就绪 ✅")
        return _checkpointer

    async def close_async_pool() -> None:
        """关闭 Checkpointer（Windows 模式下无需操作）"""
        global _checkpointer
        if _checkpointer is not None:
            _checkpointer = None
            logger.info("Windows 模式: Memory Checkpointer 已关闭")

# ============================================================
# Linux/Mac: AsyncPostgresSaver 实现
# ============================================================
else:
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
        global _checkpointer
        if _checkpointer is None:
            pool = await get_async_pool()
            _checkpointer = AsyncPostgresSaver(pool)
            await _checkpointer.setup()
            logger.info("PostgreSQL 异步 Checkpointer 就绪 ✅")
        return _checkpointer

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

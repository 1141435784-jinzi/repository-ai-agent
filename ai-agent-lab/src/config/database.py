"""
=== 数据库配置 ===

【设计原则】：
1. 敏感信息从环境变量读取
2. 支持多数据库后端配置
3. 连接池参数可配置
4. 提供统一的连接字符串
"""

import os

# ============================================================
# PostgreSQL 配置（生产级 Checkpointer 持久化）
# ============================================================
# 连接参数从环境变量读取，敏感信息不硬编码
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "agent_lab")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

# PostgreSQL 连接字符串（DSN）
POSTGRES_DSN: str = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# 连接池配置
POSTGRES_POOL_MIN_SIZE: int = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2"))
POSTGRES_POOL_MAX_SIZE: int = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))

# ============================================================
# 向量数据库配置
# ============================================================
CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

# ============================================================
# 启动校验
# ============================================================
def validate_database_settings() -> None:
    """校验数据库配置是否有效"""
    if not POSTGRES_PASSWORD:
        raise RuntimeError(
            "未设置 POSTGRES_PASSWORD，PostgreSQL Checkpointer 无法连接，服务无法启动"
        )
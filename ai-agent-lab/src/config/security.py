"""
=== 安全配置 ===

【设计原则】：
1. 所有安全相关配置集中管理
2. 默认启用安全防护措施
3. 支持细粒度的安全策略配置
"""

import os

# ============================================================
# API 认证配置
# ============================================================
# 是否启用 API 认证
ENABLE_API_AUTH: bool = os.getenv("ENABLE_API_AUTH", "false").lower() == "true"

# API Key（当启用简单认证时使用）
API_KEY: str = os.getenv("API_KEY", "")

# JWT 配置
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

# ============================================================
# 输入输出安全配置
# ============================================================
# 是否启用输入校验
ENABLE_INPUT_SANITIZATION: bool = os.getenv("ENABLE_INPUT_SANITIZATION", "true").lower() == "true"

# 是否启用输出过滤
ENABLE_OUTPUT_SANITIZATION: bool = os.getenv("ENABLE_OUTPUT_SANITIZATION", "true").lower() == "true"

# 最大输入长度限制（字符数）
MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "4096"))

# 最大输出长度限制（字符数）
MAX_OUTPUT_LENGTH: int = int(os.getenv("MAX_OUTPUT_LENGTH", "8192"))

# ============================================================
# Prompt 注入防护
# ============================================================
ENABLE_PROMPT_INJECTION_DETECTION: bool = os.getenv("ENABLE_PROMPT_INJECTION_DETECTION", "true").lower() == "true"

# ============================================================
# CORS 配置
# ============================================================
# 允许的来源列表（逗号分隔）
CORS_ALLOWED_ORIGINS: list[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# 是否允许凭证
CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

# 允许的 HTTP 方法
CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

# 允许的请求头
CORS_ALLOW_HEADERS: list[str] = ["*"]

# ============================================================
# 速率限制配置
# ============================================================
# 是否启用速率限制
ENABLE_RATE_LIMITING: bool = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"

# 每分钟最大请求数
RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))

# ============================================================
# 启动校验
# ============================================================
def validate_security_settings() -> None:
    """校验安全配置是否有效"""
    if ENABLE_API_AUTH and not API_KEY:
        raise RuntimeError("启用了 API 认证但未设置 API_KEY")
    
    if ENABLE_API_AUTH and not JWT_SECRET_KEY:
        raise RuntimeError("启用了 API 认证但未设置 JWT_SECRET_KEY")
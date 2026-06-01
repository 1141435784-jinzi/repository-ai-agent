"""
=== 项目设置配置 ===

【设计原则】：
1. 敏感信息（API Key）从环境变量读取，绝不硬编码
2. 所有可调参数集中管理，方便统一修改
3. 使用类型注解，提高代码可读性和 IDE 提示
4. 启动时校验关键配置，尽早发现问题
5. 保持简单干净，避免过度设计

【配置源优先级】：
1. 环境变量（通过 .env 文件或系统环境变量）
2. 默认值（当环境变量未设置时）
"""

import os
import logging

logger = logging.getLogger(__name__)

# 项目根目录（agent-lab/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 离线模式配置（必须在所有 import 之前设置）
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_DOWNLOAD_TIMEOUT"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HOME"] = os.getenv("HF_HOME", r"G:\Users\jinzhenfeng\AppData\Local\Programs\Huggingface")

# ============================================================
# 加载 .env 文件
# ============================================================
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ============================================================
# LLM 配置 — 多 Provider 统一管理
# ============================================================

# Provider 4：阿里云百炼（默认主模型）
BAILIAN_API_KEY: str = os.getenv("LLM_BAILIAN_API_KEY", "")
BAILIAN_BASE_URL: str = os.getenv("LLM_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
BAILIAN_MODEL: str = os.getenv("LLM_BAILIAN_MODEL", "qwen-max")


# Provider 1：DeepSeek（备用模型）
DEEPSEEK_API_KEY: str = os.getenv("LLM_DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Provider 2：智谱 AI（备用模型）
LLM_ZHIPU_API_KEY: str = os.getenv("LLM_ZHIPU_API_KEY", "")
LLM_ZHIPU_BASE_URL: str = os.getenv("LLM_ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_ZHIPU_MODEL: str = os.getenv("LLM_ZHIPU_MODEL", "glm-4-flash")

# Provider 3：Ollama（本地模型）
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_NUM_GPU: int = int(os.getenv("OLLAMA_NUM_GPU", "1"))
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))

# 多模型配置
OLLAMA_MODELS: list[str] = os.getenv("OLLAMA_MODELS", "qwen2.5:3b,llama3.2:3b").split(",")
OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")
OLLAMA_ENABLE_SMART_ROUTING: bool = os.getenv("OLLAMA_ENABLE_SMART_ROUTING", "true").lower() == "true"

# 默认 Provider
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "")

# temperature 控制 LLM 输出的随机性
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

# ============================================================
# LangSmith 可观测性配置
# ============================================================
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "agent-lab")

# ============================================================
# Agent 配置
# ============================================================
MAX_ITERATIONS: int = 50

# ============================================================
# RAG 配置
# ============================================================
KNOWLEDGE_BASE_DIR: str = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_agent")
KNOWLEDGE_BASE_SIGHTS_DIR: str = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_sights")
KNOWLEDGE_BASE_TRANSPORT_DIR: str = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_transport")
KNOWLEDGE_BASE_PLAN_DIR: str = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_plan")
KNOWLEDGE_BASE_FOOD_DIR: str = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_food")
CHROMA_DB_DIR: str = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_COLLECTION_NAME: str = "agent_knowledge"

# 知识库配置字典（统一管理，便于扩展）
KNOWLEDGE_BASES: dict[str, str] = {
    "agent_tech": KNOWLEDGE_BASE_DIR,
    "plan": KNOWLEDGE_BASE_PLAN_DIR,
    "sights": KNOWLEDGE_BASE_SIGHTS_DIR,
    "food": KNOWLEDGE_BASE_FOOD_DIR,
    "transport": KNOWLEDGE_BASE_TRANSPORT_DIR,
}

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")

# 文本分块参数
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 80

# 检索参数
TOP_K: int = 5
RERANK_TOP_N: int = 3
VECTOR_WEIGHT: float = 0.6
BM25_WEIGHT: float = 0.4
SIMILARITY_THRESHOLD: float = 0.3

# ============================================================
# 短期记忆配置（Short-term Memory）
# ============================================================
MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
MEMORY_MIN_WINDOW_SIZE: int = int(os.getenv("MEMORY_MIN_WINDOW_SIZE", "3"))
MEMORY_SEMANTIC_TOP_K: int = int(os.getenv("MEMORY_SEMANTIC_TOP_K", "3"))
MEMORY_MAX_CONTEXT_TOKENS: int = int(os.getenv("MEMORY_MAX_CONTEXT_TOKENS", "8192"))
MEMORY_RESERVED_TOKENS: int = int(os.getenv("MEMORY_RESERVED_TOKENS", "1000"))
MEMORY_MAX_MEMORY_CONTEXT_TOKENS: int = int(os.getenv("MEMORY_MAX_MEMORY_CONTEXT_TOKENS", "1500"))
MEMORY_COMPACT_CONTEXT_TOKENS: int = int(os.getenv("MEMORY_COMPACT_CONTEXT_TOKENS", "500"))

# ============================================================
# 长期记忆配置（Long-term Memory）
# ============================================================
LONG_MEMORY_SEMANTIC_TOP_K: int = int(os.getenv("LONG_MEMORY_SEMANTIC_TOP_K", "10"))
LONG_MEMORY_RERANK_TOP_N: int = int(os.getenv("LONG_MEMORY_RERANK_TOP_N", "3"))
LONG_MEMORY_PROFILE_DIR: str = os.getenv("LONG_MEMORY_PROFILE_DIR", os.path.join(PROJECT_ROOT, "user_profiles"))

# ============================================================
# 启动校验
# ============================================================
def validate_settings() -> None:
    """校验配置是否有效"""
    if not LLM_ZHIPU_API_KEY and not DEEPSEEK_API_KEY:
        logger.warning(
            "未设置任何远程 LLM API Key（LLM_ZHIPU_API_KEY / LLM_DEEPSEEK_API_KEY），"
            "将仅使用本地 Ollama 模型"
        )
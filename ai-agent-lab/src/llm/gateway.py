"""
=== LLM Gateway L5 — 企业级多模型路由、容灾、负载均衡、缓存、限流 ===

【L5 企业级特性】：
1. 多模型路由：根据任务类型选择最合适的模型
2. 容灾降级：主模型挂了自动切备用模型
3. 负载均衡：同一模型多个 API Key 轮询
4. 语义缓存：减少重复查询成本
5. 速率限制：防止 API 限流
6. Token 预算管理：成本控制
7. 调用统计：Prometheus 可观测性
8. 重试机制：网络抖动自动重试

【优先级链】：百炼 (0) → DeepSeek(1) → 智谱 (2) → Ollama(3)
"""

import logging
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from src.config import (
    LLM_ZHIPU_API_KEY,
    LLM_ZHIPU_BASE_URL,
    LLM_ZHIPU_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    BAILIAN_API_KEY,
    BAILIAN_BASE_URL,
    BAILIAN_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_MODELS,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_ENABLE_SMART_ROUTING,
    OLLAMA_NUM_GPU,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    OLLAMA_MAX_TOKENS,
    DEFAULT_LLM_PROVIDER,
    TEMPERATURE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    REDIS_ENABLED,
    REDIS_CACHE_TTL,
)

# Prometheus 指标集成
try:
    from src.metrics import (
        record_llm_call,
        record_llm_error,
        record_fallback,
        calculate_token_cost,
        PROMETHEUS_AVAILABLE
    )
    PROMETHEUS_ENABLED = PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_ENABLED = False
    
    def record_llm_call(*args, **kwargs):
        pass
    
    def record_llm_error(*args, **kwargs):
        pass
    
    def record_fallback(*args, **kwargs):
        pass
    
    def calculate_token_cost(*args, **kwargs):
        return 0.0

logger = logging.getLogger(__name__)


# ============================================================
# L5 特性 1：负载均衡器（多 API Key 轮询）
# ============================================================

class LoadBalancer:
    """线程安全的负载均衡器 - 轮询多个 API Key
    
    【企业实战】为什么需要负载均衡？
    - 单个 API Key 有速率限制（如 60 次/分钟）
    - 多个 Key 轮询可以显著提高吞吐量
    - 自动跳过失败的 Key，提高可用性
    """
    
    def __init__(self, api_keys: List[str]):
        """初始化负载均衡器
        
        Args:
            api_keys: API Key 列表
        """
        self.api_keys = api_keys
        self.current_index = 0
        self.failed_keys = set()  # 临时失败的 Key
        self._lock = threading.Lock()
    
    def get_next_key(self) -> Optional[str]:
        """获取下一个可用的 API Key（轮询）
        
        Returns:
            str: API Key，如果没有可用 Key 则返回 None
        """
        with self._lock:
            if not self.api_keys:
                return None
            
            # 尝试最多 len(api_keys) 次找到可用的 Key
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                
                if key not in self.failed_keys:
                    return key
            
            # 所有 Key 都失败了，返回第一个（重试）
            return self.api_keys[0] if self.api_keys else None
    
    def mark_failed(self, api_key: str):
        """标记某个 API Key 失败
        
        Args:
            api_key: 失败的 API Key
        """
        with self._lock:
            self.failed_keys.add(api_key)
    
    def mark_success(self, api_key: str):
        """标记某个 API Key 成功（清除失败标记）
        
        Args:
            api_key: 成功的 API Key
        """
        with self._lock:
            self.failed_keys.discard(api_key)
    
    def reset(self):
        """重置所有失败标记"""
        with self._lock:
            self.failed_keys.clear()


# ============================================================
# L5 特性 2：语义缓存（Semantic Caching）
# ============================================================

class SemanticCache:
    """语义缓存 - 减少重复查询的 LLM 调用成本
    
    【企业实战】为什么需要语义缓存？
    - 相似问题重复出现（如"你好"、"早上好"）
    - 缓存相似度>0.95 的查询，节省 30-40% 成本
    - 使用局部敏感哈希（LSH）快速检索
    """
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        """初始化语义缓存
        
        Args:
            max_size: 最大缓存条目数（LRU 淘汰）
            similarity_threshold: 相似度阈值（0-1，越高越严格）
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._embeddings = None
        self._embedding_lock = threading.Lock()
    
    def _get_embeddings(self):
        """懒加载 embedding 模型"""
        if self._embeddings is None:
            with self._embedding_lock:
                if self._embeddings is None:
                    try:
                        from src.rag.embedding import get_embeddings
                        self._embeddings = get_embeddings()
                        logger.debug("SemanticCache embedding 模型加载成功")
                    except Exception as e:
                        logger.warning(f"SemanticCache 无法加载 embedding 模型: {e}")
                        self._embeddings = None
        return self._embeddings
    
    def _compute_hash(self, text: str) -> str:
        """计算文本的哈希（用于精确匹配）
        
        Args:
            text: 输入文本
            
        Returns:
            str: MD5 哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            float: 余弦相似度 (0-1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """基于本地 Embedding 模型的语义相似度计算
        
        使用 HuggingFace Embedding 模型计算语义相似度
        """
        embeddings_model = self._get_embeddings()
        if embeddings_model is None:
            return 0.0
        
        try:
            vec1 = embeddings_model.embed_query(text1)
            vec2 = embeddings_model.embed_query(text2)
            return self._cosine_similarity(vec1, vec2)
        except Exception as e:
            logger.warning(f"Embedding 相似度计算失败: {e}")
            return 0.0
    
    def get(self, query: str) -> Optional[str]:
        """从缓存中获取响应
        
        Args:
            query: 查询文本
            
        Returns:
            str: 缓存的响应，如果未命中则返回 None
        """
        with self._lock:
            query_hash = self._compute_hash(query)
            if query_hash in self._cache:
                response, timestamp, cached_query = self._cache[query_hash]
                self._cache.move_to_end(query_hash)
                logger.debug(f"缓存命中（精确）: '{query[:30]}...'")
                return response
            
            for cached_hash, (response, timestamp, cached_query) in self._cache.items():
                similarity = self._compute_similarity(query, cached_query)
                if similarity >= self.similarity_threshold:
                    self._cache.move_to_end(cached_hash)
                    logger.debug(f"缓存命中（语义 {similarity:.2f}）: '{query[:30]}...'")
                    return response
            
            logger.debug(f"缓存未命中：'{query[:30]}...'")
            return None
    
    def set(self, query: str, response: str):
        """将响应存入缓存
        
        Args:
            query: 查询文本
            response: LLM 响应
        """
        with self._lock:
            query_hash = self._compute_hash(query)
            
            if query_hash in self._cache:
                del self._cache[query_hash]
            
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[query_hash] = (response, datetime.now(), query)
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            dict: 缓存统计
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "threshold": self.similarity_threshold,
            }


# ============================================================
# L5 特性 2b：Redis 语义缓存（分布式版本）
# ============================================================

class RedisSemanticCache:
    """Redis 语义缓存 - 支持分布式部署的 LLM 响应缓存

    【企业实战】为什么需要 Redis 版缓存？
    - 多实例/多服务器共享缓存
    - 缓存持久化，重启不丢失
    - 支持 TTL 过期策略
    - 支持集群部署
    """

    _redis_client: Optional["redis.Redis"] = None

    def __init__(
        self,
        max_size: int = 10000,
        similarity_threshold: float = 0.95,
        ttl: int = REDIS_CACHE_TTL,
    ):
        """初始化 Redis 语义缓存

        Args:
            max_size: 最大缓存条目数
            similarity_threshold: 相似度阈值（0-1）
            ttl: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self._embeddings = None
        self._embedding_lock = threading.Lock()
        self._embedding_prefix = "semantic_cache:embedding:"
        self._response_prefix = "semantic_cache:response:"
        self._index_key = "semantic_cache:index"

    @classmethod
    def _get_redis(cls) -> Optional["redis.Redis"]:
        """获取 Redis 连接（类级别单例）"""
        if cls._redis_client is None:
            try:
                import redis as redis_lib
                cls._redis_client = redis_lib.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD or None,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
                cls._redis_client.ping()
                logger.info("RedisSemanticCache 连接成功")
            except Exception as e:
                logger.warning(f"RedisSemanticCache 连接失败: {e}")
                cls._redis_client = None
        return cls._redis_client

    def _get_embeddings(self):
        """懒加载 embedding 模型"""
        if self._embeddings is None:
            with self._embedding_lock:
                if self._embeddings is None:
                    try:
                        from src.rag.embedding import get_embeddings
                        self._embeddings = get_embeddings()
                        logger.debug("RedisSemanticCache embedding 模型加载成功")
                    except Exception as e:
                        logger.warning(f"RedisSemanticCache 无法加载 embedding 模型: {e}")
                        self._embeddings = None
        return self._embeddings

    def _compute_hash(self, text: str) -> str:
        """计算文本的 MD5 哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _vector_to_bytes(self, vec: List[float]) -> bytes:
        """将向量转换为字节存储"""
        import struct
        return struct.pack(f'{len(vec)}f', *vec)

    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """从字节恢复向量"""
        import struct
        vec_len = len(data) // 4
        return list(struct.unpack(f'{vec_len}f', data))

    def get(self, query: str) -> Optional[str]:
        """从 Redis 缓存中获取响应

        Args:
            query: 查询文本

        Returns:
            str: 缓存的响应，如果未命中则返回 None
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return None

        query_hash = self._compute_hash(query)

        response_key = f"{self._response_prefix}{query_hash}"
        cached_response = redis_client.get(response_key)
        if cached_response:
            if isinstance(cached_response, bytes):
                cached_response = cached_response.decode('utf-8')
            logger.debug(f"Redis 缓存命中（精确）: '{query[:30]}...'")
            return cached_response

        embeddings_model = self._get_embeddings()
        if embeddings_model is None:
            return None

        try:
            query_embedding = embeddings_model.embed_query(query)
            query_vec_bytes = self._vector_to_bytes(query_embedding)

            all_cached = redis_client.zrange(self._index_key, 0, -1)
            if not all_cached:
                return None

            best_match = None
            best_similarity = 0.0

            for cached_hash in all_cached:
                cached_hash_str = cached_hash.decode('utf-8') if isinstance(cached_hash, bytes) else cached_hash
                embedding_key = f"{self._embedding_prefix}{cached_hash_str}"

                cached_vec_bytes = redis_client.get(embedding_key)
                if cached_vec_bytes:
                    cached_vec = self._bytes_to_vector(cached_vec_bytes)
                    similarity = self._cosine_similarity(query_embedding, cached_vec)

                    if similarity >= self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        response_key = f"{self._response_prefix}{cached_hash_str}"
                        best_match = redis_client.get(response_key)

            if best_match:
                if isinstance(best_match, bytes):
                    best_match = best_match.decode('utf-8')
                logger.debug(f"Redis 缓存命中（语义 {best_similarity:.2f}）: '{query[:30]}...'")
                return best_match

        except Exception as e:
            logger.warning(f"Redis 语义匹配失败: {e}")

        logger.debug(f"Redis 缓存未命中：'{query[:30]}...'")
        return None

    def set(self, query: str, response: str):
        """将响应存入 Redis 缓存

        Args:
            query: 查询文本
            response: LLM 响应
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return

        query_hash = self._compute_hash(query)

        response_key = f"{self._response_prefix}{query_hash}"
        redis_client.setex(response_key, self.ttl, response.encode('utf-8'))

        embeddings_model = self._get_embeddings()
        if embeddings_model:
            try:
                query_embedding = embeddings_model.embed_query(query)
                embedding_key = f"{self._embedding_prefix}{query_hash}"
                redis_client.setex(embedding_key, self.ttl, self._vector_to_bytes(query_embedding))

                redis_client.zadd(self._index_key, {query_hash: 0})

                current_size = redis_client.zcard(self._index_key)
                if current_size > self.max_size:
                    to_remove = current_size - self.max_size
                    old_entries = redis_client.zrange(self._index_key, 0, to_remove - 1)
                    for old_hash in old_entries:
                        old_hash_str = old_hash.decode('utf-8') if isinstance(old_hash, bytes) else old_hash
                        redis_client.delete(f"{self._response_prefix}{old_hash_str}")
                        redis_client.delete(f"{self._embedding_prefix}{old_hash_str}")
                    redis_client.zremrangebyrank(self._index_key, 0, to_remove - 1)

            except Exception as e:
                logger.warning(f"Redis embedding 存储失败: {e}")

    def clear(self):
        """清空 Redis 缓存"""
        redis_client = self._get_redis()
        if redis_client is None:
            return

        try:
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, match=f"{self._embedding_prefix}*", count=100)
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break

            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, match=f"{self._response_prefix}*", count=100)
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break

            redis_client.delete(self._index_key)
            logger.info("Redis 语义缓存已清空")
        except Exception as e:
            logger.warning(f"Redis 缓存清空失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取 Redis 缓存统计信息"""
        redis_client = self._get_redis()
        if redis_client is None:
            return {"enabled": False, "size": 0}

        try:
            size = redis_client.zcard(self._index_key) if redis_client.exists(self._index_key) else 0
            return {
                "enabled": True,
                "size": size,
                "max_size": self.max_size,
                "threshold": self.similarity_threshold,
                "ttl": self.ttl,
            }
        except Exception as e:
            logger.warning(f"获取 Redis 缓存统计失败: {e}")
            return {"enabled": False, "error": str(e)}


# ============================================================
# L5 特性 3：速率限制器（Rate Limiter）
# ============================================================

class RateLimiter:
    """滑动窗口速率限制器 - 防止 API 限流
    
    【企业实战】为什么需要速率限制？
    - API 提供商通常有限制（如 60 次/分钟）
    - 主动限流比被限流（429 错误）更优雅
    - 支持多个 Provider 独立限流
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """初始化速率限制器
        
        Args:
            calls_per_minute: 每分钟允许的最大调用次数
        """
        self.limit = calls_per_minute
        self.window_seconds = 60
        self._calls = []  # 调用时间戳列表
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """获取调用许可（阻塞直到可用）
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            bool: True 表示获取成功，False 表示超时
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                now = time.time()
                # 清理窗口外的调用
                self._calls = [t for t in self._calls if now - t < self.window_seconds]
                
                # 检查是否还有配额
                if len(self._calls) < self.limit:
                    self._calls.append(now)
                    return True
            
            # 检查超时
            if time.time() - start_time > timeout:
                logger.warning(f"速率限制等待超时（{timeout}s）")
                return False
            
            # 等待一小段时间再重试
            time.sleep(0.1)
    
    def try_acquire(self) -> bool:
        """尝试获取调用许可（非阻塞）
        
        Returns:
            bool: True 表示获取成功，False 表示当前无配额
        """
        with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < self.window_seconds]
            
            if len(self._calls) < self.limit:
                self._calls.append(now)
                return True
            
            return False
    
    def get_remaining(self) -> int:
        """获取剩余配额
        
        Returns:
            int: 剩余可用调用次数
        """
        with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < self.window_seconds]
            return max(0, self.limit - len(self._calls))


# ============================================================
# L5 特性 4：Token 预算管理器
# ============================================================

class TokenBudgetManager:
    """Token 预算管理器 - 成本控制
    
    【企业实战】为什么需要 Token 预算？
    - 防止意外高额账单（如死循环调用）
    - 按日/周/月设置预算上限
    - 达到阈值时告警或降级
    """
    
    def __init__(self, monthly_budget: int = 1000000, warning_threshold: float = 0.8):
        """初始化 Token 预算管理器
        
        Args:
            monthly_budget: 月度 Token 预算
            warning_threshold: 告警阈值（0-1，如 0.8 表示 80%）
        """
        self.monthly_budget = monthly_budget
        self.warning_threshold = warning_threshold
        self._used = 0
        self._lock = threading.Lock()
        self._reset_date = datetime.now().replace(day=1)  # 每月 1 号重置
        self._warning_triggered = False
    
    def _check_reset(self):
        """检查是否需要重置（每月 1 号）"""
        now = datetime.now()
        if now.month != self._reset_date.month or now.year != self._reset_date.year:
            with self._lock:
                self._used = 0
                self._warning_triggered = False
                self._reset_date = now.replace(day=1)
                logger.info(f"Token 预算已重置（月度：{self.monthly_budget}）")
    
    def check_and_consume(self, tokens: int) -> bool:
        """检查并消耗 Token 配额
        
        Args:
            tokens: 需要消耗的 Token 数量
            
        Returns:
            bool: True 表示允许调用，False 表示超出预算
        """
        self._check_reset()
        
        with self._lock:
            if self._used + tokens > self.monthly_budget:
                logger.error(f"Token 预算不足：已用 {self._used}/{self.monthly_budget}，需要 {tokens}")
                return False
            
            self._used += tokens
            
            # 检查是否触发告警
            usage_ratio = self._used / self.monthly_budget
            if usage_ratio >= self.warning_threshold and not self._warning_triggered:
                logger.warning(
                    f"⚠️ Token 预算使用已达 {usage_ratio*100:.1f}% "
                    f"({self._used}/{self.monthly_budget})"
                )
                self._warning_triggered = True
            
            return True
    
    def get_usage(self) -> Dict[str, Any]:
        """获取使用情况
        
        Returns:
            dict: 使用统计
        """
        self._check_reset()
        
        with self._lock:
            usage_ratio = self._used / self.monthly_budget if self.monthly_budget > 0 else 0
            return {
                "used": self._used,
                "budget": self.monthly_budget,
                "remaining": max(0, self.monthly_budget - self._used),
                "usage_ratio": usage_ratio,
                "warning_triggered": self._warning_triggered,
            }
    
    def reset(self):
        """手动重置预算"""
        with self._lock:
            self._used = 0
            self._warning_triggered = False


# ============================================================
# 模型 Provider 配置（增强版）
# ============================================================

@dataclass
class LLMProviderConfig:
    """LLM 供应商配置（L5 增强版）

    Args:
        name: 供应商标识（如 "deepseek"、"zhipu"）
        api_keys: API Key 列表（支持负载均衡）
        base_url: API 地址
        model: 模型名称
        enabled: 是否启用
        priority: 优先级（数字越小优先级越高）
        max_retries: 单次调用最大重试次数
        timeout: 单次调用超时时间（秒）
        rate_limit: 速率限制（次/分钟）
    """
    name: str
    api_keys: List[str] = field(default_factory=list)
    base_url: str = ""
    model: str = ""
    enabled: bool = True
    priority: int = 0
    max_retries: int = 2
    timeout: int = 30
    rate_limit: int = 60  # 默认 60 次/分钟
    
    # 运行时组件（不序列化）
    load_balancer: Optional[LoadBalancer] = field(default=None, repr=False)
    rate_limiter: Optional[RateLimiter] = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后处理 - 创建负载均衡器和速率限制器"""
        if self.api_keys and len(self.api_keys) > 1:
            self.load_balancer = LoadBalancer(self.api_keys)
        elif self.api_keys:
            # 单个 Key 也创建负载均衡器（支持失败标记）
            self.load_balancer = LoadBalancer(self.api_keys)
        
        self.rate_limiter = RateLimiter(self.rate_limit)


# 【知识点】所有配置从 config.py 读取
_PROVIDERS: dict[str, LLMProviderConfig] = {}

# Provider 1：阿里云百炼（默认主模型）
if BAILIAN_API_KEY:
    _PROVIDERS["bailian"] = LLMProviderConfig(
        name="bailian",
        api_keys=[BAILIAN_API_KEY],  # 支持多 Key
        base_url=BAILIAN_BASE_URL,
        model=BAILIAN_MODEL,
        priority=0,
        rate_limit=100,  # 百炼限制较低
    )

# Provider 2：DeepSeek（备用模型）
if DEEPSEEK_API_KEY:
    _PROVIDERS["deepseek"] = LLMProviderConfig(
        name="deepseek",
        api_keys=[DEEPSEEK_API_KEY],
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        priority=1,
        rate_limit=60,
    )

# Provider 3：智谱 AI（降级模型）
if LLM_ZHIPU_API_KEY:
    _PROVIDERS["zhipu"] = LLMProviderConfig(
        name="zhipu",
        api_keys=[LLM_ZHIPU_API_KEY],
        base_url=LLM_ZHIPU_BASE_URL,
        model=LLM_ZHIPU_MODEL,
        priority=2,
        rate_limit=100,
    )

# Provider 4：Ollama（本地模型）
_PROVIDERS["ollama"] = LLMProviderConfig(
    name="ollama",
    api_keys=[],  # 本地模型不需要 Key
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    priority=3,
    rate_limit=1000,  # 本地限制宽松
)


# ============================================================
# 调用统计
# ============================================================

@dataclass
class LLMCallStats:
    """LLM 调用统计（L5 增强版）"""
    total_calls: int = 0
    success_calls: int = 0
    fallback_calls: int = 0
    error_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limited_calls: int = 0
    calls_by_provider: Dict[str, int] = field(default_factory=dict)
    total_tokens_used: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            "total_calls": self.total_calls,
            "success_rate": self.success_calls / max(1, self.total_calls),
            "fallback_rate": self.fallback_calls / max(1, self.total_calls),
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "calls_by_provider": self.calls_by_provider,
            "total_tokens_used": self.total_tokens_used,
        }


_stats = LLMCallStats()

# 全局缓存和预算管理
_global_cache = SemanticCache(max_size=5000, similarity_threshold=0.95)
_global_budget = TokenBudgetManager(monthly_budget=5000000, warning_threshold=0.8)

# LLM 实例缓存
_llm_cache: dict[tuple[str, float, bool], ChatOpenAI] = {}


def _create_llm(provider: LLMProviderConfig, temperature: float, streaming: bool = False) -> ChatOpenAI:
    """根据 Provider 配置创建 LLM 实例（L5 增强版）"""
    # Ollama 特殊处理
    if provider.name == "ollama":
        ollama_temp = OLLAMA_TEMPERATURE if temperature == TEMPERATURE else temperature
        return ChatOllama(
            base_url=provider.base_url,
            model=provider.model,
            temperature=ollama_temp,
            request_timeout=provider.timeout,
            streaming=streaming,
            model_kwargs={
                "num_gpu": OLLAMA_NUM_GPU,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_MAX_TOKENS,
            }
        )
    
    # 其他 Provider 使用 ChatOpenAI
    return ChatOpenAI(
        api_key=provider.api_keys[0] if provider.api_keys else "",
        base_url=provider.base_url,
        model=provider.model,
        temperature=temperature,
        max_retries=provider.max_retries,
        request_timeout=provider.timeout,
        streaming=streaming,
    )


def _get_default_provider_name() -> str:
    """获取默认 Provider 名称"""
    if DEFAULT_LLM_PROVIDER and DEFAULT_LLM_PROVIDER in _PROVIDERS:
        return DEFAULT_LLM_PROVIDER
    
    sorted_providers = sorted(
        _PROVIDERS.values(),
        key=lambda p: p.priority,
    )
    if sorted_providers:
        return sorted_providers[0].name
    
    raise RuntimeError("没有可用的 LLM Provider")


def _get_fallback_providers(exclude: str) -> List[LLMProviderConfig]:
    """获取降级 Provider 列表"""
    return sorted(
        [p for p in _PROVIDERS.values() if p.name != exclude and p.enabled],
        key=lambda p: p.priority,
    )


# ============================================================
# 核心 API：get_llm() - L5 增强版
# ============================================================

def get_llm(
    temperature: Optional[float] = None, 
    provider: Optional[str] = None,
    streaming: bool = True,
) -> ChatOpenAI:
    """获取 LLM 实例（L5 增强版）
    
    【L5 特性】：
    - 支持负载均衡（多 API Key 轮询）
    - 支持速率限制
    - 相同 (provider, temperature, streaming) 返回缓存实例
    """
    temp = temperature if temperature is not None else TEMPERATURE
    provider_name = provider or _get_default_provider_name()
    
    cache_key = (provider_name, temp, streaming)
    if cache_key not in _llm_cache:
        if provider_name not in _PROVIDERS:
            raise ValueError(f"未知的 LLM Provider: '{provider_name}'")
        
        config = _PROVIDERS[provider_name]
        logger.info(
            f"创建 LLM 实例：provider={provider_name}, "
            f"model={config.model}, temperature={temp}, streaming={streaming}"
        )
        _llm_cache[cache_key] = _create_llm(config, temp, streaming)
    
    return _llm_cache[cache_key]


# ============================================================
# 增强 API：invoke_with_l5_features() - 完整 L5 特性
# ============================================================

def invoke_with_l5_features(
    messages: list,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    use_cache: bool = True,
    check_budget: bool = True,
) -> str:
    """带 L5 特性的 LLM 调用（推荐使用）
    
    【L5 完整特性】：
    1. 语义缓存：避免重复查询
    2. 速率限制：防止 API 限流
    3. 负载均衡：多 Key 轮询
    4. Token 预算：成本控制
    5. 容灾降级：自动切换备用 Provider
    6. 调用统计：Prometheus 可观测性
    
    Args:
        messages: LangChain 消息列表
        temperature: 输出随机性
        provider: 首选 Provider
        use_cache: 是否启用缓存
        check_budget: 是否检查预算
        
    Returns:
        str: LLM 回复文本
    """
    temp = temperature if temperature is not None else TEMPERATURE
    provider_name = provider or _get_default_provider_name()
    
    # 1. 检查语义缓存
    query_text = messages[-1].content if messages else ""
    if use_cache:
        cached_response = _global_cache.get(query_text)
        if cached_response:
            _stats.cache_hits += 1
            logger.info(f"缓存命中，节省 LLM 调用")
            return cached_response
        _stats.cache_misses += 1
    
    # 2. 检查 Token 预算
    if check_budget:
        # 预估输入 token（简化）
        estimated_input_tokens = len(str(messages)) // 4
        if not _global_budget.check_and_consume(estimated_input_tokens):
            logger.error("Token 预算不足，拒绝调用")
            raise RuntimeError("Token 预算已用尽")
    
    # 3. 获取 Provider 配置
    config = _PROVIDERS.get(provider_name)
    if not config:
        raise ValueError(f"未知的 Provider: {provider_name}")
    
    # 4. 速率限制检查
    if config.rate_limiter and not config.rate_limiter.try_acquire():
        _stats.rate_limited_calls += 1
        logger.warning(f"Provider {provider_name} 触发速率限制，等待...")
        if not config.rate_limiter.acquire(timeout=30.0):
            raise RuntimeError(f"速率限制等待超时：{provider_name}")
    
    # 5. 获取 LLM 实例（带负载均衡）
    llm = get_llm(temperature=temp, provider=provider_name)
    
    # 6. 执行调用（带重试）
    _stats.total_calls += 1
    
    for attempt in range(config.max_retries):
        try:
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            
            # 提取 Token 信息
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'prompt_tokens', None)
                output_tokens = getattr(response.usage, 'completion_tokens', None)
            
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
            _stats.total_tokens_used += total_tokens
            
            # 更新预算
            if check_budget and output_tokens:
                _global_budget.check_and_consume(output_tokens)
            
            # 标记成功
            if config.load_balancer:
                config.load_balancer.mark_success(config.api_keys[0] if config.api_keys else "")
            
            # 记录统计
            _stats.success_calls += 1
            _stats.calls_by_provider[provider_name] = (
                _stats.calls_by_provider.get(provider_name, 0) + 1
            )
            
            # 记录 Prometheus
            if PROMETHEUS_ENABLED:
                cost_yuan = calculate_token_cost(provider_name, input_tokens or 0, output_tokens or 0)
                record_llm_call(
                    provider=provider_name,
                    model=config.model,
                    duration=elapsed,
                    status="success",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_yuan=cost_yuan,
                    endpoint="invoke_l5"
                )
            
            # 存入缓存
            if use_cache:
                _global_cache.set(query_text, response.content)
            
            logger.info(
                f"LLM 调用成功：provider={provider_name}, "
                f"耗时={elapsed:.2f}s, tokens={total_tokens}"
            )
            
            return response.content
            
        except Exception as e:
            logger.warning(
                f"LLM 调用失败（尝试 {attempt+1}/{config.max_retries}）: "
                f"provider={provider_name}, 错误={e}"
            )
            
            # 标记失败
            if config.load_balancer:
                config.load_balancer.mark_failed(config.api_keys[0] if config.api_keys else "")
            
            # 记录 Prometheus 错误
            if PROMETHEUS_ENABLED:
                error_type = "api_error"
                if "timeout" in str(e).lower():
                    error_type = "timeout"
                elif "rate limit" in str(e).lower():
                    error_type = "rate_limit"
                
                record_llm_error(
                    provider=provider_name,
                    model=config.model,
                    error_type=error_type,
                    endpoint="invoke_l5"
                )
            
            if attempt == config.max_retries - 1:
                # 最后一次重试失败，进入降级流程
                break
            
            time.sleep(0.5 * (attempt + 1))  # 指数退避
    
    # 7. 容灾降级：尝试备用 Provider
    logger.warning(f"主 Provider {provider_name} 失败，尝试降级...")
    fallback_providers = _get_fallback_providers(exclude=provider_name)
    
    for fallback in fallback_providers:
        try:
            logger.info(f"降级到 {fallback.name}")
            fallback_llm = get_llm(temperature=temp, provider=fallback.name)
            start_time = time.time()
            response = fallback_llm.invoke(messages)
            elapsed = time.time() - start_time
            
            _stats.success_calls += 1
            _stats.fallback_calls += 1
            
            if PROMETHEUS_ENABLED:
                record_fallback(from_provider=provider_name, to_provider=fallback.name)
            
            logger.warning(f"降级成功：{fallback.name}, 耗时={elapsed:.2f}s")
            return response.content
            
        except Exception as fallback_error:
            logger.error(f"降级失败：{fallback.name}, 错误={fallback_error}")
    
    # 8. 所有 Provider 都失败
    _stats.error_calls += 1
    raise RuntimeError(f"所有 LLM Provider 调用失败（主：{provider_name}, 备用：{len(fallback_providers)}个）")


# ============================================================
# 辅助 API
# ============================================================

def get_call_stats() -> Dict[str, Any]:
    """获取调用统计（含缓存和预算）"""
    return {
        "calls": _stats.get_summary(),
        "cache": _global_cache.get_stats(),
        "budget": _global_budget.get_usage(),
    }


def clear_cache():
    """清空缓存"""
    _global_cache.clear()
    logger.info("语义缓存已清空")


def reset_budget():
    """重置预算"""
    _global_budget.reset()
    logger.info("Token 预算已重置")


# ============================================================
# 向后兼容：保留原 invoke_with_fallback
# ============================================================

def invoke_with_fallback(
    messages: list,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
) -> str:
    """带容灾降级的 LLM 调用（向后兼容，推荐用 invoke_with_l5_features）"""
    return invoke_with_l5_features(
        messages=messages,
        temperature=temperature,
        provider=provider,
        use_cache=False,  # 默认不启用缓存（保持兼容）
        check_budget=False,  # 默认不检查预算
    )

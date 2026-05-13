"""
=== LLM Gateway — 多模型路由、容灾降级、调用统计 ===

【知识点】为什么需要 LLM Gateway？

企业级 Agent 不会直接调用某个模型的 API，而是通过一个统一的 Gateway 层：

1. 多模型路由：根据任务类型选择最合适的模型
   - 日常对话用便宜快速的模型（glm-4-flash）
   - 复杂推理用强模型（deepseek-chat）
   - 摘要/评估用低 temperature 的稳定模型

2. 容灾降级：主模型挂了自动切备用模型
   - DeepSeek API 超时 → 自动切智谱 API
   - 所有模型都挂了 → 返回友好的降级提示

3. 负载均衡：同一模型多个 API Key 轮询（本模块预留接口）

4. 调用统计：记录每次调用的 provider、model、耗时
   - 用于成本核算、性能监控、异常告警

5. 重试机制：网络抖动时自动重试，对上层透明

【现实例子 — 电商客服系统】
- 用户问"你好" → 路由到 deepseek-chat（默认主模型）
- DeepSeek API 突然 503 → 自动降级到智谱 API，用户无感知
- 月底看报表：本月调用 10 万次，DeepSeek 8 万次 + 智谱 2 万次，总成本 ¥xxx
"""

import logging
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from src.config import (
    LLM_ZHIPU_API_KEY,
    LLM_ZHIPU_BASE_URL,
    LLM_ZHIPU_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
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
)

# Prometheus 指标集成
try:
    from prometheus_metrics import (
        record_llm_call,
        record_llm_error,
        record_fallback,
        calculate_token_cost,
        PROMETHEUS_AVAILABLE
    )
    PROMETHEUS_ENABLED = PROMETHEUS_AVAILABLE
except ImportError:
    # 如果 prometheus_metrics 模块不存在，使用空实现
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
# 模型 Provider 配置
# ============================================================

@dataclass
class LLMProviderConfig:
    """LLM 供应商配置

    Args:
        name: 供应商标识（如 "deepseek"、"zhipu"）
        api_key: API Key
        base_url: API 地址
        model: 模型名称
        enabled: 是否启用
        priority: 优先级（数字越小优先级越高，用于降级排序）
        max_retries: 单次调用最大重试次数
        timeout: 单次调用超时时间（秒）
    """
    name: str
    api_key: str
    base_url: str
    model: str
    enabled: bool = True
    priority: int = 0
    max_retries: int = 2
    timeout: int = 30


# 【知识点】所有配置从 config.py 读取，不直接碰 os.getenv
_PROVIDERS: dict[str, LLMProviderConfig] = {}

# Provider 1：DeepSeek（默认主模型）
if DEEPSEEK_API_KEY:
    _PROVIDERS["deepseek"] = LLMProviderConfig(
        name="deepseek",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        priority=0,
    )

# Provider 2：智谱 AI（备用模型）
if LLM_ZHIPU_API_KEY:
    _PROVIDERS["zhipu"] = LLMProviderConfig(
        name="zhipu",
        api_key=LLM_ZHIPU_API_KEY,
        base_url=LLM_ZHIPU_BASE_URL,
        model=LLM_ZHIPU_MODEL,
        priority=1,
    )

# Provider 3：Ollama（本地模型）
# 本地模型不需要API Key，始终启用
_PROVIDERS["ollama"] = LLMProviderConfig(
    name="ollama",
    api_key="",  # 本地运行不需要API Key
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    priority=2,  # 优先级最低，作为降级选项
    timeout=OLLAMA_TIMEOUT,  # 使用配置的超时时间
)


# ============================================================
# 调用统计
# ============================================================

@dataclass
class LLMCallStats:
    """LLM 调用统计（进程内存级，重启清零）

    【知识点】生产环境中这些指标应该上报到 Prometheus/Grafana
    Prometheus 是"数据引擎"（收集、存储、计算）
    Grafana 是"展示界面"（可视化、分析、告警）
    这里用内存字典简单实现，方便开发阶段观察
    """
    total_calls: int = 0
    success_calls: int = 0
    fallback_calls: int = 0
    error_calls: int = 0
    calls_by_provider: dict[str, int] = field(default_factory=dict)


_stats = LLMCallStats()


def get_call_stats() -> LLMCallStats:
    """获取调用统计数据

    Args:
        无参数

    Returns:
        LLMCallStats: 当前的调用统计
    """
    return _stats


# ============================================================
# LLM 实例缓存
# ============================================================
# 缓存 key = (provider_name, temperature)
_llm_cache: dict[tuple[str, float], ChatOpenAI] = {}


def _create_llm(provider: LLMProviderConfig, temperature: float, streaming: bool = False) -> ChatOpenAI:
    """根据 Provider 配置创建 LLM 实例

    Args:
        provider: LLM 供应商配置
        temperature: 输出随机性
        streaming: 是否启用流式输出（默认False）

    Returns:
        ChatOpenAI: LLM 实例
    """
    # Ollama 需要特殊处理
    if provider.name == "ollama":
        # 使用Ollama专用temperature配置
        ollama_temp = OLLAMA_TEMPERATURE if temperature == TEMPERATURE else temperature
        
        # ChatOllama 使用 request_timeout 而不是 timeout
        return ChatOllama(
            base_url=provider.base_url,
            model=provider.model,
            temperature=ollama_temp,
            request_timeout=provider.timeout,
            streaming=streaming,  # 添加流式支持
            # Ollama 特定参数通过 model_kwargs 传递
            model_kwargs={
                "num_gpu": OLLAMA_NUM_GPU,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_MAX_TOKENS,
            }
        )
    
    # 其他Provider使用ChatOpenAI
    # 注意：strict参数在最新版本的langchain-openai中可能不被支持
    # 根据错误信息，OpenAI客户端不支持strict参数
    return ChatOpenAI(
        api_key=provider.api_key,
        base_url=provider.base_url,
        model=provider.model,
        temperature=temperature,
        max_retries=provider.max_retries,
        request_timeout=provider.timeout,
        streaming=streaming,  # 添加流式支持
    )


def _get_default_provider_name() -> str:
    """获取默认 Provider 名称（优先级最高的已启用 Provider）

    Args:
        无参数

    Returns:
        str: Provider 名称
    """
    if DEFAULT_LLM_PROVIDER and DEFAULT_LLM_PROVIDER in _PROVIDERS:
        return DEFAULT_LLM_PROVIDER

    # 按 priority 排序，取第一个
    sorted_providers = sorted(
        _PROVIDERS.values(),
        key=lambda p: p.priority,
    )
    if sorted_providers:
        return sorted_providers[0].name

    raise RuntimeError("没有可用的 LLM Provider，请检查 .env 中的 API Key 配置")


def _get_fallback_providers(exclude: str) -> list[LLMProviderConfig]:
    """获取降级 Provider 列表（排除当前失败的 Provider）

    Args:
        exclude: 需要排除的 Provider 名称

    Returns:
        list[LLMProviderConfig]: 按优先级排序的备用 Provider 列表
    """
    return sorted(
        [p for p in _PROVIDERS.values() if p.name != exclude and p.enabled],
        key=lambda p: p.priority,
    )


# ============================================================
# 核心 API：get_llm()
# ============================================================

def get_llm(
    temperature: float | None = None, 
    provider: str | None = None,
    streaming: bool = False,) -> ChatOpenAI:
    """获取 LLM 实例（Gateway 核心方法）

    【知识点】Gateway 模式的核心入口：
    - 指定 provider → 使用指定的模型供应商
    - 不指定 provider → 使用默认（优先级最高的）供应商
    - 相同 (provider, temperature) 组合返回缓存实例
    - 消费方无需关心具体用的是哪个模型，只管调用

    Args:
        temperature: LLM 输出的随机性控制，None 时使用 config.TEMPERATURE 默认值
        provider: 指定 LLM 供应商名称（如 "deepseek"、"zhipu"），
                  None 时使用默认供应商
        streaming: 是否启用流式输出（默认False）

    Returns:
        ChatOpenAI: LLM 实例
    """
    temp = temperature if temperature is not None else TEMPERATURE
    provider_name = provider or _get_default_provider_name()

    # 缓存key需要包含streaming参数
    cache_key = (provider_name, temp, streaming)
    if cache_key not in _llm_cache:
        if provider_name not in _PROVIDERS:
            raise ValueError(
                f"未知的 LLM Provider: '{provider_name}'，"
                f"可用: {list(_PROVIDERS.keys())}"
            )
        config = _PROVIDERS[provider_name]
        logger.info(
            f"创建 LLM 实例: provider={provider_name}, "
            f"model={config.model}, temperature={temp}, streaming={streaming}"
        )
        _llm_cache[cache_key] = _create_llm(config, temp, streaming)

    return _llm_cache[cache_key]


def invoke_with_fallback(
    messages: list,
    temperature: float | None = None,
    provider: str | None = None,) -> str:
    """带容灾降级的 LLM 调用

    【知识点】容灾降级流程：
    1. 先用主 Provider 调用
    2. 如果超时/报错，自动切到下一个备用 Provider
    3. 所有 Provider 都失败，抛出异常
    4. 每次调用记录统计数据

    【企业实战 — 为什么需要降级？】
    - DeepSeek API 偶尔 503 维护 → 自动切智谱 API，用户无感知
    - 智谱 API 限流 → 自动切 DeepSeek
    - 单点故障不影响业务连续性

    Args:
        messages: LangChain 消息列表（SystemMessage / HumanMessage 等）
        temperature: 输出随机性
        provider: 首选 Provider，None 时使用默认

    Returns:
        str: LLM 回复文本
    """
    temp = temperature if temperature is not None else TEMPERATURE
    provider_name = provider or _get_default_provider_name()
    config = _PROVIDERS[provider_name]

    _stats.total_calls += 1

    # 尝试主 Provider
    try:
        llm = get_llm(temperature=temp, provider=provider_name)
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time

        _stats.success_calls += 1
        _stats.calls_by_provider[provider_name] = (
            _stats.calls_by_provider.get(provider_name, 0) + 1
        )
        logger.info(
            f"LLM 调用成功: provider={provider_name}, "
            f"耗时={elapsed:.2f}s"
        )
        
        # 记录 Prometheus 指标
        if PROMETHEUS_ENABLED:
            # 提取 Token 使用信息
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'prompt_tokens', None)
                output_tokens = getattr(response.usage, 'completion_tokens', None)
            
            # 计算成本
            cost_yuan = calculate_token_cost(provider_name, input_tokens or 0, output_tokens or 0)
            
            # 记录成功调用
            record_llm_call(
                provider=provider_name,
                model=config.model,
                duration=elapsed,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_yuan=cost_yuan,
                endpoint="invoke"
            )
        
        return response.content

    except Exception as e:
        logger.warning(
            f"LLM 调用失败: provider={provider_name}, 错误={e}，尝试降级..."
        )
        
        # 记录 Prometheus 错误指标
        if PROMETHEUS_ENABLED:
            # 判断错误类型
            error_type = "api_error"
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                error_type = "timeout"
            elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
                error_type = "rate_limit"
            elif "connection" in str(e).lower() or "network" in str(e).lower():
                error_type = "network"
            
            record_llm_error(
                provider=provider_name,
                model=config.model,
                error_type=error_type,
                endpoint="invoke"
            )

    # 降级：尝试备用 Provider
    fallback_providers = _get_fallback_providers(exclude=provider_name)
    for fallback in fallback_providers:
        try:
            llm = get_llm(temperature=temp, provider=fallback.name)
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - start_time

            _stats.success_calls += 1
            _stats.fallback_calls += 1
            _stats.calls_by_provider[fallback.name] = (
                _stats.calls_by_provider.get(fallback.name, 0) + 1
            )
            logger.warning(
                f"LLM 降级成功: fallback={fallback.name}, "
                f"耗时={elapsed:.2f}s"
            )
            
            # 记录 Prometheus 降级指标
            if PROMETHEUS_ENABLED:
                # 记录降级事件
                record_fallback(
                    from_provider=provider_name,
                    to_provider=fallback.name
                )
                
                # 提取 Token 使用信息
                input_tokens = None
                output_tokens = None
                if hasattr(response, 'usage'):
                    input_tokens = getattr(response.usage, 'prompt_tokens', None)
                    output_tokens = getattr(response.usage, 'completion_tokens', None)
                
                # 计算成本
                cost_yuan = calculate_token_cost(fallback.name, input_tokens or 0, output_tokens or 0)
                
                # 记录降级调用
                record_llm_call(
                    provider=fallback.name,
                    model=fallback.model,
                    duration=elapsed,
                    status="fallback",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_yuan=cost_yuan,
                    endpoint="invoke"
                )
            
            return response.content

        except Exception as fallback_error:
            logger.error(
                f"LLM 降级失败: provider={fallback.name}, "
                f"错误={fallback_error}"
            )
            
            # 记录 Prometheus 错误指标
            if PROMETHEUS_ENABLED:
                error_type = "api_error"
                if "timeout" in str(fallback_error).lower():
                    error_type = "timeout"
                
                record_llm_error(
                    provider=fallback.name,
                    model=fallback.model,
                    error_type=error_type,
                    endpoint="invoke"
                )

    # 所有 Provider 都失败
    _stats.error_calls += 1
    
    # 记录最终失败指标
    if PROMETHEUS_ENABLED:
        record_llm_error(
            provider=provider_name,
            model=config.model,
            error_type="all_failed",
            endpoint="invoke"
        )
    
    raise RuntimeError(
        f"所有 LLM Provider 均不可用: {list(_PROVIDERS.keys())}"
    )


# ============================================================
# 启动时打印可用 Provider
# ============================================================
if _PROVIDERS:
    _provider_list = ", ".join(
        f"{p.name}({p.model})" for p in sorted(_PROVIDERS.values(), key=lambda x: x.priority)
    )
    logger.info(f"LLM Gateway 就绪，可用 Provider: {_provider_list}")
else:
    logger.warning("LLM Gateway: 没有配置任何 Provider，请检查 .env")
# ============================================================
# Ollama 健康检查与模型管理
# ============================================================

def check_ollama_health() -> bool:
    """检查 Ollama 服务是否可用
    
    【知识点】生产环境健康检查：
    1. 服务启动时检查，避免调用时才发现不可用
    2. 定期监控，及时发现服务异常
    3. 提供详细的错误信息，方便故障排查
    
    Returns:
        bool: Ollama 服务是否可用
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            logger.info(f"Ollama 健康检查通过: {OLLAMA_BASE_URL}")
            return True
        else:
            logger.warning(
                f"Ollama 健康检查失败: HTTP {response.status_code}, "
                f"URL: {OLLAMA_BASE_URL}"
            )
            return False
    except requests.exceptions.ConnectionError:
        logger.error(
            f"无法连接到 Ollama 服务: {OLLAMA_BASE_URL}，"
            "请确保 Ollama 已启动并运行在指定端口"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(f"Ollama 健康检查超时: {OLLAMA_BASE_URL}")
        return False
    except Exception as e:
        logger.error(f"Ollama 健康检查异常: {e}")
        return False


def get_ollama_models() -> List[Dict[str, Any]]:
    """获取 Ollama 中已下载的模型列表
    
    【知识点】模型管理：
    1. 动态获取可用模型，避免硬编码
    2. 支持多模型切换，适应不同场景
    3. 提供模型信息，方便运维管理
    
    Returns:
        List[Dict[str, Any]]: 模型列表，每个模型包含 name、size、modified 等信息
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        else:
            logger.error(f"获取 Ollama 模型列表失败: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"获取 Ollama 模型列表异常: {e}")
        return []


def check_model_available(model_name: str) -> bool:
    """检查指定模型是否在 Ollama 中可用
    
    Args:
        model_name: 模型名称（如 "qwen2.5:3b"）
        
    Returns:
        bool: 模型是否可用
    """
    models = get_ollama_models()
    for model in models:
        if model.get("name") == model_name:
            return True
    return False


def pull_ollama_model(model_name: str) -> bool:
    """从 Ollama 拉取指定模型
    
    【企业实战 — 模型部署自动化】：
    1. 首次部署时自动下载所需模型
    2. 支持模型版本更新
    3. 提供进度反馈，避免长时间无响应
    
    Args:
        model_name: 要拉取的模型名称
        
    Returns:
        bool: 拉取是否成功
    """
    try:
        logger.info(f"开始拉取 Ollama 模型: {model_name}")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            timeout=300  # 模型下载可能需要较长时间
        )
        
        if response.status_code == 200:
            logger.info(f"Ollama 模型拉取成功: {model_name}")
            return True
        else:
            logger.error(f"Ollama 模型拉取失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ollama 模型拉取异常: {e}")
        return False


def get_ollama_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """获取 Ollama 模型的详细信息
    
    Args:
        model_name: 模型名称
        
    Returns:
        Optional[Dict[str, Any]]: 模型详细信息，如果模型不存在则返回 None
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model_name},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"获取 Ollama 模型信息失败: {model_name}")
            return None
    except Exception as e:
        logger.error(f"获取 Ollama 模型信息异常: {e}")
        return None


def optimize_ollama_for_production() -> Dict[str, Any]:
    """优化 Ollama 生产环境配置建议
    
    【知识点】生产环境优化：
    1. GPU 内存分配优化
    2. 批处理大小调整
    3. 上下文长度平衡
    4. 并发请求处理
    
    Returns:
        Dict[str, Any]: 优化建议
    """
    recommendations = {
        "current_config": {
            "model": OLLAMA_MODEL,
            "num_gpu": OLLAMA_NUM_GPU,
            "num_ctx": OLLAMA_NUM_CTX,
            "timeout": OLLAMA_TIMEOUT,
            "max_tokens": OLLAMA_MAX_TOKENS,
        },
        "recommendations": []
    }
    
    # 根据 GPU 数量给出建议
    if OLLAMA_NUM_GPU == 0:
        recommendations["recommendations"].append(
            "当前配置为 CPU 模式，建议至少使用 1 个 GPU 以获得更好的性能"
        )
    elif OLLAMA_NUM_GPU == 1:
        recommendations["recommendations"].append(
            "单 GPU 配置适合中小规模生产环境，建议监控 GPU 内存使用率"
        )
    else:
        recommendations["recommendations"].append(
            f"多 GPU 配置（{OLLAMA_NUM_GPU}个）适合高并发生产环境"
        )
    
    # 上下文长度建议
    if OLLAMA_NUM_CTX < 2048:
        recommendations["recommendations"].append(
            f"当前上下文长度 {OLLAMA_NUM_CTX} 较小，建议增加到 4096 以支持更长的对话"
        )
    elif OLLAMA_NUM_CTX > 8192:
        recommendations["recommendations"].append(
            f"当前上下文长度 {OLLAMA_NUM_CTX} 较大，可能影响性能，建议根据实际需求调整"
        )
    
    # 超时时间建议
    if OLLAMA_TIMEOUT < 30:
        recommendations["recommendations"].append(
            f"当前超时时间 {OLLAMA_TIMEOUT} 秒较短，建议增加到 60 秒以上以避免超时"
        )
    
    return recommendations


# ============================================================
# 启动时执行健康检查
# ============================================================
if "ollama" in _PROVIDERS:
    if check_ollama_health():
        logger.info("Ollama 服务健康检查通过")
        
        # 检查配置的模型是否可用
        if not check_model_available(OLLAMA_MODEL):
            logger.warning(
                f"配置的 Ollama 模型 '{OLLAMA_MODEL}' 不存在，"
                f"可用模型: {[m.get('name', '') for m in get_ollama_models()]}"
            )
            
            # 提供优化建议
            recommendations = optimize_ollama_for_production()
            logger.info(f"Ollama 生产环境优化建议: {recommendations}")
    else:
        logger.warning("Ollama 服务健康检查失败，本地模型将不可用")
else:
    logger.info("Ollama Provider 未启用，跳过健康检查")
# ============================================================
# 智能模型路由功能
# ============================================================

def detect_language(text: str) -> str:
    """检测文本的主要语言
    
    【企业实战 — 语言检测】：
    1. 简单但有效的语言检测
    2. 专注于中英文区分（你的主要需求）
    3. 性能高效，适合生产环境
    
    Args:
        text: 要检测的文本
        
    Returns:
        str: "zh"（中文）或 "en"（英文）或 "mixed"（混合）
    """
    if not text:
        return "en"  # 默认英文
    
    # 统计中文字符数量
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    # 统计英文字符数量（字母和空格）
    english_chars = sum(1 for c in text if c.isalpha() or c.isspace())
    
    # 如果文本太短，使用默认
    if len(text) < 5:
        return "en"
    
    # 计算中文比例
    chinese_ratio = chinese_chars / len(text) if len(text) > 0 else 0
    
    # 决策逻辑
    if chinese_ratio > 0.3:  # 中文比例超过30%
        return "zh"
    elif chinese_ratio > 0.1:  # 中文比例10%-30%
        return "mixed"
    else:
        return "en"


def select_model_by_language(text: str, available_models: List[str] = None) -> str:
    """根据语言选择最合适的模型
    
    【智能路由策略】：
    1. 中文 → qwen2.5:3b（中文优化）
    2. 英文 → llama3.2:3b（英文优化）
    3. 混合 → qwen2.5:3b（多语言支持更好）
    4. 默认 → OLLAMA_DEFAULT_MODEL
    
    Args:
        text: 输入文本
        available_models: 可用的模型列表，None时使用配置的OLLAMA_MODELS
        
    Returns:
        str: 选择的模型名称
    """
    if not OLLAMA_ENABLE_SMART_ROUTING:
        return OLLAMA_DEFAULT_MODEL
    
    # 获取可用模型
    if available_models is None:
        available_models = OLLAMA_MODELS
    
    # 检查模型可用性
    available_models = [m for m in available_models if check_model_available(m)]
    
    if not available_models:
        logger.warning("没有可用的 Ollama 模型，使用默认配置")
        return OLLAMA_MODEL
    
    # 检测语言
    language = detect_language(text)
    
    # 根据语言选择模型
    if language == "zh":
        # 优先选择 qwen2.5:3b
        if "qwen2.5:3b" in available_models:
            logger.debug(f"中文文本，选择 qwen2.5:3b")
            return "qwen2.5:3b"
        elif "llama3.2:3b" in available_models:
            logger.debug(f"中文文本，qwen2.5:3b 不可用，选择 llama3.2:3b")
            return "llama3.2:3b"
    
    elif language == "en":
        # 优先选择 llama3.2:3b
        if "llama3.2:3b" in available_models:
            logger.debug(f"英文文本，选择 llama3.2:3b")
            return "llama3.2:3b"
        elif "qwen2.5:3b" in available_models:
            logger.debug(f"英文文本，llama3.2:3b 不可用，选择 qwen2.5:3b")
            return "qwen2.5:3b"
    
    else:  # mixed 或其他
        # 混合语言优先选择 qwen2.5:3b（多语言支持更好）
        if "qwen2.5:3b" in available_models:
            logger.debug(f"混合语言文本，选择 qwen2.5:3b")
            return "qwen2.5:3b"
        elif "llama3.2:3b" in available_models:
            logger.debug(f"混合语言文本，qwen2.5:3b 不可用，选择 llama3.2:3b")
            return "llama3.2:3b"
    
    # 如果以上都不满足，使用第一个可用模型
    logger.debug(f"使用第一个可用模型: {available_models[0]}")
    return available_models[0]


def get_smart_llm(
    text: str = "",
    temperature: float | None = None,
    force_model: str | None = None) -> ChatOpenAI:
    """智能获取 LLM 实例（根据语言自动选择模型）
    
    【企业级智能路由】：
    1. 自动检测输入文本语言
    2. 选择最适合的本地模型
    3. 支持手动指定模型（覆盖自动选择）
    4. 与现有缓存机制兼容
    
    Args:
        text: 输入文本（用于语言检测）
        temperature: 输出随机性
        force_model: 强制使用指定模型（覆盖自动选择）
        
    Returns:
        ChatOpenAI: LLM 实例
    """
    # 确定使用的模型
    if force_model:
        model_name = force_model
        logger.info(f"强制使用指定模型: {model_name}")
    elif OLLAMA_ENABLE_SMART_ROUTING and text:
        model_name = select_model_by_language(text)
        logger.info(f"智能路由选择模型: {model_name} (语言: {detect_language(text)})")
    else:
        model_name = OLLAMA_DEFAULT_MODEL
        logger.info(f"使用默认模型: {model_name}")
    
    # 检查模型是否可用
    if not check_model_available(model_name):
        logger.warning(f"模型 {model_name} 不可用，尝试使用其他可用模型")
        available_models = [m for m in OLLAMA_MODELS if check_model_available(m)]
        if available_models:
            model_name = available_models[0]
            logger.warning(f"切换到可用模型: {model_name}")
        else:
            raise RuntimeError(f"没有可用的 Ollama 模型，请检查模型下载状态")
    
    # 创建临时Provider配置
    temp_provider = LLMProviderConfig(
        name="ollama_smart",
        api_key="",
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        timeout=OLLAMA_TIMEOUT,
    )
    
    # 使用Ollama专用temperature配置
    ollama_temp = OLLAMA_TEMPERATURE if temperature == TEMPERATURE else temperature
    
    return ChatOllama(
        base_url=temp_provider.base_url,
        model=temp_provider.model,
        temperature=ollama_temp,
        request_timeout=temp_provider.timeout,
        model_kwargs={
            "num_gpu": OLLAMA_NUM_GPU,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_MAX_TOKENS,
        }
    )


def invoke_with_smart_routing(
    messages: list,
    temperature: float | None = None,
    force_model: str | None = None,) -> str:
    """带智能路由的 LLM 调用
    
    【企业级智能调用】：
    1. 自动分析消息内容语言
    2. 选择最优本地模型
    3. 保留容灾降级能力
    4. 提供详细的调用日志
    
    Args:
        messages: LangChain 消息列表
        temperature: 输出随机性
        force_model: 强制使用指定模型
        
    Returns:
        str: LLM 回复文本
    """
    # 提取所有消息文本用于语言检测
    all_text = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            all_text += str(msg.content) + " "
    
    # 获取智能LLM实例
    llm = get_smart_llm(
        text=all_text,
        temperature=temperature,
        force_model=force_model
    )
    
    # 记录调用统计
    _stats.total_calls += 1
    
    try:
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        _stats.success_calls += 1
        model_name = llm.model if hasattr(llm, 'model') else "unknown"
        _stats.calls_by_provider[model_name] = (
            _stats.calls_by_provider.get(model_name, 0) + 1
        )
        
        logger.info(
            f"智能路由调用成功: model={model_name}, "
            f"耗时={elapsed:.2f}s"
        )
        
        # 记录 Prometheus 智能路由指标
        if PROMETHEUS_ENABLED:
            # 检测语言
            from prometheus_metrics import detect_language, record_smart_routing
            language = detect_language(all_text)
            
            # 记录智能路由
            record_smart_routing(
                language=language,
                selected_model=model_name
            )
            
            # 提取 Token 使用信息
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'prompt_tokens', None)
                output_tokens = getattr(response.usage, 'completion_tokens', None)
            
            # 计算成本（Ollama本地模型成本为0）
            cost_yuan = 0.0
            
            # 记录调用
            record_llm_call(
                provider="ollama",
                model=model_name,
                duration=elapsed,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_yuan=cost_yuan,
                endpoint="smart_routing"
            )
        
        return response.content
        
    except Exception as e:
        logger.error(f"智能路由调用失败: {e}")
        _stats.error_calls += 1
        
        # 记录 Prometheus 错误指标
        if PROMETHEUS_ENABLED:
            model_name = llm.model if hasattr(llm, 'model') else "unknown"
            record_llm_error(
                provider="ollama",
                model=model_name,
                error_type="smart_routing_failed",
                endpoint="smart_routing"
            )
        
        # 如果智能路由失败，尝试使用默认的invoke_with_fallback
        logger.info("智能路由失败，尝试使用默认降级调用...")
        return invoke_with_fallback(
            messages=messages,
            temperature=temperature,
            provider="ollama"  # 使用默认Ollama配置
        )


# ============================================================
# 多模型管理功能
# ============================================================

def get_available_ollama_models() -> List[str]:
    """获取实际可用的 Ollama 模型列表
    
    Returns:
        List[str]: 可用的模型名称列表
    """
    all_models = get_ollama_models()
    return [model.get("name") for model in all_models if model.get("name")]


def switch_default_model(new_model: str) -> bool:
    """切换默认模型
    
    Args:
        new_model: 新的默认模型名称
        
    Returns:
        bool: 切换是否成功
    """
    if not check_model_available(new_model):
        logger.error(f"模型 {new_model} 不可用，无法切换")
        return False
    
    # 更新环境变量（临时，进程内有效）
    global OLLAMA_DEFAULT_MODEL
    OLLAMA_DEFAULT_MODEL = new_model
    
    logger.info(f"默认模型已切换为: {new_model}")
    return True


def get_model_performance_stats() -> Dict[str, Any]:
    """获取各模型性能统计
    
    Returns:
        Dict[str, Any]: 模型性能统计数据
    """
    stats = {
        "total_calls": _stats.total_calls,
        "success_calls": _stats.success_calls,
        "error_calls": _stats.error_calls,
        "fallback_calls": _stats.fallback_calls,
        "model_performance": {}
    }
    
    # 计算各模型平均响应时间（需要额外记录）
    # 这里可以扩展为更详细的性能监控
    
    return stats


# ============================================================
# 启动时初始化多模型支持
# ============================================================
if "ollama" in _PROVIDERS and check_ollama_health():
    # 检查配置的所有模型是否可用
    available_models = get_available_ollama_models()
    configured_models = OLLAMA_MODELS
    
    logger.info(f"Ollama 多模型支持初始化:")
    logger.info(f"  配置的模型: {configured_models}")
    logger.info(f"  可用的模型: {available_models}")
    
    # 检查默认模型是否可用
    if OLLAMA_DEFAULT_MODEL not in available_models:
        logger.warning(
            f"默认模型 '{OLLAMA_DEFAULT_MODEL}' 不可用，"
            f"可用模型: {available_models}"
        )
        # 尝试使用第一个可用模型
        if available_models:
            logger.info(f"自动切换到第一个可用模型: {available_models[0]}")
            OLLAMA_DEFAULT_MODEL = available_models[0]
    
    # 记录智能路由状态
    if OLLAMA_ENABLE_SMART_ROUTING:
        logger.info("智能路由功能已启用")
        logger.info("  路由策略: 中文→qwen2.5:3b, 英文→llama3.2:3b, 混合→qwen2.5:3b")
    else:
        logger.info("智能路由功能已禁用，使用默认模型")
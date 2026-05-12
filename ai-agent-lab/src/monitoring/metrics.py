"""
=== Prometheus 指标定义模块 ===

基于 Google SRE 四大黄金指标设计：
1. 延迟（Latency）: 请求处理时间
2. 流量（Traffic）: 请求速率/QPS
3. 错误（Errors）: 错误率
4. 饱和度（Saturation）: 资源使用率

企业级 Agent 监控指标体系：
- LLM 调用指标（多 Provider、多模型）
- Agent 执行指标（多 Agent 类型）
- RAG 检索指标
- 业务指标（会话、成本）
- 系统资源指标
"""

import time
from typing import Optional, Tuple
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess, start_http_server
)
from fastapi import Response
import os

# ============================================================
# 1. 延迟指标（Latency）
# ============================================================

# LLM 调用延迟
LLM_CALL_DURATION = Histogram(
    'llm_call_duration_seconds',
    'LLM 调用耗时（秒）',
    ['provider', 'model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Agent 执行延迟
AGENT_EXECUTION_DURATION = Histogram(
    'agent_execution_duration_seconds',
    'Agent 执行耗时（秒）',
    ['agent_type', 'route'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# API 请求延迟
API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API 请求耗时（秒）',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# RAG 检索延迟
RAG_RETRIEVAL_DURATION = Histogram(
    'rag_retrieval_duration_seconds',
    'RAG 检索耗时（秒）',
    ['knowledge_base'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# ============================================================
# 2. 流量指标（Traffic）
# ============================================================

# LLM 调用次数
LLM_CALLS_TOTAL = Counter(
    'llm_calls_total',
    'LLM 总调用次数',
    ['provider', 'model', 'status']  # status: success/error/fallback
)

# Agent 调用次数
AGENT_CALLS_TOTAL = Counter(
    'agent_calls_total',
    'Agent 总调用次数',
    ['agent_type', 'route', 'status']  # agent_type: supervisor/tech/java/general
)

# API 请求次数
API_REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'API 总请求次数',
    ['method', 'endpoint', 'status_code']
)

# 用户会话数
USER_SESSIONS_TOTAL = Counter(
    'user_sessions_total',
    '用户会话总数',
    ['status']  # status: created/active/closed
)

# ============================================================
# 3. 错误指标（Errors）
# ============================================================

# LLM 错误率
LLM_ERRORS_TOTAL = Counter(
    'llm_errors_total',
    'LLM 错误总数',
    ['provider', 'model', 'error_type']  # error_type: timeout/rate_limit/api_error/network
)

# Agent 错误率
AGENT_ERRORS_TOTAL = Counter(
    'agent_errors_total',
    'Agent 错误总数',
    ['agent_type', 'error_type']  # error_type: execution/timeout/memory
)

# API 错误率
API_ERRORS_TOTAL = Counter(
    'api_errors_total',
    'API 错误总数',
    ['endpoint', 'error_type']  # error_type: validation/authentication/internal
)

# RAG 检索错误
RAG_ERRORS_TOTAL = Counter(
    'rag_errors_total',
    'RAG 检索错误总数',
    ['knowledge_base', 'error_type']
)

# ============================================================
# 4. 饱和度指标（Saturation）
# ============================================================

# 活跃会话数
USER_SESSIONS_ACTIVE = Gauge(
    'user_sessions_active',
    '活跃用户会话数'
)

# LLM Token 使用量
LLM_TOKENS_TOTAL = Counter(
    'llm_tokens_total',
    'LLM Token 总使用量',
    ['provider', 'model', 'token_type']  # token_type: input/output/total
)

# 成本指标
LLM_COST_TOTAL = Counter(
    'llm_cost_total_yuan',
    'LLM 总成本（元）',
    ['provider', 'model']
)

# 系统资源使用率（需要外部采集）
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    '系统 CPU 使用率（百分比）'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    '系统内存使用率（百分比）'
)

# ============================================================
# 5. 业务指标
# ============================================================

# 降级调用次数
LLM_FALLBACKS_TOTAL = Counter(
    'llm_fallbacks_total',
    'LLM 降级调用总数',
    ['from_provider', 'to_provider']
)

# 智能路由统计
SMART_ROUTING_TOTAL = Counter(
    'smart_routing_total',
    '智能路由调用总数',
    ['language', 'selected_model']
)

# RAG 检索质量
RAG_RELEVANCE_SCORE = Histogram(
    'rag_relevance_score',
    'RAG 检索相关性分数',
    ['knowledge_base'],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

# ============================================================
# 工具函数
# ============================================================

class Timer:
    """计时器上下文管理器，用于测量代码块执行时间"""
    
    def __init__(self, metric: Histogram, **labels):
        self.metric = metric
        self.labels = labels
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.metric.labels(**self.labels).observe(elapsed)


def record_llm_call(
    provider: str,
    model: str,
    duration: float,
    status: str = "success",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_yuan: Optional[float] = None,
    endpoint: str = "invoke"
):
    """记录 LLM 调用指标
    
    Args:
        provider: LLM 供应商（deepseek/zhipu/ollama）
        model: 模型名称
        duration: 调用耗时（秒）
        status: 调用状态（success/error/fallback）
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        cost_yuan: 调用成本（元）
        endpoint: 调用端点
    """
    # 记录延迟
    LLM_CALL_DURATION.labels(
        provider=provider,
        model=model,
        endpoint=endpoint
    ).observe(duration)
    
    # 记录流量
    LLM_CALLS_TOTAL.labels(
        provider=provider,
        model=model,
        status=status
    ).inc()
    
    # 记录 Token 使用量
    if input_tokens is not None:
        LLM_TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            token_type="input"
        ).inc(input_tokens)
        
    if output_tokens is not None:
        LLM_TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            token_type="output"
        ).inc(output_tokens)
        
    if input_tokens is not None and output_tokens is not None:
        LLM_TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            token_type="total"
        ).inc(input_tokens + output_tokens)
    
    # 记录成本
    if cost_yuan is not None:
        LLM_COST_TOTAL.labels(
            provider=provider,
            model=model
        ).inc(cost_yuan)


def record_llm_error(
    provider: str,
    model: str,
    error_type: str = "api_error",
    endpoint: str = "invoke"
):
    """记录 LLM 错误指标
    
    Args:
        provider: LLM 供应商
        model: 模型名称
        error_type: 错误类型（timeout/rate_limit/api_error/network）
        endpoint: 调用端点
    """
    LLM_ERRORS_TOTAL.labels(
        provider=provider,
        model=model,
        error_type=error_type
    ).inc()
    
    # 同时记录为失败的调用
    LLM_CALLS_TOTAL.labels(
        provider=provider,
        model=model,
        status="error"
    ).inc()


def record_agent_call(
    agent_type: str,
    route: str,
    duration: float,
    status: str = "success"
):
    """记录 Agent 调用指标
    
    Args:
        agent_type: Agent 类型（supervisor/tech/java/general）
        route: 路由结果
        duration: 执行耗时（秒）
        status: 执行状态（success/error）
    """
    AGENT_EXECUTION_DURATION.labels(
        agent_type=agent_type,
        route=route
    ).observe(duration)
    
    AGENT_CALLS_TOTAL.labels(
        agent_type=agent_type,
        route=route,
        status=status
    ).inc()


def record_api_request(
    method: str,
    endpoint: str,
    duration: float,
    status_code: int
):
    """记录 API 请求指标
    
    Args:
        method: HTTP 方法（GET/POST等）
        endpoint: API 端点
        duration: 请求耗时（秒）
        status_code: HTTP 状态码
    """
    API_REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).observe(duration)
    
    API_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()
    
    # 记录错误（4xx, 5xx）
    if status_code >= 400:
        error_type = "client_error" if status_code < 500 else "server_error"
        API_ERRORS_TOTAL.labels(
            endpoint=endpoint,
            error_type=error_type
        ).inc()


def record_rag_retrieval(
    knowledge_base: str,
    duration: float,
    documents_retrieved: int,
    relevance_score: Optional[float] = None
):
    """记录 RAG 检索指标
    
    Args:
        knowledge_base: 知识库名称
        duration: 检索耗时（秒）
        documents_retrieved: 检索到的文档数
        relevance_score: 相关性分数（0-1）
    """
    RAG_RETRIEVAL_DURATION.labels(
        knowledge_base=knowledge_base
    ).observe(duration)
    
    # 记录检索到的文档数
    RAG_DOCUMENTS_RETRIEVED = Histogram(
        'rag_documents_retrieved',
        'RAG 检索到的文档数',
        ['knowledge_base'],
        buckets=[1, 3, 5, 10, 20, 50]
    )
    RAG_DOCUMENTS_RETRIEVED.labels(
        knowledge_base=knowledge_base
    ).observe(documents_retrieved)
    
    # 记录相关性分数
    if relevance_score is not None:
        RAG_RELEVANCE_SCORE.labels(
            knowledge_base=knowledge_base
        ).observe(relevance_score)


def record_fallback(
    from_provider: str,
    to_provider: str
):
    """记录降级调用
    
    Args:
        from_provider: 原 Provider
        to_provider: 降级后的 Provider
    """
    LLM_FALLBACKS_TOTAL.labels(
        from_provider=from_provider,
        to_provider=to_provider
    ).inc()


def record_smart_routing(
    language: str,
    selected_model: str
):
    """记录智能路由
    
    Args:
        language: 检测到的语言
        selected_model: 选择的模型
    """
    SMART_ROUTING_TOTAL.labels(
        language=language,
        selected_model=selected_model
    ).inc()


def record_user_session(status: str = "created"):
    """记录用户会话
    
    Args:
        status: 会话状态（created/active/closed）
    """
    USER_SESSIONS_TOTAL.labels(
        status=status
    ).inc()
    
    if status == "active":
        USER_SESSIONS_ACTIVE.inc()
    elif status == "closed":
        USER_SESSIONS_ACTIVE.dec()


def calculate_token_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """计算 Token 成本（元）
    
    Args:
        provider: LLM 供应商
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        
    Returns:
        float: 成本（元）
    """
    # 各供应商定价（元/Token）
    pricing = {
        "deepseek": {"input": 0.000014, "output": 0.000028},
        "zhipu": {"input": 0.000010, "output": 0.000020},
        "ollama": {"input": 0.0, "output": 0.0},  # 本地模型免费
    }
    
    price = pricing.get(provider, {"input": 0.0, "output": 0.0})
    return (input_tokens * price["input"] + 
            output_tokens * price["output"])


# ============================================================
# Prometheus 端点
# ============================================================

def get_metrics_registry():
    """获取 Prometheus 注册表（支持多进程）"""
    if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return None


def metrics_endpoint():
    """返回 Prometheus 格式的 metrics"""
    registry = get_metrics_registry()
    if registry:
        data = generate_latest(registry)
    else:
        data = generate_latest()
    
    return Response(
        data,
        media_type=CONTENT_TYPE_LATEST,
        headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
        }
    )


def start_metrics_server(port: int = 8001):
    """启动独立的 metrics 服务器（可选）"""
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")


# ============================================================
# 初始化检查
# ============================================================

def check_prometheus_ready():
    """检查 Prometheus 客户端是否就绪"""
    try:
        # 尝试导入 prometheus_client
        import prometheus_client
        return True
    except ImportError:
        print("警告: prometheus_client 未安装，监控功能将受限")
        print("请运行: pip install prometheus-client")
        return False


# 模块导入时检查
PROMETHEUS_AVAILABLE = check_prometheus_ready()

if not PROMETHEUS_AVAILABLE:
    print("提示: 如需完整监控功能，请安装 prometheus-client")
    print("使用虚拟环境安装: agent-lab/.venv/Scripts/pip.exe install prometheus-client")
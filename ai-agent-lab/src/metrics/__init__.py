"""
=== 监控指标模块 ===

【功能】：
1. 基于 Prometheus 的指标收集
2. LLM 调用监控
3. Agent 执行监控
4. RAG 检索监控
5. API 请求监控

【导出内容】：
- 所有指标对象（Counter/Histogram/Gauge）
- 指标记录函数
- 工具函数
"""

# 从 metrics.py 导入所有内容
from .metrics import *

# 导出列表
__all__ = [
    # 延迟指标
    'LLM_CALL_DURATION',
    'AGENT_EXECUTION_DURATION',
    'API_REQUEST_DURATION',
    'RAG_RETRIEVAL_DURATION',
    
    # 流量指标
    'LLM_CALLS_TOTAL',
    'AGENT_CALLS_TOTAL',
    'API_REQUESTS_TOTAL',
    'USER_SESSIONS_TOTAL',
    
    # 错误指标
    'LLM_ERRORS_TOTAL',
    'AGENT_ERRORS_TOTAL',
    'API_ERRORS_TOTAL',
    'RAG_ERRORS_TOTAL',
    
    # 饱和度指标
    'USER_SESSIONS_ACTIVE',
    'LLM_TOKENS_TOTAL',
    'LLM_COST_TOTAL',
    'SYSTEM_CPU_USAGE',
    'SYSTEM_MEMORY_USAGE',
    
    # 业务指标
    'LLM_FALLBACKS_TOTAL',
    'SMART_ROUTING_TOTAL',
    'RAG_RELEVANCE_SCORE',
    
    # 工具函数
    'Timer',
    'record_llm_call',
    'record_llm_error',
    'record_agent_call',
    'record_api_request',
    'record_rag_retrieval',
    'record_fallback',
    'record_smart_routing',
    'record_user_session',
    'calculate_token_cost',
    
    # 端点函数
    'get_metrics_registry',
    'metrics_endpoint',
    'start_metrics_server',
    
    # 状态检查
    'check_prometheus_ready',
    'PROMETHEUS_AVAILABLE',
]

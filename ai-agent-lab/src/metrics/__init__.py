
"""
监控指标模块

提供 Prometheus 指标收集和暴露功能
"""

from .metrics import (
    LLMCallMetrics,
    RAGMetrics,
    AgentMetrics,
    api_requests_total,
    api_requests_duration,
    llm_calls_total,
    llm_tokens_used,
    rag_queries_total,
    rag_relevant_docs_found,
    agent_execution_time,
    tool_calls_total,
    PROMETHEUS_AVAILABLE,
    record_llm_call,
    record_llm_error,
    record_fallback,
    calculate_token_cost,
    registry,
)

__all__ = [
    "LLMCallMetrics",
    "RAGMetrics",
    "AgentMetrics",
    "api_requests_total",
    "api_requests_duration",
    "llm_calls_total",
    "llm_tokens_used",
    "rag_queries_total",
    "rag_relevant_docs_found",
    "agent_execution_time",
    "tool_calls_total",
    "PROMETHEUS_AVAILABLE",
    "record_llm_call",
    "record_llm_error",
    "record_fallback",
    "calculate_token_cost",
    "registry",
]

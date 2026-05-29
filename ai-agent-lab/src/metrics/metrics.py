
"""
监控指标定义

包含所有 Prometheus 指标的定义和收集逻辑
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
)
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


# 创建全局注册器
registry = CollectorRegistry()

# ==================== API 指标 ====================

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_requests_duration = Histogram(
    "api_request_duration_seconds",
    "Duration of API requests in seconds",
    ["endpoint", "method"],
    registry=registry,
)

api_active_requests = Gauge(
    "api_active_requests",
    "Number of active API requests",
    ["endpoint"],
    registry=registry,
)

# ==================== LLM 指标 ====================

llm_calls_total = Counter(
    "llm_calls_total",
    "Total number of LLM calls",
    ["provider", "model", "success"],
    registry=registry,
)

llm_tokens_used = Counter(
    "llm_tokens_used",
    "Total tokens used by LLM",
    ["provider", "model", "type"],  # type: prompt/completion
    registry=registry,
)

llm_call_duration = Histogram(
    "llm_call_duration_seconds",
    "Duration of LLM calls in seconds",
    ["provider", "model"],
    registry=registry,
)

llm_fallback_calls = Counter(
    "llm_fallback_calls_total",
    "Number of fallback calls",
    ["from_provider", "to_provider"],
    registry=registry,
)

# ==================== RAG 指标 ====================

rag_queries_total = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["collection", "found"],
    registry=registry,
)

rag_relevant_docs_found = Summary(
    "rag_relevant_docs_found",
    "Number of relevant documents found",
    ["collection"],
    registry=registry,
)

rag_query_duration = Histogram(
    "rag_query_duration_seconds",
    "Duration of RAG queries in seconds",
    ["collection"],
    registry=registry,
)

rag_retrieval_recall = Summary(
    "rag_retrieval_recall",
    "RAG retrieval recall score",
    ["collection"],
    registry=registry,
)

# ==================== Agent 指标 ====================

agent_execution_time = Histogram(
    "agent_execution_time_seconds",
    "Execution time of agent nodes",
    ["agent_name", "node_type"],
    registry=registry,
)

agent_calls_total = Counter(
    "agent_calls_total",
    "Total number of agent calls",
    ["agent_name", "success"],
    registry=registry,
)

agent_tool_calls = Counter(
    "agent_tool_calls_total",
    "Total number of tool calls by agents",
    ["agent_name", "tool_name"],
    registry=registry,
)

agent_iterations = Summary(
    "agent_iterations",
    "Number of iterations per agent execution",
    ["agent_name"],
    registry=registry,
)

# ==================== 工具指标 ====================

tool_calls_total = Counter(
    "tool_calls_total",
    "Total number of tool calls",
    ["tool_name", "success"],
    registry=registry,
)

tool_call_duration = Histogram(
    "tool_call_duration_seconds",
    "Duration of tool calls in seconds",
    ["tool_name"],
    registry=registry,
)


@dataclass
class LLMCallMetrics:
    """LLM 调用指标收集器"""
    
    provider: str
    model: str
    start_time: float = field(default_factory=lambda: time.time())
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error_message: Optional[str] = None
    
    def record(self):
        """记录指标"""
        duration = time.time() - self.start_time
        
        llm_calls_total.labels(
            provider=self.provider,
            model=self.model,
            success=str(self.success)
        ).inc()
        
        llm_call_duration.labels(
            provider=self.provider,
            model=self.model
        ).observe(duration)
        
        if self.prompt_tokens > 0:
            llm_tokens_used.labels(
                provider=self.provider,
                model=self.model,
                type="prompt"
            ).inc(self.prompt_tokens)
        
        if self.completion_tokens > 0:
            llm_tokens_used.labels(
                provider=self.provider,
                model=self.model,
                type="completion"
            ).inc(self.completion_tokens)


@dataclass
class RAGMetrics:
    """RAG 查询指标收集器"""
    
    collection: str = "default"
    start_time: float = field(default_factory=lambda: time.time())
    found: bool = False
    doc_count: int = 0
    recall_score: float = 0.0
    
    def record(self):
        """记录指标"""
        duration = time.time() - self.start_time
        
        rag_queries_total.labels(
            collection=self.collection,
            found=str(self.found)
        ).inc()
        
        rag_query_duration.labels(
            collection=self.collection
        ).observe(duration)
        
        rag_relevant_docs_found.labels(
            collection=self.collection
        ).observe(self.doc_count)
        
        if self.recall_score > 0:
            rag_retrieval_recall.labels(
                collection=self.collection
            ).observe(self.recall_score)


@dataclass
class AgentMetrics:
    """Agent 执行指标收集器"""
    
    agent_name: str
    node_type: str = "default"
    start_time: float = field(default_factory=lambda: time.time())
    iterations: int = 1
    success: bool = True
    tool_calls: Dict[str, int] = field(default_factory=dict)
    
    def record(self):
        """记录指标"""
        duration = time.time() - self.start_time
        
        agent_execution_time.labels(
            agent_name=self.agent_name,
            node_type=self.node_type
        ).observe(duration)
        
        agent_calls_total.labels(
            agent_name=self.agent_name,
            success=str(self.success)
        ).inc()
        
        agent_iterations.labels(
            agent_name=self.agent_name
        ).observe(self.iterations)
        
        for tool_name, count in self.tool_calls.items():
            agent_tool_calls.labels(
                agent_name=self.agent_name,
                tool_name=tool_name
            ).inc(count)

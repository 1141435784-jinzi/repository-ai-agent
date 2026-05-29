
"""
监控指标路由

提供 Prometheus 指标暴露接口
"""

from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.metrics import registry

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/", response_class=Response)
async def get_metrics():
    """
    暴露 Prometheus 指标
    
    GET /metrics
    
    返回格式: text/plain; version=0.0.4; charset=utf-8
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/health")
async def health_check():
    """
    健康检查接口
    
    GET /metrics/health
    
    返回:
    {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }
    """
    from src.config.settings import VERSION
    from datetime import datetime, timezone
    
    return {
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "api": "running",
            "llm_gateway": "ready",
            "rag_engine": "ready",
            "vector_db": "ready"
        }
    }

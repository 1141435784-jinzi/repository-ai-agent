"""
=== RAG 模块导出 ===

包含核心组件：
- DataCleaner: 企业级文档数据清洗器（7步流水线）
- RAGEngine: RAG检索增强生成引擎
- get_embeddings: 获取Embedding模型
- IncrementalUpdater: 增量更新器（基于watchdog文件变化检测）
- KnowledgeBaseWatcher: 文件监听服务（自动检测变化）
- DocumentService: 文档上传与管理服务
- start_all_file_watchers: 启动所有知识库的文件监听
- stop_all_file_watchers: 停止所有文件监听
"""

from .data_cleaning import DataCleaner
from .engine import RAGEngine
from .embedding import get_embeddings
from .incremental_update import IncrementalUpdater, create_incremental_updater
from .file_watcher import (
    KnowledgeBaseWatcher,
    create_knowledge_base_watcher,
    start_all_file_watchers,
    stop_all_file_watchers,
)
from .document_service import (
    DocumentService,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    get_document_service,
)

__all__ = [
    "DataCleaner",
    "RAGEngine",
    "get_embeddings",
    "IncrementalUpdater",
    "create_incremental_updater",
    "KnowledgeBaseWatcher",
    "create_knowledge_base_watcher",
    "start_all_file_watchers",
    "stop_all_file_watchers",
    "DocumentService",
    "DocumentUploadResponse",
    "DocumentListResponse",
    "DocumentDeleteResponse",
    "DocumentInfo",
    "get_document_service",
]
"""
RAG 服务

【功能】：
1. 文档上传与处理
2. 文档列表查询
3. 文档删除
4. RAG 索引重建

【设计原则】：
1. 增量更新：支持文档增量更新
2. 质量控制：文档质量评分
3. 错误处理：统一的异常处理
"""

import logging
import os
from typing import Optional

from fastapi import HTTPException

from src.config import KNOWLEDGE_BASE_DIR
from src.rag import (
    get_document_service,
    RAGEngine,
    DocumentService,
)

logger = logging.getLogger(__name__)


class RagService:
    """RAG 服务类"""

    def __init__(self):
        self.doc_service: Optional[DocumentService] = None

    def _get_doc_service(self) -> DocumentService:
        """获取文档服务实例"""
        if self.doc_service is None:
            self.doc_service = get_document_service()
        return self.doc_service

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        knowledge_base_name: str = "knowledge_base_agent",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        上传文档到知识库
        
        Args:
            file_content: 文件内容
            filename: 文件名
            knowledge_base_name: 知识库名称
            chunk_size: 分块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            DocumentUploadResponse: 上传结果
        """
        doc_service = self._get_doc_service()
        result = await doc_service.upload_document(
            file_content=file_content,
            filename=filename,
            knowledge_base_name=knowledge_base_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return result

    def list_documents(self, knowledge_base_name: str = "knowledge_base_agent"):
        """
        获取知识库中的文档列表
        
        Args:
            knowledge_base_name: 知识库名称
            
        Returns:
            DocumentListResponse: 文档列表
        """
        doc_service = self._get_doc_service()
        return doc_service.list_documents(knowledge_base_name)

    def delete_document(self, knowledge_base_name: str, doc_name: str):
        """
        删除知识库中的文档
        
        Args:
            knowledge_base_name: 知识库名称
            doc_name: 文档名称
            
        Returns:
            DocumentDeleteResponse: 删除结果
        """
        doc_service = self._get_doc_service()
        return doc_service.delete_document(knowledge_base_name, doc_name)

    def rebuild_index(self, knowledge_base_name: str = "knowledge_base_agent"):
        """
        重建 RAG 索引
        
        Args:
            knowledge_base_name: 知识库名称
            
        Returns:
            dict: 重建结果
        """
        knowledge_base_path = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)

        if not os.path.exists(knowledge_base_path):
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_name}")

        engine = RAGEngine(knowledge_dir=knowledge_base_path)

        return {
            "success": True,
            "message": "RAG索引重建成功",
            "knowledge_base": knowledge_base_name
        }

    def get_knowledge_base_list(self):
        """
        获取所有知识库列表
        
        Returns:
            list: 知识库列表
        """
        try:
            if not os.path.exists(KNOWLEDGE_BASE_DIR):
                return {"knowledge_bases": [], "total": 0}
            
            knowledge_bases = []
            for entry in os.listdir(KNOWLEDGE_BASE_DIR):
                entry_path = os.path.join(KNOWLEDGE_BASE_DIR, entry)
                if os.path.isdir(entry_path):
                    knowledge_bases.append({
                        "name": entry,
                        "path": entry_path,
                        "document_count": len([f for f in os.listdir(entry_path) if os.path.isfile(os.path.join(entry_path, f))])
                    })
            
            return {"knowledge_bases": knowledge_bases, "total": len(knowledge_bases)}
        except Exception as e:
            logger.error(f"获取知识库列表失败: {e}")
            raise

"""
=== 文档上传与管理服务 ===

【功能】：
1. 文档上传到知识库
2. 文档列表获取
3. 文档删除
4. 文档清洗处理

【设计】：
- 将文档上传逻辑从 API 层分离，提高可维护性
- 支持多种文件格式的自动识别和处理
- 集成 DataCleaner 实现文档质量控制
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.rag.data_cleaning import DataCleaner

logger = logging.getLogger(__name__)


# ============================================================
# 响应模型
# ============================================================
class DocumentUploadResponse(BaseModel):
    """文档上传响应体"""
    success: bool = Field(..., description="是否上传成功")
    message: str = Field(..., description="响应消息")
    file_name: str = Field(..., description="上传的文件名")
    saved_path: Optional[str] = Field(None, description="保存到知识库的路径")
    quality_score: float = Field(0.0, description="文档质量评分")
    duplicate_detected: bool = Field(False, description="是否检测到重复")
    chunk_count: int = Field(0, description="分块数量")
    metadata: Optional[dict] = Field(None, description="文档元数据")


class DocumentListResponse(BaseModel):
    """文档列表响应体"""
    documents: list = Field(..., description="文档列表")
    total: int = Field(..., description="文档总数")


class DocumentDeleteResponse(BaseModel):
    """文档删除响应体"""
    success: bool = Field(..., description="是否删除成功")
    message: str = Field(..., description="响应消息")


class DocumentInfo(BaseModel):
    """文档信息"""
    name: str = Field(..., description="文件名")
    path: str = Field(..., description="文件路径")
    size: int = Field(..., description="文件大小（字节）")
    modified_at: str = Field(..., description="修改时间（ISO格式）")


# ============================================================
# 文档服务类
# ============================================================
class DocumentService:
    """文档服务 - 处理文档上传、列表、删除等操作"""

    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls',
                          '.txt', '.md', '.html', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    def __init__(self, knowledge_base_dir: str):
        """初始化文档服务

        Args:
            knowledge_base_dir: 知识库根目录
        """
        self._knowledge_base_dir = knowledge_base_dir
        self._temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "temp"
        )
        os.makedirs(self._temp_dir, exist_ok=True)

    def _get_knowledge_base_path(self, knowledge_base_name: str) -> str:
        """获取指定知识库的路径"""
        return os.path.join(self._knowledge_base_dir, knowledge_base_name)

    def _validate_file_extension(self, filename: str) -> bool:
        """验证文件扩展名是否支持"""
        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in self.ALLOWED_EXTENSIONS

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        knowledge_base_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> DocumentUploadResponse:
        """上传文档到知识库

        Args:
            file_content: 文件内容（字节数据）
            filename: 文件名
            knowledge_base_name: 目标知识库名称
            chunk_size: 分块大小
            chunk_overlap: 块重叠大小

        Returns:
            DocumentUploadResponse: 上传结果
        """
        try:
            # 检查文件类型
            if not self._validate_file_extension(filename):
                return DocumentUploadResponse(
                    success=False,
                    message=f"不支持的文件类型: {os.path.splitext(filename)[1]}",
                    file_name=filename,
                    chunk_count=0
                )

            # 创建临时文件
            temp_path = os.path.join(self._temp_dir, filename)

            # 保存上传的文件
            with open(temp_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"文件已保存到临时目录: {temp_path}")

            # 初始化数据清洗器
            cleaner = DataCleaner()

            # 执行完整的7步处理流水线
            result, chunks = cleaner.process(
                temp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # 构建知识库路径
            knowledge_base_path = self._get_knowledge_base_path(knowledge_base_name)

            # 保存到知识库
            base_name = os.path.splitext(filename)[0]
            saved = cleaner.save_to_knowledge_base(result, chunks, knowledge_base_path, base_name)

            # 删除临时文件
            os.remove(temp_path)

            if saved:
                saved_path = os.path.join(knowledge_base_path, f"{base_name}.md")
                return DocumentUploadResponse(
                    success=True,
                    message="文档上传并清洗成功",
                    file_name=filename,
                    saved_path=saved_path,
                    quality_score=result.quality_score,
                    duplicate_detected=result.duplicate_detected,
                    chunk_count=len(chunks),
                    metadata=result.metadata
                )
            else:
                if result.duplicate_detected:
                    return DocumentUploadResponse(
                        success=False,
                        message="检测到重复文档，未保存",
                        file_name=filename,
                        duplicate_detected=True,
                        chunk_count=len(chunks),
                        metadata=result.metadata
                    )
                else:
                    return DocumentUploadResponse(
                        success=False,
                        message="文档保存失败",
                        file_name=filename,
                        chunk_count=len(chunks),
                        metadata=result.metadata
                    )

        except Exception as e:
            logger.error(f"文档上传失败: {e}", exc_info=True)
            return DocumentUploadResponse(
                success=False,
                message=f"文档处理失败: {str(e)}",
                file_name=filename,
                chunk_count=0
            )

    def list_documents(self, knowledge_base_name: str) -> DocumentListResponse:
        """获取知识库中的文档列表

        Args:
            knowledge_base_name: 知识库名称

        Returns:
            DocumentListResponse: 文档列表
        """
        try:
            knowledge_base_path = self._get_knowledge_base_path(knowledge_base_name)

            if not os.path.exists(knowledge_base_path):
                return DocumentListResponse(documents=[], total=0)

            documents = []
            for filename in os.listdir(knowledge_base_path):
                if filename.endswith('.md') and not filename.endswith('_metadata.json'):
                    file_path = os.path.join(knowledge_base_path, filename)
                    file_stat = os.stat(file_path)
                    documents.append(DocumentInfo(
                        name=filename,
                        path=file_path,
                        size=file_stat.st_size,
                        modified_at=datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    ))

            return DocumentListResponse(documents=documents, total=len(documents))

        except Exception as e:
            logger.error(f"获取文档列表失败: {e}", exc_info=True)
            return DocumentListResponse(documents=[], total=0)

    def delete_document(self, knowledge_base_name: str, filename: str) -> DocumentDeleteResponse:
        """删除知识库中的文档

        Args:
            knowledge_base_name: 知识库名称
            filename: 要删除的文件名

        Returns:
            DocumentDeleteResponse: 删除结果
        """
        try:
            knowledge_base_path = self._get_knowledge_base_path(knowledge_base_name)
            file_path = os.path.join(knowledge_base_path, filename)

            if not os.path.exists(file_path):
                return DocumentDeleteResponse(
                    success=False,
                    message=f"文件不存在: {filename}"
                )

            # 删除主文件
            os.remove(file_path)

            # 同时删除元数据文件（如果存在）
            metadata_path = f"{file_path}_metadata.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            return DocumentDeleteResponse(
                success=True,
                message=f"文档已删除: {filename}"
            )

        except Exception as e:
            logger.error(f"删除文档失败: {e}", exc_info=True)
            return DocumentDeleteResponse(
                success=False,
                message=f"删除失败: {str(e)}"
            )


# ============================================================
# 全局单例
# ============================================================
_global_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """获取文档服务单例

    Returns:
        DocumentService: 文档服务实例
    """
    global _global_document_service

    if _global_document_service is None:
        from src.config import KNOWLEDGE_BASE_DIR
        _global_document_service = DocumentService(KNOWLEDGE_BASE_DIR)

    return _global_document_service


"""
单元测试：RAG 引擎模块

测试覆盖：
1. 文档加载与分块
2. 向量数据库构建
3. 混合检索器
4. Rerank 重排序
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.rag.engine import (
    _load_documents,
    _split_documents,
    RAGEngine,
)
from langchain_core.documents import Document


class TestDocumentLoading:
    """测试文档加载"""

    def test_load_documents_empty_dir(self, tmp_path):
        """测试空目录"""
        docs = _load_documents(str(tmp_path))
        assert len(docs) == 0

    def test_load_documents_with_files(self, tmp_path):
        """测试加载markdown文件"""
        # 创建测试文件
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test content.")
        
        docs = _load_documents(str(tmp_path))
        
        assert len(docs) == 1
        assert "Test Document" in docs[0].page_content
        assert docs[0].metadata["source_file"] == "test.md"

    def test_load_documents_nonexistent_dir(self):
        """测试不存在的目录"""
        docs = _load_documents("/nonexistent/path")
        assert len(docs) == 0


class TestDocumentSplitting:
    """测试文档分块"""

    def test_split_documents(self):
        """测试分块功能"""
        doc = Document(
            page_content="\n\n".join(["Paragraph " + str(i) for i in range(10)]),
            metadata={"source_file": "test.md"}
        )
        
        chunks = _split_documents([doc])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert len(chunk.page_content) <= 500  # CHUNK_SIZE

    def test_split_preserves_metadata(self):
        """测试元数据保留"""
        doc = Document(
            page_content="Test content",
            metadata={"source_file": "test.md", "custom_key": "custom_value"}
        )
        
        chunks = _split_documents([doc])
        
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.md"
            assert chunk.metadata["custom_key"] == "custom_value"


class TestRAGEngine:
    """测试RAG引擎"""

    @patch("src.rag.engine.get_embeddings")
    @patch("src.rag.engine._build_vectorstore")
    @patch("src.rag.engine._build_hybrid_retriever")
    def test_engine_initialization(self, mock_retriever, mock_vectorstore, mock_embeddings):
        """测试引擎初始化"""
        mock_embeddings.return_value = Mock()
        mock_vectorstore.return_value = Mock()
        mock_retriever.return_value = Mock()
        
        engine = RAGEngine()
        
        assert engine._embeddings is not None
        assert engine._vectorstore is not None
        assert engine._retriever is not None

    @patch("src.rag.engine.get_embeddings")
    @patch("src.rag.engine._build_vectorstore")
    @patch("src.rag.engine._build_hybrid_retriever")
    def test_query_with_results(self, mock_retriever, mock_vectorstore, mock_embeddings):
        """测试查询有结果"""
        mock_embeddings.return_value = Mock()
        mock_vectorstore.return_value = Mock()
        
        mock_docs = [
            Document(page_content="Relevant content", metadata={"source_file": "doc1.md", "relevance_score": 0.9})
        ]
        mock_retriever.return_value.invoke.return_value = mock_docs
        
        engine = RAGEngine()
        engine._retriever = mock_retriever.return_value
        
        result = engine.query("test query")
        
        assert result["found"] is True
        assert result["answer_context"] == "Relevant content"
        assert result["sources"] == ["doc1.md"]

    @patch("src.rag.engine.get_embeddings")
    @patch("src.rag.engine._build_vectorstore")
    @patch("src.rag.engine._build_hybrid_retriever")
    def test_query_no_results(self, mock_retriever, mock_vectorstore, mock_embeddings):
        """测试查询无结果"""
        mock_embeddings.return_value = Mock()
        mock_vectorstore.return_value = Mock()
        mock_retriever.return_value.invoke.return_value = []
        
        engine = RAGEngine()
        engine._retriever = mock_retriever.return_value
        
        result = engine.query("test query")
        
        assert result["found"] is False
        assert result["answer_context"] == ""
        assert result["sources"] == []

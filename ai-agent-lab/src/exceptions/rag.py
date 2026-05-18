"""
RAG 相关异常类

【功能】：
1. RAG 基础异常
2. 文档未找到异常
3. 文档解析异常
4. 索引异常
5. 嵌入异常
"""

from .base import AgentError


class RAGException(AgentError):
    """RAG 基础异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, "RAG_ERROR", original_exception)


class DocumentNotFoundError(RAGException):
    """文档未找到异常"""
    
    def __init__(self, document_name: str, knowledge_base: str = None):
        message = f"文档未找到: {document_name}"
        if knowledge_base:
            message += f" (知识库: {knowledge_base})"
        super().__init__(message)
        self.error_code = "DOCUMENT_NOT_FOUND"
        self.document_name = document_name
        self.knowledge_base = knowledge_base


class DocumentParseError(RAGException):
    """文档解析异常"""
    
    def __init__(self, document_name: str, original_exception: Exception = None):
        message = f"文档解析失败: {document_name}"
        super().__init__(message, original_exception)
        self.error_code = "DOCUMENT_PARSE_ERROR"
        self.document_name = document_name


class IndexError(RAGException):
    """索引异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, original_exception)
        self.error_code = "INDEX_ERROR"


class EmbeddingError(RAGException):
    """嵌入异常"""
    
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message, original_exception)
        self.error_code = "EMBEDDING_ERROR"

"""
RAG 路由模块

【功能】：
1. 文档上传到知识库
2. 文档列表查询
3. 文档删除
4. RAG 索引重建

【接口列表】：
- POST /rag/upload - 上传文档到知识库
- GET /rag/documents - 获取文档列表
- DELETE /rag/documents/{doc_name} - 删除文档
- POST /rag/rebuild - 重建 RAG 索引
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from src.services.rag_service import RagService

router = APIRouter(prefix="/rag")


@router.post("/upload")
async def upload_document_to_knowledge_base(
    file: UploadFile = File(...),
    knowledge_base_name: str = "knowledge_base_agent",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    上传文档到知识库

    【支持的文件类型】
    - PDF: .pdf
    - Word: .docx, .doc
    - Excel: .xlsx, .xls
    - Text: .txt
    - Markdown: .md
    - HTML: .html
    - Image: .png, .jpg, .jpeg, .bmp, .tiff（需要OCR支持）

    Args:
        file: 上传的文件
        knowledge_base_name: 目标知识库名称，默认为 knowledge_base_agent
        chunk_size: 分块大小，默认500字符
        chunk_overlap: 块重叠大小，默认50字符

    Returns:
        DocumentUploadResponse: 上传结果，包含质量评分、分块数等信息
    """
    rag_service = RagService()
    
    try:
        result = await rag_service.upload_document(
            file_content=await file.read(),
            filename=file.filename,
            knowledge_base_name=knowledge_base_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if result.success:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/documents")
async def list_documents(knowledge_base_name: str = "knowledge_base_agent"):
    """
    获取知识库中的文档列表

    Args:
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        DocumentListResponse: 文档列表
    """
    rag_service = RagService()
    try:
        return rag_service.list_documents(knowledge_base_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, knowledge_base_name: str = "knowledge_base_agent"):
    """
    删除知识库中的文档

    Args:
        doc_name: 文档名称（含扩展名）
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        DocumentDeleteResponse: 删除结果
    """
    rag_service = RagService()
    try:
        result = rag_service.delete_document(knowledge_base_name, doc_name)

        if result.success:
            return result
        else:
            raise HTTPException(status_code=404, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@router.post("/rebuild")
async def rebuild_rag_index(knowledge_base_name: str = "knowledge_base_agent"):
    """
    重建指定知识库的 RAG 索引

    Args:
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        dict: 重建结果
    """
    rag_service = RagService()
    try:
        return rag_service.rebuild_index(knowledge_base_name)
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG索引重建失败: {str(e)}")

"""
=== FastAPI 服务入口 ===

【知识点】FastAPI 服务架构：
1. 使用 Lifespan 管理应用生命周期（启动/关闭）
2. CORS 中间件配置，支持跨域请求
3. 路由模块化，按功能划分
4. 统一的错误处理和日志记录

【生产实践】企业级 FastAPI 项目结构：
- 主入口文件（server.py）：负责应用创建、生命周期、中间件配置
- 路由模块（routes/）：按功能划分 API 路由
- 服务模块（services/）：业务逻辑处理
- 数据模型（models/）：Pydantic 数据模型
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    MAX_ITERATIONS,
    KNOWLEDGE_BASE_DIR,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# 全局变量
# ============================================================
mcp_manager = None
app_instance: Optional[FastAPI] = None


# ============================================================
# 数据模型
# ============================================================
class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., description="用户消息")
    thread_id: Optional[str] = Field(None, description="会话线程 ID")


class ChatResponse(BaseModel):
    """聊天响应体"""
    response: str = Field(..., description="AI 响应")
    thread_id: str = Field(..., description="会话线程 ID")
    sources: list = Field(default_factory=list, description="引用的知识库来源")
    found_in_kb: bool = Field(False, description="是否在知识库中找到相关信息")


class SessionResponse(BaseModel):
    """新建会话响应体"""
    thread_id: str = Field(..., description="新创建的会话 ID")


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期钩子

    【知识点】FastAPI 的 lifespan 机制：
    - 在应用启动前执行异步初始化（如连接数据库、加载模型）
    - 在应用关闭时执行异步清理（如关闭连接、释放资源）
    - 使用 asynccontextmanager 确保资源正确释放

    【生产实践】启动检查清单：
    1. 初始化日志系统
    2. 初始化 MCP 管理器（可选）
    3. 启动知识库文件监听服务
    4. 检查外部服务状态（如 Prometheus）
    """
    global mcp_manager

    logger.info("=" * 60)
    logger.info("Agent API 正在启动...")
    logger.info("=" * 60)

    # 1. 初始化 MCP 管理器（可选功能）
    PROMETHEUS_ENABLED = False
    try:
        from src.tools import get_mcp_manager
        from src.metrics import PROMETHEUS_AVAILABLE
        mcp_manager = await get_mcp_manager()
        if mcp_manager:
            logger.info("✅ MCP 管理器初始化成功")
        else:
            logger.warning("⚠️ MCP 管理器初始化失败，功能可能不可用")
    except Exception as e:
        logger.warning(f"MCP 管理器初始化失败: {e}")
        logger.warning("MCP 功能可能不可用，但服务将继续运行")

    # 【生产实践】初始化领域专家系统（加载知识库）
    # 这会创建所有专家的 RAG 引擎并加载知识库内容到向量数据库
    logger.info("正在初始化领域专家系统...")
    try:
        from src.agents import initialize_experts
        await initialize_experts()
        logger.info("✅ 领域专家系统初始化成功")
    except Exception as e:
        logger.warning(f"领域专家系统初始化失败: {e}")
        logger.warning("专家功能可能不可用，但服务将继续运行")

    # 【生产实践】启动知识库文件监听服务
    # 当知识库文件发生变化时，自动触发增量更新
    logger.info("正在启动知识库文件监听服务...")
    try:
        from src.rag import start_all_file_watchers
        start_all_file_watchers()
        logger.info("知识库文件监听服务已启动")
    except Exception as e:
        logger.warning(f"文件监听服务启动失败: {e}")
        logger.warning("文件监听功能不可用，知识库更新需要手动触发")

    logger.info("Agent API 启动成功。")
    logger.info("=" * 60)
    logger.info("服务已就绪，可通过以下地址访问:")
    logger.info("  - API 文档: http://localhost:8000/docs")
    logger.info("  - 健康检查: http://localhost:8000/health")
    logger.info("  - MCP 状态: http://localhost:8000/mcp/status")
    logger.info("=" * 60)

    # 检查 Prometheus 是否可用
    if PROMETHEUS_AVAILABLE:
        logger.info("Prometheus 监控已启用")
    else:
        logger.warning("Prometheus 监控未启用，请安装 prometheus-client")

    yield

    # 【生产实践】优雅关闭 — 释放所有资源
    logger.info("Agent API 正在关闭，释放所有资源...")

    # 1. 停止知识库文件监听服务
    try:
        from src.rag import stop_all_file_watchers
        logger.info("正在停止知识库文件监听服务...")
        stop_all_file_watchers()
        logger.info("知识库文件监听服务已停止")
    except Exception as e:
        logger.warning(f"停止文件监听服务失败: {e}")

    # 2. 关闭 MCP 管理器
    try:
        from src.tools import close_mcp_manager
        logger.info("正在关闭 MCP 管理器...")
        await close_mcp_manager()
        logger.info("MCP 管理器已关闭")
    except Exception as e:
        logger.warning(f"关闭 MCP 管理器失败: {e}")

    # 3. 关闭内存池
    try:
        from src.memory import close_async_pool
        logger.info("正在关闭记忆服务...")
        await close_async_pool()
        logger.info("记忆服务已关闭")
    except Exception as e:
        logger.warning(f"关闭记忆服务失败: {e}")

    logger.info("Agent API 已关闭")


# ============================================================
# 应用创建
# ============================================================
app = FastAPI(
    title="AI Agent API",
    description="基于 LangGraph 的多专家 Agent 系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务配置
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 - 实时返回AI响应"""
    from src.agents.workflow import get_async_agent
    from src.memory.manager import get_memory_manager

    try:
        thread_id = request.thread_id
        if not thread_id:
            thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        memory_manager = get_memory_manager()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "model": "deepseek" if DEEPSEEK_API_KEY else "ollama",
            }
        }

        # 获取异步 Agent 实例
        agent_executor = await get_async_agent()

        async def stream_generator():
            async for event in agent_executor.astream_events(
                {"messages": [("user", request.message)]},
                config=config,
                version="v1"
            ):
                # 只处理 token 级别的输出
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {content}\n\n"
                elif event["event"] == "on_end":
                    # 流式结束标记
                    yield "data: [END]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"流式聊天接口出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session", response_model=SessionResponse)
async def create_session():
    """创建新会话"""
    import uuid
    thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    return SessionResponse(thread_id=thread_id)


# ============================================================
# 专家 Agent 接口
# ============================================================
@app.get("/experts")
async def list_experts():
    """获取所有专家 Agent 列表"""
    from src.agents import list_experts

    try:
        experts = list_experts()
        return {"experts": experts, "total": len(experts)}
    except Exception as e:
        logger.error(f"获取专家列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experts/{expert_name}")
async def get_expert_info(expert_name: str):
    """获取指定专家 Agent 信息"""
    from src.agents import get_expert

    try:
        expert = get_expert(expert_name)
        if not expert:
            raise HTTPException(status_code=404, detail=f"专家 {expert_name} 不存在")
        return expert.get_metadata()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取专家信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experts/{expert_name}/chat")
async def chat_with_expert(expert_name: str, request: ChatRequest):
    """与指定专家 Agent 聊天"""
    from src.agents import get_expert

    try:
        expert = get_expert(expert_name)
        if not expert:
            raise HTTPException(status_code=404, detail=f"专家 {expert_name} 不存在")

        result = await expert.process(
            query=request.message,
            config={"configurable": {"model": "deepseek" if DEEPSEEK_API_KEY else "ollama"}},
            context={"thread_id": request.thread_id}
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"专家聊天接口出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RAG 接口
# ============================================================
@app.post("/rag/upload")
async def upload_document_to_knowledge_base(
    file: UploadFile = File(...),
    knowledge_base_name: str = "knowledge_base_agent",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    上传文档到知识库

    【知识点】完整的文档处理流程：
    1. 接收上传的文件（支持PDF/Word/Excel/TXT/MD/HTML/图片）
    2. 使用 DataCleaner 执行7步清洗流水线
    3. 将清洗后的Markdown文档保存到知识库目录
    4. 支持自定义知识库名称和分块参数

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
    from src.rag import get_document_service

    try:
        doc_service = get_document_service()
        result = await doc_service.upload_document(
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
        logger.error(f"文档上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.get("/rag/documents")
async def list_documents(knowledge_base_name: str = "knowledge_base_agent"):
    """
    获取知识库中的文档列表

    Args:
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        DocumentListResponse: 文档列表
    """
    from src.rag import get_document_service

    try:
        doc_service = get_document_service()
        return doc_service.list_documents(knowledge_base_name)
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.delete("/rag/documents/{doc_name}")
async def delete_document(doc_name: str, knowledge_base_name: str = "knowledge_base_agent"):
    """
    删除知识库中的文档

    Args:
        doc_name: 文档名称（含扩展名）
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        DocumentDeleteResponse: 删除结果
    """
    from src.rag import get_document_service

    try:
        doc_service = get_document_service()
        result = doc_service.delete_document(knowledge_base_name, doc_name)

        if result.success:
            return result
        else:
            raise HTTPException(status_code=404, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@app.post("/rag/rebuild")
async def rebuild_rag_index(knowledge_base_name: str = "knowledge_base_agent"):
    """
    重建指定知识库的 RAG 索引

    Args:
        knowledge_base_name: 知识库名称，默认为 knowledge_base_agent

    Returns:
        dict: 重建结果
    """
    from src.rag import RAGEngine
    from src.config import KNOWLEDGE_BASE_DIR

    try:
        knowledge_base_path = os.path.join(KNOWLEDGE_BASE_DIR, knowledge_base_name)

        if not os.path.exists(knowledge_base_path):
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_name}")

        engine = RAGEngine(knowledge_dir=knowledge_base_path)

        return {
            "success": True,
            "message": "RAG索引重建成功",
            "knowledge_base": knowledge_base_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG索引重建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG索引重建失败: {str(e)}")


# ============================================================
# 记忆接口
# ============================================================
@app.get("/memory/{thread_id}")
async def get_memory(thread_id: str):
    """获取指定会话的记忆"""
    from src.memory.manager import get_memory_manager

    try:
        memory_manager = get_memory_manager()
        memory = await memory_manager.get_memory(thread_id)
        return {"thread_id": thread_id, "memory": memory}
    except Exception as e:
        logger.error(f"获取记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{thread_id}")
async def clear_memory(thread_id: str):
    """清除指定会话的记忆"""
    from src.memory.manager import get_memory_manager

    try:
        memory_manager = get_memory_manager()
        await memory_manager.clear_memory(thread_id)
        return {"success": True, "message": f"记忆已清除: {thread_id}"}
    except Exception as e:
        logger.error(f"清除记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LLM 统计接口
# ============================================================
@app.get("/llm/stats")
async def get_llm_stats():
    """获取 LLM 调用统计"""
    from src.llm.gateway import get_call_stats

    try:
        stats = get_call_stats()
        return stats
    except Exception as e:
        logger.error(f"获取LLM统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MCP 接口
# ============================================================
@app.get("/mcp/status")
async def get_mcp_status():
    """获取 MCP 服务状态"""
    global mcp_manager

    if mcp_manager is None:
        return {
            "available": False,
            "message": "MCP 管理器未初始化"
        }

    try:
        return {
            "available": True,
            "message": "MCP 管理器运行中",
            "server_count": len(mcp_manager._servers) if hasattr(mcp_manager, '_servers') else 0
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"MCP 状态检查失败: {str(e)}"
        }


@app.get("/mcp/servers")
async def list_mcp_servers():
    """获取已注册的 MCP 服务器列表"""
    global mcp_manager

    if mcp_manager is None:
        return {"servers": [], "total": 0}

    try:
        servers = mcp_manager.list_servers() if hasattr(mcp_manager, 'list_servers') else []
        return {"servers": servers, "total": len(servers)}
    except Exception as e:
        logger.error(f"获取MCP服务器列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/execute")
async def execute_mcp_tool(server_name: str, tool_name: str, arguments: dict = None):
    """执行 MCP 工具"""
    global mcp_manager

    if mcp_manager is None:
        raise HTTPException(status_code=503, detail="MCP 管理器未初始化")

    try:
        result = await mcp_manager.execute_tool(server_name, tool_name, arguments or {})
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"MCP工具执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
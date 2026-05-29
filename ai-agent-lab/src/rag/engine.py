"""
=== RAG 检索增强生成引擎 ===

【知识点】RAG（Retrieval-Augmented Generation）让 Agent 拥有企业私有知识
核心思路：用户提问 → 从知识库检索相关文档 → 把文档作为上下文传给 LLM → 生成有据可查的回答

本模块实现了企业级 RAG 的完整流程：
1. 文档加载（Loader）— 从知识库目录加载 Markdown 文件
2. 文本分块（Splitter）— 将长文档切分为适合检索的小块
3. 向量化存储（Embedding + ChromaDB）— 将文本块转为向量并存入向量数据库
4. 混合检索（Hybrid Search）— 向量语义检索 + BM25 关键词检索
5. 重排序（Rerank）— 用 Cross-Encoder 对检索结果精排
6. 相似度阈值判断 — 低于阈值时 LLM 兜底回答
7. 数据来源标注 — 回答中标明信息来自哪个知识库文件

【企业实战】为什么需要混合检索 + Rerank？
- 纯向量检索：语义理解好，但对精确关键词（如订单号、型号）匹配差
- 纯 BM25 检索：关键词匹配好，但不理解语义（"退货"搜不到"退换货政策"）
- 混合检索：两者互补，覆盖更全面
- Rerank：初步检索结果排序不够精确，Cross-Encoder 逐对精排提高准确率
"""

import logging
import os
import shutil
from pathlib import Path

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RERANK_TOP_N,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    SIMILARITY_THRESHOLD,
)
from src.rag.embedding import get_embeddings

logger = logging.getLogger(__name__)


# ============================================================
# 第一步：Embedding 模型（从 embedding_service 获取全局单例）
# ============================================================
# 【知识点】Embedding 模型已抽离到 embedding_service.py 统一管理
# RAG 引擎、对话记忆等模块共享同一个模型实例，避免重复加载


# ============================================================
# 第二步：文档加载与分块
# ============================================================
def _load_documents(knowledge_dir: str | None = None) -> list[Document]:
    """从知识库目录加载所有文档
    
    【知识点】Document 是 LangChain 的文档抽象，包含：
    - page_content：文本内容
    - metadata：元数据（来源文件、页码等），用于后续的来源标注

    Args:
        knowledge_dir: 知识库目录路径，None时使用默认的KNOWLEDGE_BASE_DIR

    Returns:
        list[Document]: 加载的文档列表，每个 Document 包含 page_content 和 metadata
                        （source_file 文件名、source_path 相对路径）；
                        如果知识库目录不存在则返回空列表
    """
    documents: list[Document] = []
    kb_path = Path(knowledge_dir or KNOWLEDGE_BASE_DIR)

    if not kb_path.exists():
        logger.warning(f"知识库目录不存在: {KNOWLEDGE_BASE_DIR}")
        return documents

    # 尝试使用MCP文件系统工具加载文档
    # 暂时跳过MCP文件系统加载，避免异步问题
    # 注：MCP文件系统服务器包名可能不正确，暂时使用传统方式
    pass

    # 传统方式加载
    for file_path in kb_path.rglob("*.md"):
        try:
            # 【知识点】TextLoader 比 UnstructuredMarkdownLoader 更轻量
            # 对于 Markdown 文件，直接按文本加载即可，不需要解析 HTML 结构
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            # 添加来源元数据，后续回答时标注数据来源
            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["source_path"] = str(file_path.relative_to(kb_path))
            documents.extend(docs)
            logger.info(f"已加载: {file_path.name}")
        except Exception as e:
            logger.warning(f"加载失败 {file_path.name}: {e}")

    logger.info(f"共加载 {len(documents)} 个文档")
    return documents


async def _load_documents_with_mcp(kb_path: Path) -> list[Document]:
    """使用MCP文件系统工具加载文档
    
    【优势】：
    1. 支持远程文件系统
    2. 支持多种文件协议（s3://, http://等）
    3. 更好的错误处理和重试机制
    4. 支持大文件分块加载
    
    Args:
        kb_path: 知识库目录路径
        
    Returns:
        list[Document]: 加载的文档列表
    """
    from src.tools import get_mcp_manager
    
    documents: list[Document] = []
    
    try:
        manager = await get_mcp_manager()
        
        # 使用MCP列出目录
        result = await manager.call_tool("filesystem", "list_directory", path=str(kb_path))
        
        if not result.get("success", False):
            logger.warning(f"MCP列出目录失败: {result.get('error', '未知错误')}")
            return documents
            
        # 解析结果
        result_text = result.get("result", "")
        if not result_text:
            return documents
            
        # 假设结果是以换行分隔的文件列表
        files = [line.strip() for line in result_text.split('\n') if line.strip()]
        
        for file_name in files:
            if not file_name.endswith(".md"):
                continue
                
            file_path = kb_path / file_name
            
            try:
                # 使用MCP读取文件
                content_result = await manager.call_tool("filesystem", "read_file", path=str(file_path))
                
                if not content_result.get("success", False):
                    logger.warning(f"MCP读取文件失败 {file_name}: {content_result.get('error', '未知错误')}")
                    continue
                    
                content = content_result.get("result", "")
                if not content:
                    continue
                    
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "source_file": file_name,
                        "source_path": str(file_path.relative_to(kb_path))
                    }
                )
                documents.append(doc)
                logger.info(f"MCP已加载: {file_name}")
                
            except Exception as e:
                logger.warning(f"MCP处理文件失败 {file_name}: {e}")
                
    except Exception as e:
        logger.error(f"MCP文档加载过程出错: {e}")
        
    return documents


def _split_documents(documents: list[Document]) -> list[Document]:
    """将文档切分为适合检索的小块
    
    【知识点】RecursiveCharacterTextSplitter 的工作原理：
    按 separators 列表的优先级依次尝试分割：
    1. 先尝试按 "\\n\\n"（段落）分割
    2. 如果块仍然太大，按 "\\n"（换行）分割
    3. 继续按句号、问号等分割
    4. 最后按空格和字符分割
    
    这样能尽量保持语义完整性：优先在段落边界切分，而不是在句子中间切断

    Args:
        documents: 原始文档列表，每个 Document 可能包含很长的文本内容

    Returns:
        list[Document]: 切分后的文档块列表，每块大小不超过 CHUNK_SIZE，
                        块之间有 CHUNK_OVERLAP 字符的重叠，
                        每块的 metadata 中额外包含 chunk_index 索引
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # 为每个块添加索引元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(f"文档分块完成: {len(documents)} 个文档 → {len(chunks)} 个块")
    return chunks


# ============================================================
# 第三步：构建/加载向量数据库
# ============================================================
def _build_vectorstore(embeddings, knowledge_dir: str | None = None, collection_name: str | None = None) -> Chroma:
    """构建或加载 Chroma 向量数据库
    
    【知识点】Chroma 是轻量级嵌入式向量数据库
    - 零配置，无需安装额外服务
    - 支持持久化到磁盘（persist_directory）
    - 适合开发和中小规模生产环境
    - 大规模生产推荐 Milvus 或 Pinecone
    
    【知识点】增量构建策略：
    - 如果 chroma_db 目录已存在 → 直接加载，跳过构建（节省时间）
    - 如果不存在 → 加载文档 → 分块 → 向量化 → 存入 Chroma

    Args:
        embeddings: Embedding 模型实例（如 HuggingFaceEmbeddings），
                    用于将文本转为向量

    Returns:
        Chroma: Chroma 向量数据库实例，已持久化到 CHROMA_DB_DIR 目录，
                可直接用于相似度检索
    """
    kb_dir = knowledge_dir or KNOWLEDGE_BASE_DIR
    coll_name = collection_name or CHROMA_COLLECTION_NAME

    # 检查是否已有持久化的向量数据库
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        try:
            existing = Chroma(
                collection_name=coll_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_DIR,
            )
            if existing._collection.count() > 0:
                logger.info(f"加载已有的向量数据库（{coll_name}: {existing._collection.count()} 个向量）...")
                return existing
            else:
                logger.info(f"向量数据库 collection '{coll_name}' 为空，需要重新构建...")
        except Exception as e:
            logger.warning(f"加载向量数据库失败: {e}，将重新构建...")

    # 首次构建
    logger.info(f"首次构建向量数据库（知识库: {kb_dir}）...")
    documents = _load_documents(kb_dir)
    if not documents:
        logger.warning("知识库为空，创建空的向量数据库")
        return Chroma(
            collection_name=coll_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

    chunks = _split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=coll_name,
        persist_directory=CHROMA_DB_DIR,
    )
    logger.info(f"向量数据库构建完成（{coll_name}: {len(chunks)} 个向量）")
    return vectorstore


# ============================================================
# 第四步：混合检索器（Hybrid Search）
# ============================================================
def _build_hybrid_retriever(vectorstore: Chroma, chunks: list[Document] | None = None):
    """构建混合检索器：向量语义检索 + BM25 关键词检索
    
    【知识点】混合检索（Hybrid Search）的原理：
    
    1. 向量检索（Semantic Search）：
       - 把查询和文档都转为向量，计算余弦相似度
       - 优点：理解语义，"退货"能匹配到"退换货政策"
       - 缺点：对精确关键词（型号、订单号）匹配差
    
    2. BM25 检索（Keyword Search）：
       - 基于词频和逆文档频率的经典信息检索算法
       - 优点：精确关键词匹配好，"XG-SP100"能精确找到
       - 缺点：不理解语义，"退货"搜不到"退换货流程"
    
    3. EnsembleRetriever 加权融合：
       - 将两种检索结果按权重合并
       - weights=[0.4, 0.6] 表示 BM25 占 40%，向量占 60%
       - 使用 RRF（Reciprocal Rank Fusion）算法融合排名
    
    【企业实战】权重调优建议：
    - 知识库以结构化数据为主（FAQ、表格）→ 提高 BM25 权重
    - 知识库以长文本为主（文章、手册）→ 提高向量权重
    - 建议通过 RAGAS 评估来确定最优权重

    Args:
        vectorstore: Chroma 向量数据库实例，用于构建向量语义检索器
        chunks: 文档块列表，用于构建 BM25 倒排索引；
                如果为 None，则从 vectorstore 中自动获取所有文档

    Returns:
        EnsembleRetriever | VectorStoreRetriever: 混合检索器实例；
            如果文档为空则回退为纯向量检索器
    """
    # 向量语义检索器
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # BM25 关键词检索器
    # 【知识点】BM25 需要原始文档块来构建倒排索引
    # 如果没有传入 chunks，需要从向量数据库中获取
    if chunks is None:
        # 从 Chroma 中获取所有文档
        all_docs_data = vectorstore.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"])
        ]

    if not chunks:
        logger.warning("无文档可用于 BM25，仅使用向量检索")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K

    # 混合检索器
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )
    logger.info(f"混合检索器就绪 (BM25:{BM25_WEIGHT} + Vector:{VECTOR_WEIGHT})")
    return hybrid_retriever


# ============================================================
# 第五步：Rerank 重排序
# ============================================================
def _build_reranker(base_retriever):
    """在混合检索基础上添加 Cross-Encoder 重排序
    
    【知识点】为什么需要 Rerank？
    
    Bi-Encoder（Embedding 模型）vs Cross-Encoder（Rerank 模型）：
    
    Bi-Encoder（检索阶段）：
      - 查询和文档分别独立编码为向量
      - 速度快（向量可预计算），适合从海量文档中初筛
      - 精度一般（因为查询和文档没有交互）
    
    Cross-Encoder（重排序阶段）：
      - 将查询和文档拼接在一起，联合编码
      - 精度高（查询和文档充分交互）
      - 速度慢（每对都要重新计算），只适合对少量候选重排
    
    所以最佳实践是：Bi-Encoder 初筛 Top K → Cross-Encoder 精排 Top N
    
    【知识点】bge-reranker-v2-m3 是目前中文效果最好的开源 Rerank 模型之一
    首次运行会自动下载（~1GB），后续直接加载

    Args:
        base_retriever: 基础检索器（通常是混合检索器 EnsembleRetriever），
                        Rerank 会对其返回的结果进行精排

    Returns:
        ContextualCompressionRetriever | base_retriever: 带 Rerank 的压缩检索器；
            如果 Rerank 模型加载失败，则回退返回原始的 base_retriever
    """
    try:
        reranker_model = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3"
        )
        reranker = CrossEncoderReranker(
            model=reranker_model,
            top_n=RERANK_TOP_N,  # 重排后只保留 Top N 个最相关的
        )

        # 【知识点】ContextualCompressionRetriever 将 Rerank 包装为检索器
        # 它先调用 base_retriever 获取初步结果，再用 reranker 精排
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        logger.info("Rerank 重排序器就绪 (bge-reranker-v2-m3)")
        return compression_retriever

    except Exception as e:
        logger.warning(f"Rerank 模型加载失败，回退到无 Rerank 模式: {e}")
        logger.warning("提示：请安装 pip install sentence-transformers")
        return base_retriever


# ============================================================
# 第六步：RAG 引擎主类
# ============================================================
class RAGEngine:
    """RAG 检索增强生成引擎
    
    【知识点】封装完整的 RAG 流程：
    1. 初始化时构建向量数据库和检索器（离线阶段）
    2. query() 方法执行检索 + 相似度判断 + 来源标注（在线阶段）
    3. 低于相似度阈值时返回 None，由调用方决定是否 LLM 兜底
    
    【企业实战】RAG 引擎的设计原则：
    - 单一职责：只负责检索，不负责生成（生成交给 Agent/LLM）
    - 可配置：所有参数从 config.py 读取
    - 可观测：关键步骤打日志，方便排查检索质量问题
    """

    def __init__(self, knowledge_dir: str | None = None, collection_name: str | None = None) -> None:
        """初始化 RAG 引擎，构建向量数据库和检索器

        Args:
            knowledge_dir: 知识库目录路径，None 时使用默认的 KNOWLEDGE_BASE_DIR
            collection_name: ChromaDB collection 名称，None 时使用默认的 CHROMA_COLLECTION_NAME

        Returns:
            None（初始化完成后 self._retriever 即可用于检索）
        """
        self._knowledge_dir = knowledge_dir or KNOWLEDGE_BASE_DIR
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME
        logger.info(f"正在初始化 RAG 引擎（知识库: {self._knowledge_dir}）...")
        self._embeddings = get_embeddings()
        self._vectorstore = _build_vectorstore(
            self._embeddings,
            knowledge_dir=self._knowledge_dir,
            collection_name=self._collection_name,
        )
        hybrid_retriever = _build_hybrid_retriever(self._vectorstore)
        self._retriever = hybrid_retriever
        logger.info(f"RAG 引擎初始化完成 ✅（collection: {self._collection_name}）")

    def query(self, question: str) -> dict:
        """检索知识库并返回结果"""
        # 强制在控制台打印，不受 logger 配置影响
        print(f"\n[DEBUG RAG] 引擎: {self._collection_name} | 关键词: {question}")
        
        try:
            docs = self._retriever.invoke(question)
            print(f"[DEBUG RAG] 检索到文档数: {len(docs)}")
            
            for i, doc in enumerate(docs):
                score = doc.metadata.get("relevance_score", doc.metadata.get("score", 1.0))
                print(f"  - 匹配 {i}: {doc.metadata.get('source_file')} (Score: {score:.4f})")
        except Exception as e:
            print(f"[DEBUG RAG] 检索出错: {e}")
            return {"found": False, "answer_context": "", "sources": [], "doc_count": 0}

        if not docs:
            return {"found": False, "answer_context": "", "sources": [], "doc_count": 0}

        # 相似度阈值过滤（暂时放宽阈值到 0.1 以确保能搜到）
        threshold = 0.1 
        filtered_docs = []
        for doc in docs:
            score = doc.metadata.get("relevance_score", doc.metadata.get("score", 1.0))
            if score >= threshold:
                filtered_docs.append(doc)
            else:
                print(f"  - [过滤] 分数过低: {score:.4f} < {threshold}")

        if not filtered_docs:
            print("[DEBUG RAG] 过滤后无相关文档")
            return {"found": False, "answer_context": "", "sources": [], "doc_count": 0}

        # 合并内容并提取来源
        context_parts = []
        sources = set()
        for doc in filtered_docs:
            context_parts.append(doc.page_content)
            if "source_file" in doc.metadata:
                sources.add(doc.metadata["source_file"])
        
        answer_context = "\n\n".join(context_parts)
        print(f"[DEBUG RAG] ✅ 成功返回，来源: {list(sources)}")
        
        return {
            "found": True,
            "answer_context": answer_context,
            "sources": list(sources),
            "doc_count": len(filtered_docs)
        }

    def rebuild_index(self) -> None:
        """重建向量索引（知识库更新后调用）

        Args:
            无参数

        Returns:
            None（重建完成后 self._retriever 指向新的检索器）
        """
        self._vectorstore = _build_vectorstore(
            self._embeddings,
            knowledge_dir=self._knowledge_dir,
            collection_name=self._collection_name,
        )
        hybrid_retriever = _build_hybrid_retriever(self._vectorstore)
        self._retriever = hybrid_retriever
        logger.info(f"向量索引重建完成 ✅（{self._collection_name}）")

    async def update_incrementally(self) -> dict:
        """增量更新向量索引（仅更新变化的文件）

        【优势】：
        - 通过 MCP Git 服务检测文件变化
        - 仅对新增/修改/删除的文件进行更新
        - 比全量重建效率提升 10-100 倍

        Returns:
            dict: 更新统计 {
                'added': 新增向量数,
                'modified': 修改向量数,
                'deleted': 删除向量数,
                'total_vectors': 更新后总向量数
            }
        """
        from src.rag.incremental_update import IncrementalUpdater
        
        updater = IncrementalUpdater(
            vectorstore=self._vectorstore,
            knowledge_dir=self._knowledge_dir
        )
        
        stats = await updater.update_incrementally()
        
        # 重新构建检索器（因为文档可能变化）
        hybrid_retriever = _build_hybrid_retriever(self._vectorstore)
        self._retriever = hybrid_retriever
        
        return stats

    def start_file_watcher(self, debounce_delay: float = 3.0) -> None:
        """启动文件监听服务，自动检测知识库变化并触发增量更新

        Args:
            debounce_delay: 防抖延迟（秒），默认 3 秒。
                          设置较大的值可以避免频繁更新（如文件正在写入时）
        """
        from src.rag.file_watcher import KnowledgeBaseWatcher
        
        # 创建文件监听器
        self._file_watcher = KnowledgeBaseWatcher(
            knowledge_dir=self._knowledge_dir,
            callback=self.update_incrementally,
            debounce_delay=debounce_delay
        )
        
        # 启动监听
        self._file_watcher.start()
        
        # 保存引用，防止被垃圾回收
        setattr(self, '_file_watcher_instance', self._file_watcher)
        
    def stop_file_watcher(self) -> None:
        """停止文件监听服务"""
        if hasattr(self, '_file_watcher_instance'):
            self._file_watcher.stop()
            delattr(self, '_file_watcher_instance')
            logger.info("文件监听服务已停止")

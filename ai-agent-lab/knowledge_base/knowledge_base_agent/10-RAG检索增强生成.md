# 06 - RAG 检索增强生成

> RAG 让 Agent 拥有企业私有知识，不再"一本正经地胡说八道"。这是企业级 Agent 最核心的能力之一。

---

## 一、为什么需要 RAG？

### 1.1 LLM 的致命缺陷

LLM 的知识有截止日期，而且不知道你公司的内部信息：
- 不知道你公司的产品手册
- 不知道你公司的规章制度
- 不知道最新的行业政策
- 会"幻觉"——自信地编造不存在的信息

### 1.2 现实类比：RAG 就像开卷考试

**没有 RAG 的 Agent**：闭卷考试，只能靠记忆（训练数据）回答，记不住就瞎编。

**有 RAG 的 Agent**：开卷考试，遇到不确定的问题先翻书（检索知识库），找到相关内容后再回答。答案有据可查，不会瞎编。

### 1.3 真实企业场景

**场景：金蝶 ERP 智能客服**

用户问："金蝶云星空的应收账款模块怎么做账龄分析？"

- 没有 RAG：LLM 可能给出通用的账龄分析方法，但不知道金蝶云星空的具体操作步骤
- 有 RAG：从金蝶产品文档知识库中检索到相关操作手册，给出准确的步骤指引

---

## 二、RAG 的完整流程

```
                    离线阶段（索引构建）
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 文档加载  │ →  │ 文本分块  │ →  │ 向量化   │ →  │ 存入向量DB│
│ (Loader) │    │ (Splitter)│    │(Embedding)│    │ (VectorDB)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘

                    在线阶段（检索生成）
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 用户提问  │ →  │ 向量检索  │ →  │ 重排序   │ →  │ LLM 生成  │
│          │    │(Retriever)│    │ (Rerank) │    │ (带上下文)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## 三、文档加载（Document Loading）

### 3.1 多格式文档加载

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

# PDF 文档
pdf_loader = PyPDFLoader("knowledge_base/product_manual.pdf")
pdf_docs = pdf_loader.load()

# Word 文档
docx_loader = Docx2txtLoader("knowledge_base/policy.docx")
docx_docs = docx_loader.load()

# Markdown 文档
md_loader = UnstructuredMarkdownLoader("knowledge_base/api_docs.md")
md_docs = md_loader.load()

# CSV 数据
csv_loader = CSVLoader("knowledge_base/faq.csv", encoding="utf-8")
csv_docs = csv_loader.load()

# 网页
web_loader = WebBaseLoader("https://docs.example.com/guide")
web_docs = web_loader.load()
```

### 3.2 企业级文档加载器（批量处理）

```python
from pathlib import Path
from langchain_core.documents import Document

class EnterpriseDocLoader:
    """企业级文档加载器：支持批量加载、元数据提取、增量更新"""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def __init__(self, knowledge_base_dir: str):
        self.base_dir = Path(knowledge_base_dir)

    def load_all(self) -> list[Document]:
        """加载目录下所有支持的文档"""
        documents = []
        for file_path in self.base_dir.rglob("*"):
            if file_path.suffix in self.LOADER_MAP:
                loader_cls = self.LOADER_MAP[file_path.suffix]
                try:
                    loader = loader_cls(str(file_path))
                    docs = loader.load()
                    # 添加元数据
                    for doc in docs:
                        doc.metadata.update({
                            "source_file": file_path.name,
                            "file_type": file_path.suffix,
                            "category": file_path.parent.name,
                        })
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载失败 {file_path}: {e}")
        return documents
```

- `langchain_core.documents` — 定义了 `Document` 这个**数据结构**（就是一个类，包含 `page_content` 和 `metadata` 两个字段）
- `langchain_community.document_loaders` — 提供各种**加载器**，负责把文件读进来，转成上面那个 `Document` 对象

它们是上下游关系：

```
PDF/Word/Markdown 文件
        ↓
document_loaders（加载器，读文件）
        ↓
Document（数据结构，page_content + metadata）
        ↓
后续的切分、embedding、检索都操作这个 Document
```

打个比方：`Document` 是"快递箱"的标准规格，`document_loaders` 是"打包员"——打包员把各种东西装进标准快递箱里，后面的分拣、运输都只认快递箱这个标准格式。

你项目里 `from langchain_core.documents import Document` 就是引入这个数据结构，用来手动创建 Document 对象（比如从 vectorstore 取回数据后重新包装）。而实际从文件加载文档用的是 `document_loaders` 里的 Loader。

---

## 四、文本分块（Text Splitting）—— RAG 的关键环节

### 4.1 为什么分块如此重要？

**现实类比**：你在图书馆找资料。如果整本书是一个"块"，你检索到的是整本书——太多了，LLM 处理不了。如果每个字是一个"块"——太碎了，没有上下文。好的分块就像把书按"段落"或"章节"切分，每块都是一个完整的知识单元。

**分块太大**：检索精度低，Token 浪费，可能超过上下文窗口
**分块太小**：丢失上下文，语义不完整，检索结果碎片化
**分块刚好**：每块是一个完整的知识点，检索精准，LLM 能充分理解

### 4.2 常用分块策略

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)

# 策略一：递归字符分割（最通用）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # 每块最大字符数
    chunk_overlap=200,        # 块之间的重叠字符数（保持上下文连贯）
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],  # 中文优化
    length_function=len,
)

# 策略二：按 Markdown 标题分割（适合结构化文档）
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

# 策略三：按 Token 分割（精确控制 Token 数）
token_splitter = TokenTextSplitter(
    chunk_size=500,           # 每块最大 Token 数
    chunk_overlap=50,
)

# 实际使用
chunks = recursive_splitter.split_documents(documents)
```

### 4.3 分块最佳实践

```python
class SmartChunker:
    """智能分块器：根据文档类型选择最优分块策略"""

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunked = []
        for doc in documents:
            file_type = doc.metadata.get("file_type", "")
            if file_type == ".md":
                chunks = self._chunk_markdown(doc)
            elif file_type == ".csv":
                chunks = [doc]  # CSV 每行已经是独立记录
            else:
                chunks = self._chunk_recursive(doc)

            # 为每个块添加上下文元数据
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            chunked.extend(chunks)
        return chunked

    def _chunk_recursive(self, doc: Document) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        return splitter.split_documents([doc])

    def _chunk_markdown(self, doc: Document) -> list[Document]:
        # 先按标题分，再按大小分
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        md_chunks = md_splitter.split_text(doc.page_content)

        # 对过长的块再次分割
        recursive = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        final_chunks = []
        for chunk in md_chunks:
            if len(chunk.page_content) > 800:
                final_chunks.extend(recursive.split_documents([chunk]))
            else:
                final_chunks.append(chunk)
        return final_chunks
```

---

## 五、向量化与存储

### 5.1 Embedding 模型选择

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 方案一：OpenAI Embedding（效果好，需要网络）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 方案二：本地 Embedding（离线可用，适合数据敏感场景）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",  # 中文效果好
    model_kwargs={"device": "cuda"},        # GPU 加速
)
```

### 5.2 向量数据库选型

| 数据库 | 适用场景 | 特点 |
|--------|----------|------|
| ChromaDB | 开发/小规模 | 轻量、嵌入式、零配置 |
| Pinecone | 云端生产 | 全托管、高可用、自动扩缩 |
| Milvus | 大规模生产 | 高性能、分布式、开源 |
| PostgreSQL+pgvector | 已有 PG 基础设施 | 复用现有数据库，运维成本低 |
| Weaviate | 多模态检索 | 支持文本+图片+音频 |

### 5.3 构建向量索引

```python
from langchain_community.vectorstores import Chroma

# 构建索引
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="enterprise_knowledge",
    persist_directory="./chroma_db",
)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",       # 相似度检索
    search_kwargs={"k": 5},         # 返回 Top 5
)

# 也可以用 MMR（最大边际相关性）减少冗余
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,              # 先检索 20 个，再从中选 5 个最多样化的
        "lambda_mult": 0.7,         # 多样性权重
    },
)
```

---

## 六、检索优化：从"能用"到"好用"

### 6.1 混合检索（Hybrid Search）

**问题**：纯向量检索对精确关键词匹配不够好。用户搜"ORD-12345"，向量检索可能找不到。

**解决方案**：向量检索（语义）+ BM25（关键词）混合。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 向量语义检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 混合检索（加权融合）
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # BM25 权重 40%，向量权重 60%
)
```

### 6.2 查询改写（Query Rewriting）

**问题**：用户的提问可能模糊、口语化，直接用来检索效果差。

```python
from langchain_core.prompts import ChatPromptTemplate

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个查询优化专家。将用户的口语化问题改写为更适合知识库检索的查询。
规则：
1. 提取核心关键词
2. 补充可能的同义词
3. 去除口语化表达
4. 如果问题包含多个子问题，拆分为多个查询"""),
    ("human", "{question}")
])

def rewrite_query(question: str) -> list[str]:
    """将用户问题改写为检索友好的查询"""
    result = llm.invoke(query_rewrite_prompt.format(question=question))
    # 返回改写后的查询列表
    return result.content.strip().split("\n")

# 示例
# 输入："那个什么应收的账龄怎么看啊"
# 输出：["应收账款账龄分析操作步骤", "应收账款账龄报表查看方法"]
```

### 6.3 重排序（Rerank）

**问题**：初步检索返回的 Top K 结果中，排序不一定准确。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 使用 Cross-Encoder 重排序
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=reranker_model, top_n=3)

# 包装成压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,  # 先混合检索，再重排序
)
```

---

## 七、RAG 与 Agent 的集成

### 7.1 RAG 作为 Agent 的工具

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """在企业知识库中搜索相关信息。适用于查询产品文档、操作手册、规章制度等。

    Args:
        query: 搜索查询，尽量使用关键词而非完整句子
    """
    docs = compression_retriever.invoke(query)
    if not docs:
        return "未找到相关信息"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "未知来源")
        results.append(f"[{i}] 来源: {source}\n{doc.page_content}")

    return "\n\n---\n\n".join(results)

# Agent 会在需要时自动调用知识库搜索
agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, ...],
    prompt="你是企业智能助手。回答问题时，优先从知识库中检索信息，确保回答准确有据。"
)
```

### 7.2 完整的 RAG Agent 流程

```python
class RAGAgentState(TypedDict):
    messages: Annotated[list, add]
    retrieved_docs: list[str]
    confidence: float | None

def retrieve(state: RAGAgentState) -> dict:
    """检索相关文档"""
    question = state["messages"][-1].content
    docs = compression_retriever.invoke(question)
    doc_texts = [doc.page_content for doc in docs]
    return {"retrieved_docs": doc_texts}

def generate(state: RAGAgentState) -> dict:
    """基于检索结果生成回答"""
    context = "\n\n".join(state["retrieved_docs"])
    prompt = f"""基于以下参考资料回答用户问题。如果参考资料中没有相关信息，请明确说明。

参考资料：
{context}

用户问题：{state["messages"][-1].content}

要求：
1. 回答必须基于参考资料，不要编造
2. 引用具体的来源
3. 如果信息不足，建议用户提供更多细节"""

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}

graph = StateGraph(RAGAgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
```

---

## 八、RAG 评估与优化

### 8.1 RAG 的关键指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| 检索召回率 | 相关文档是否被检索到 | 相关文档数 / 总相关文档数 |
| 检索精确率 | 检索到的文档是否相关 | 相关文档数 / 检索文档数 |
| 答案忠实度 | 回答是否基于检索内容 | LLM 评估或人工评估 |
| 答案相关性 | 回答是否切题 | LLM 评估或人工评估 |

### 8.2 常见问题与优化方向

| 问题 | 原因 | 优化方向 |
|------|------|----------|
| 检索不到相关文档 | 分块不合理/Embedding 质量差 | 优化分块策略，换更好的 Embedding 模型 |
| 检索到但不相关 | 语义理解偏差 | 加入 Rerank，使用混合检索 |
| 回答有幻觉 | LLM 没有严格基于检索内容 | 优化 Prompt，加入"不知道就说不知道"的约束 |
| 回答不完整 | 检索的 Top K 太少 | 增加 K 值，或使用查询改写扩展检索 |

---

## 九、RAG 2026 最新优化技术

### 9.1 Graph RAG（知识图谱增强检索）

Graph RAG 将文档转换为知识图谱，通过图结构理解实体和关系，显著提升复杂问题的回答质量：

```python
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# 1. 将文档转换为知识图谱
llm = ChatOpenAI(temperature=0, model="gpt-4o")
graph_transformer = LLMGraphTransformer(llm=llm)
graph_documents = graph_transformer.convert_to_graph_documents(documents)

# 2. 存储到图数据库
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)
graph.add_graph_documents(graph_documents)

# 3. 图检索 + 向量检索 混合
from langchain_community.chains import GraphCypherQAChain

graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# 混合检索：图查询 + 向量检索
def hybrid_graph_rag(query):
    # 图结构化查询
    graph_result = graph_chain.invoke({"query": query})
    # 向量语义检索
    vector_result = retriever.invoke(query)
    # 合并结果
    combined_context = f"结构化信息:\n{graph_result}\n\n语义信息:\n{vector_result}"
    return combined_context
```

### 9.2 ColBERT / ColBERTv2（迟交互式检索）

ColBERT 在检索阶段也进行轻量级的交互式匹配，大幅提升检索精度：

```python
from ragatouille import RAGPretrainedModel

# 使用 ColBERTv2
rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# 索引文档
rag.index(
    collection=[doc.page_content for doc in documents],
    index_name="enterprise_knowledge"
)

# 检索（ColBERT 比纯向量检索更精确）
results = rag.search(query="应收账款账龄分析", k=5)
```

### 9.3 RAG Fusion（多路检索融合）

RAG Fusion 通过生成多个查询变体，分别检索后用 Reciprocal Rank Fusion 合并结果：

```python
from langchain.retrievers import MergerRetriever
from langchain.prompts import FewShotPromptTemplate

# 1. 查询改写生成多个变体
query_rewrite_prompt = FewShotPromptTemplate(
    examples=[
        {"input": "怎么查订单？", "output": "订单查询方法\n如何查看订单\n订单状态查询"},
    ],
    input_variables=["input"],
    template="输入: {input}\n输出: {output}"
)

def generate_query_variations(original_query, num_variations=4):
    variations = llm.invoke(query_rewrite_prompt.format(input=original_query))
    return [original_query] + variations.content.strip().split("\n")

# 2. 每路查询分别检索
def multi_query_retrieval(query):
    variations = generate_query_variations(query)
    all_results = []
    for q in variations:
        results = retriever.invoke(q)
        all_results.extend(results)
    return all_results

# 3. Reciprocal Rank Fusion (RRF) 去重合并
def rrf_merge(results_list, k=60):
    doc_scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("source", str(doc))
            score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
    # 按分数排序返回
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc_id, doc in sorted_docs[:10]]
```

### 9.4 RAG 缓存（显著降低成本和延迟）

```python
from langchain.cache import InMemoryCache, RedisCache
import langchain

# 配置缓存
langchain.llm_cache = RedisCache(redis_=redis_client)

# 相似查询缓存（使用 Embedding 相似度匹配）
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import OpenAIEmbeddings

class CachedRetriever:
    def __init__(self, base_retriever, cache_embeddings, threshold=0.95):
        self.base_retriever = base_retriever
        self.cache = {}
        self.cache_embeddings = cache_embeddings
        self.threshold = threshold
    
    def get_from_cache(self, query):
        query_embedding = self.cache_embeddings.embed_query(query)
        for cached_query, (cached_embedding, cached_result) in self.cache.items():
            similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
            if similarity > self.threshold:
                return cached_result
        return None
    
    def invoke(self, query):
        cached = self.get_from_cache(query)
        if cached:
            return cached
        result = self.base_retriever.invoke(query)
        query_embedding = self.cache_embeddings.embed_query(query)
        self.cache[query] = (query_embedding, result)
        return result
```

### 9.5 RAG 评估自动化（RAGAS 框架）

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# 准备评估数据
eval_data = Dataset.from_dict({
    "question": ["什么是应收账款账龄分析？", "如何优化库存周转率？"],
    "answer": ["...", "..."],
    "contexts": [["..."], ["..."]],
    "ground_truth": ["...", "..."]
})

# 自动化评估
result = evaluate(
    dataset=eval_data,
    metrics=[
        faithfulness,      # 答案忠实度
        answer_relevancy,  # 答案相关性
        context_precision, # 上下文精确度
        context_recall     # 上下文召回率
    ]
)

print(result)
# 输出: {'faithfulness': 0.89, 'answer_relevancy': 0.92, ...}
```

### 9.6 自适应 RAG（Adaptive RAG）

根据查询复杂度动态选择检索策略：

```python
def adaptive_rag(query):
    # 1. 查询复杂度分析
    complexity = analyze_query_complexity(query)
    
    if complexity == "simple":
        # 简单查询：快速检索 + 直接回答
        docs = fast_retriever.invoke(query)
        return generate_simple_answer(query, docs)
    
    elif complexity == "medium":
        # 中等复杂度：混合检索 + Rerank
        docs = hybrid_retriever.invoke(query)
        reranked_docs = reranker.compress_documents(docs, query)
        return generate_answer(query, reranked_docs)
    
    else:  # complex
        # 复杂查询：查询改写 + RAG Fusion + 图检索
        query_variations = generate_query_variations(query)
        all_docs = []
        for q in query_variations:
            docs = hybrid_retriever.invoke(q)
            all_docs.extend(docs)
        
        # 加入图检索结果
        graph_result = graph_chain.invoke({"query": query})
        graph_docs = [Document(page_content=str(graph_result))]
        
        # RRF 合并
        merged_docs = rrf_merge([all_docs, graph_docs])
        return generate_detailed_answer(query, merged_docs)
```

---

## 十、本章面试要点

### 基础面试题

1. **RAG 的完整流程是什么？**
   → 文档加载 → 分块 → 向量化 → 存储 → 检索 → 重排序 → 生成

2. **分块策略怎么选？chunk_size 和 overlap 怎么定？**
   → 根据文档类型选策略；chunk_size 通常 500-1000 字符；overlap 通常 10-20%；需要实验调优

3. **纯向量检索有什么局限？怎么解决？**
   → 对精确关键词匹配差。用混合检索（向量 + BM25）解决

4. **什么是 Rerank？为什么需要它？**
   → 用 Cross-Encoder 对初步检索结果重新排序，提高精确率。因为 Bi-Encoder（Embedding）的排序不够精确

5. **RAG 的幻觉问题怎么解决？**
   → Prompt 约束 + 答案溯源 + 置信度评估 + 人工审核兜底

### 进阶面试题

6. **Graph RAG 相比传统 RAG 有什么优势？**
   → 能理解实体间的关系，支持复杂推理；适合回答需要多跳推理的问题；结构化查询更精确

7. **ColBERT 的工作原理是什么？相比纯向量检索有什么提升？**
   → ColBERT 使用迟交互式匹配，在检索阶段对查询和文档的 token 进行细粒度匹配；显著提升检索精度，特别是对复杂查询

8. **RAG Fusion 的核心思想是什么？RRF（Reciprocal Rank Fusion）怎么实现？**
   → 生成多个查询变体分别检索，用 RRF 合并结果；RRF 通过 1/(k+rank) 计算分数，k 通常取 60，避免单一查询的偏差

9. **如何设计 RAG 缓存策略？相似度阈值怎么设定？**
   → 精确匹配缓存 + 语义相似度缓存；阈值通过实验确定，通常 0.9-0.97 之间；平衡缓存命中率和正确性

10. **RAGAS 框架的四个核心指标是什么？分别如何计算？**
    → Faithfulness（答案忠实度，检查答案是否基于上下文）、Answer Relevancy（答案相关性，检查答案是否切题）、Context Precision（上下文精确度）、Context Recall（上下文召回率）

### 企业级实战面试题

11. **如何设计企业级 RAG 的知识更新机制？**
    → 增量更新（只更新变更文档）、版本管理、实时索引 + 批量索引结合、更新通知、一致性校验、回滚机制

12. **Adaptive RAG 如何根据查询复杂度动态选择策略？**
    → 查询分析（长度、实体数、问题类型）、规则或模型判断复杂度、简单→快速检索、中等→混合检索+Rerank、复杂→RAG Fusion+图检索

13. **RAG 系统如何评估和持续优化？**
    → 建立评估数据集、定期 RAGAS 自动评估、A/B 测试、用户反馈收集、Bad Case 分析、迭代优化（分块/Embedding/检索策略/Prompt）

14. **如何解决 RAG 的"检索到但不相关"问题？**
    → 混合检索、Rerank 重排序、查询改写、更好的 Embedding 模型、优化分块策略、用户反馈校正检索结果

15. **企业级 RAG 系统的成本优化有哪些方法？**
    → 缓存策略（相似查询复用结果）、模型路由（简单查询用小模型）、批量处理、Embedding 预计算、合理的检索策略、压缩优化

### 架构设计面试题

16. **请设计一个百万级文档的企业级 RAG 系统架构。**
    → 文档处理层（异步处理、队列解耦）→ 索引层（向量数据库分片、混合索引）→ 检索层（多路检索+RRF+Rerank）→ 服务层（API、缓存、限流）→ 监控层（可观测性、评估）→ 数据层（对象存储、向量库、关系库、缓存）

17. **如何设计 RAG 系统的多租户隔离方案？**
    → 租户数据隔离（按租户分库/分表/命名空间）、Embedding 按租户隔离、权限控制、配置隔离、资源配额、计费计量

18. **如何保障 RAG 系统的数据安全和合规？**
    → 数据加密（传输+存储）、访问控制、PII 脱敏、审计日志、合规检查、水印溯源、版本管理、灾难恢复

19. **RAG 如何与 Agent 系统深度集成？**
    → RAG 作为 Agent 工具、Agent 根据需要决定是否调用 RAG、检索结果作为上下文、Agent 可追问或多轮检索、检索结果置信度评估

20. **如何设计 RAG 系统的灰度发布和回滚机制？**
    → Prompt 版本管理、Embedding 模型版本、索引版本、流量切分（按用户/比例）、指标对比（RAGAS、用户满意度）、快速回滚能力

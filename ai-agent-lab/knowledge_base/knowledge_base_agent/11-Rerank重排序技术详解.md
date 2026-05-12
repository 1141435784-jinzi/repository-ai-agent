# Rerank 重排序技术详解

> 整理来源：基于 [BGE Reranker Cross-Encoder Reranking for RAG](https://markaicode.com/bge-reranker-cross-encoder-reranking-rag/)、[Advanced RAG Reranking](https://towardsdatascience.com/advanced-rag-retrieval-cross-encoders-reranking/)、[Reranking for RAG](https://webcoderspeed.com/blog/scaling/rag-reranking-production)、[Best Reranker Models for RAG 2026](https://docs.bswen.com/blog/2026-02-25-best-reranker-models) 归纳改写
> 最后更新：2026 年 4 月

---

## 一、为什么需要 Rerank

向量检索（Bi-Encoder）的核心问题：**相似度 ≠ 相关性**。

向量检索找到的是"语义空间中距离最近的文档"，但这不一定是"对回答问题最有帮助的文档"。

**现实例子**：你问"LangGraph 的 checkpoint 机制怎么用？"
- 向量检索可能返回：一篇讲 LangGraph 概述的文档（语义相近但不够具体）
- 你真正需要的：一篇详细讲 checkpoint API 用法的文档

Rerank 就是在初筛结果上做"精排"，把真正相关的文档排到前面。

**效果数据**：Cross-Encoder Rerank 通常能将检索准确率提升 15-40%。

---

## 二、两阶段检索架构

```
用户查询
    ↓
┌─────────────────────────────────┐
│  第一阶段：初筛（Retrieval）      │
│  BM25 + 向量检索 → Top-K 候选    │
│  速度快，覆盖面广                 │
│  K 通常取 10~50                  │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  第二阶段：精排（Rerank）         │
│  Cross-Encoder 对 K 个候选打分    │
│  速度慢但精度高                   │
│  返回 Top-N（N < K）             │
└──────────────┬──────────────────┘
               ↓
         最终 Top-N 结果 → 送入 LLM 生成回答
```

你项目中的实现：
- 第一阶段：`EnsembleRetriever`（BM25 40% + 向量 60%）→ Top-K
- 第二阶段：`CrossEncoderReranker` → Top-N

---

## 三、Cross-Encoder 工作原理

### 3.1 与 Bi-Encoder 的本质区别

**Bi-Encoder**（Embedding 模型）：
```
查询 → [Encoder] → 向量A ─┐
                            ├→ 计算距离 → 分数
文档 → [Encoder] → 向量B ─┘
```
查询和文档独立编码，无法捕捉两者之间的细粒度交互。

**Cross-Encoder**（Rerank 模型）：
```
[查询 + [SEP] + 文档] → [Encoder] → 直接输出相关性分数
```
查询和文档拼接后一起输入模型，模型内部的 Attention 机制能捕捉两者之间的每一个词级交互。

### 3.2 为什么 Cross-Encoder 更准

Bi-Encoder 把查询和文档压缩成固定维度的向量（如 768 维），信息必然有损失。Cross-Encoder 不做压缩，直接在原始 token 级别做交叉注意力，信息保留更完整。

代价是：每对（查询, 文档）都要过一遍完整的 Transformer 模型，所以只能用于少量候选的精排，不能用于大规模初筛。

---

## 四、主流 Rerank 模型对比

### 4.1 开源模型

| 模型 | 大小 | 语言 | 特点 |
|---|---|---|---|
| bge-reranker-base | ~1.1GB | 中英 | 速度快，基础精度 |
| bge-reranker-large | ~1.3GB | 中英 | 精度更高 |
| bge-reranker-v2-m3 | ~2.2GB | 多语言 | 当前最佳开源 Reranker 之一 |
| Jina Reranker v2 | ~1.5GB | 多语言 | 支持长文本 |
| ms-marco-MiniLM-L-6-v2 | ~80MB | 英文 | 极轻量，英文场景 |

### 4.2 API 服务

| 服务 | 特点 |
|---|---|
| Cohere Rerank 3.5 | 商业 API，精度高，支持多语言 |
| Voyage Rerank | 高精度，按调用量计费 |

### 4.3 选型建议

- **中文场景** → bge-reranker-large 或 bge-reranker-v2-m3
- **数据敏感** → 必须用开源本地模型
- **快速验证** → Cohere Rerank API
- **资源受限** → ms-marco-MiniLM（英文）或 bge-reranker-base（中文）

---

## 五、在 LangChain 中使用 Rerank

### 5.1 使用 CrossEncoderReranker

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 加载 Cross-Encoder 模型
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# 创建 Reranker
reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=3  # 精排后保留 Top-3
)

# 包装为压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever  # 初筛检索器
)
```

### 5.2 工作流程

1. 用户查询 → `hybrid_retriever` 初筛返回 Top-K（如 10 个）
2. Reranker 对这 10 个文档逐一与查询做 Cross-Encoder 打分
3. 按分数重新排序，返回 Top-N（如 3 个）
4. Top-3 文档作为上下文送入 LLM 生成回答

---

## 六、Rerank 的调优要点

### 6.1 初筛数量 K 的选择

| K 值 | 效果 |
|---|---|
| 太小（如 3） | Rerank 没有足够的候选可选，可能漏掉好文档 |
| 适中（如 10-20） | 推荐范围，兼顾召回率和 Rerank 速度 |
| 太大（如 100） | Rerank 速度慢，且大量不相关文档浪费计算 |

### 6.2 精排数量 N 的选择

| N 值 | 效果 |
|---|---|
| 太小（如 1） | 信息可能不够，LLM 缺乏足够上下文 |
| 适中（如 3-5） | 推荐范围，信息充足且不超 token 限制 |
| 太大（如 10+） | 可能引入噪音，且占用过多 token |

### 6.3 什么时候不需要 Rerank

- 知识库很小（<100 个文档），向量检索已经足够准确
- 查询都是精确关键词匹配（BM25 就够了）
- 对延迟要求极高（Rerank 增加 50-200ms）
- 初筛结果已经很好（通过评估确认）

---

## 七、Rerank 效果评估

### 7.1 关键指标

- **MRR@K（Mean Reciprocal Rank）**：正确答案在 Top-K 中排名的倒数平均值
- **NDCG@K**：考虑排名位置的综合相关性指标
- **Hit Rate@K**：Top-K 中包含正确答案的查询比例

### 7.2 A/B 对比方法

```
实验组 A：BM25 + 向量检索（无 Rerank）
实验组 B：BM25 + 向量检索 + Rerank

对比指标：MRR@3、Hit Rate@3、最终回答的 Faithfulness
```

通过 RAGAS 评估框架可以自动化这个对比过程。

---

## 八、性能优化

| 优化手段 | 说明 |
|---|---|
| GPU 加速 | Cross-Encoder 推理放到 GPU 上，速度提升 5-10 倍 |
| 批量推理 | 多个（查询, 文档）对一起推理，利用 GPU 并行 |
| 模型量化 | 用 INT8/FP16 量化减少模型大小和推理时间 |
| 缓存 | 对高频查询缓存 Rerank 结果 |
| 减少 K | 初筛返回更少的候选，减少 Rerank 计算量 |

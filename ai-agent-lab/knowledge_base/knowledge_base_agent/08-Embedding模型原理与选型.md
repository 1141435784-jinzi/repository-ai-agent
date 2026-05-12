# Embedding 模型原理与选型

> 项目使用模型：BAAI/bge-base-zh-v1.5（768 维，~400MB）
> 整理来源：基于 [HuggingFace BGE 模型页](https://huggingface.co/BAAI/bge-base-zh-v1.5)、[Best Open-Source Embedding Models](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)、[Mastering BGE Embeddings](https://sparkco.ai/blog/mastering-bge-embeddings-with-hugging-face-in-2025) 归纳改写
> 最后更新：2026 年 4 月

---

## 一、什么是 Embedding

Embedding（嵌入/向量化）是把文本转换为一组数字（向量）的过程。转换后，语义相近的文本在向量空间中距离更近。

**现实例子**：想象一个巨大的图书馆，每本书都有一个 GPS 坐标。内容相似的书被放在相近的位置——烹饪书在一个区域，编程书在另一个区域。Embedding 就是给每段文本分配这样一个"语义坐标"。

**为什么需要 Embedding**：计算机不能直接理解文字，但能计算数字之间的距离。把文本变成向量后，"找最相关的文档"就变成了"找距离最近的向量"——这是向量数据库和 RAG 的基础。

---

## 二、Bi-Encoder vs Cross-Encoder

这是理解 Embedding 和 Rerank 的关键区别。

### 2.1 Bi-Encoder（双编码器）

Embedding 模型就是 Bi-Encoder：查询和文档各自独立编码为向量，然后计算相似度。

```
查询 → [Encoder] → 查询向量 ─┐
                               ├→ 计算余弦相似度 → 相关性分数
文档 → [Encoder] → 文档向量 ─┘
```

**优点**：
- 文档向量可以预计算并存储，查询时只需计算一个查询向量
- 速度极快，适合从百万级文档中初筛
- 支持近似最近邻（ANN）加速检索

**缺点**：
- 查询和文档独立编码，无法捕捉两者之间的细粒度交互
- 精度不如 Cross-Encoder

### 2.2 Cross-Encoder（交叉编码器）

Rerank 模型就是 Cross-Encoder：把查询和文档拼在一起同时输入模型，直接输出相关性分数。

```
[查询 + 文档] → [Encoder] → 相关性分数
```

**优点**：
- 能捕捉查询和文档之间的细粒度语义交互
- 精度显著高于 Bi-Encoder

**缺点**：
- 每对（查询, 文档）都要过一遍模型，无法预计算
- 速度慢，不适合大规模初筛，只适合对少量候选做精排

### 2.3 两阶段检索架构

生产环境的标准做法是两者结合：

```
百万级文档 → Bi-Encoder 初筛 Top-K → Cross-Encoder 精排 Top-N → 最终结果
（快但粗）                          （慢但准）
```

你项目里的架构正是如此：BM25 + 向量检索（Bi-Encoder）初筛 → Rerank（Cross-Encoder）精排。

---

## 三、BGE 模型家族

BGE（BAAI General Embedding）是北京智源人工智能研究院（BAAI）开源的 Embedding 模型系列，在中文场景中表现优异。

### 3.1 BGE Embedding 模型对比

| 模型 | 维度 | 大小 | 适用场景 |
|---|---|---|---|
| bge-small-zh-v1.5 | 512 | ~95MB | 资源受限、快速原型 |
| bge-base-zh-v1.5 | 768 | ~400MB | **平衡之选（你项目在用）** |
| bge-large-zh-v1.5 | 1024 | ~1.3GB | 追求最高精度的生产环境 |
| bge-m3 | 1024 | ~2.2GB | 多语言、多粒度、多检索方式 |

### 3.2 你项目使用的 bge-base-zh-v1.5

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # 归一化
)
```

关键参数说明：
- `device="cpu"`：学习阶段用 CPU 足够，生产环境用 `"cuda"` 加速
- `normalize_embeddings=True`：对向量做 L2 归一化，使余弦相似度等价于点积（内积），加速计算

### 3.3 BGE Reranker 模型

BGE 同时提供了 Cross-Encoder 重排序模型：

| 模型 | 用途 |
|---|---|
| bge-reranker-base | 基础重排序，速度快 |
| bge-reranker-large | 高精度重排序 |
| bge-reranker-v2-m3 | 多语言重排序，当前最佳开源 Reranker 之一 |

官方推荐：用 Embedding 模型做初筛，再用 Reranker 模型做精排。

---

## 四、向量归一化的意义

你项目中设置了 `normalize_embeddings=True`，这很重要：

**归一化前**：向量长度不一，余弦相似度和点积结果不同
**归一化后**：所有向量长度为 1，余弦相似度 = 点积

```
余弦相似度 = (A · B) / (|A| × |B|)

归一化后 |A| = |B| = 1，所以：
余弦相似度 = A · B = 点积
```

好处：点积计算比余弦相似度快，向量数据库可以用更高效的内积索引（如 FAISS 的 IndexFlatIP）。

---

## 五、Embedding 模型选型指南

### 5.1 选型维度

| 维度 | 考虑因素 |
|---|---|
| 语言 | 中文场景优先 BGE-zh 系列；多语言场景用 BGE-M3 |
| 维度 | 维度越高精度越好，但存储和计算成本也越高 |
| 模型大小 | 影响加载时间和内存占用 |
| 是否需要 GPU | 大模型推理慢，CPU 上可能不可接受 |
| 开源 vs API | 开源模型数据不出域；API 模型（OpenAI）更方便但有隐私风险 |

### 5.2 主流 Embedding 模型对比（2026 年）

| 模型 | 类型 | 维度 | 特点 |
|---|---|---|---|
| BAAI/bge-base-zh-v1.5 | 开源/本地 | 768 | 中文最佳性价比 |
| BAAI/bge-m3 | 开源/本地 | 1024 | 多语言、支持稀疏+稠密混合检索 |
| OpenAI text-embedding-3-small | API | 1536 | 方便但数据需上传 |
| OpenAI text-embedding-3-large | API | 3072 | 最高精度 API 方案 |
| Jina Embeddings v3 | 开源/API | 1024 | 多语言，支持长文本 |
| Nomic Embed Text v2 | 开源 | 768 | 轻量高效 |

### 5.3 企业选型建议

- **数据敏感（金融、医疗、政务）**→ 必须用开源本地模型，数据不出域
- **快速原型验证** → OpenAI API 最省事
- **中文为主的知识库** → BGE-zh 系列是首选
- **多语言混合场景** → BGE-M3 或 Jina v3

---

## 六、Embedding 质量评估

### 6.1 常用评估指标

- **Recall@K**：Top-K 结果中包含正确答案的比例
- **MRR（Mean Reciprocal Rank）**：正确答案排名的倒数的平均值
- **NDCG（Normalized Discounted Cumulative Gain）**：考虑排名位置的综合指标

### 6.2 MTEB 排行榜

MTEB（Massive Text Embedding Benchmark）是 Embedding 模型的标准评测：
- 官网：https://huggingface.co/spaces/mteb/leaderboard
- 覆盖检索、分类、聚类、语义相似度等多个任务
- 选模型时参考 MTEB 排名是最靠谱的方式

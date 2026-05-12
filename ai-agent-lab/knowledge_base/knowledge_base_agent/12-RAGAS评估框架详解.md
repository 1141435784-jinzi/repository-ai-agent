# RAGAS 评估框架详解

> 项目应用：rag_evaluator.py 使用 RAGAS 指标评估 RAG 质量
> 整理来源：基于 [RAGAS 官方文档](https://docs.ragas.io/)、[RAG Evaluation with RAGAS](https://tutorialq.com/ai/dl-applications/rag-evaluation-ragas)、[RAGAS Metrics for Production](https://markaicode.com/rag-evaluation-ragas-metrics-production/)、[Evaluating RAG Pipelines with RAGAS](https://jheiduk.com/posts/ragas-evaluation-tutorial/) 归纳改写
> 最后更新：2026 年 4 月

---

## 一、为什么需要评估 RAG

RAG 系统有多个环节（检索、排序、生成），任何一个环节出问题都会导致最终回答质量下降。但问题出在哪里？没有量化指标就只能靠"感觉"。

**现实例子**：用户问"退货政策是什么？"，Agent 回答错了。原因可能是：
1. 检索器没找到退货政策文档（检索问题）
2. 找到了但排在第 5 位，被截断了（排序问题）
3. 找到了正确文档，但 LLM 没有正确理解（生成问题）

RAGAS 的四个核心指标分别对应不同环节，帮你精确定位问题。

---

## 二、RAGAS 是什么

RAGAS（Retrieval Augmented Generation Assessment）是一个专门为 RAG 系统设计的评估框架。

**核心特点**：
- 不需要人工标注的 ground truth 数据集（LLM-as-Judge）
- 四个互补的指标覆盖检索和生成两个阶段
- 每个指标分数范围 0-1，越高越好
- 开源，可本地运行

---

## 三、四大核心指标

### 3.1 Faithfulness（忠实度）

**衡量什么**：回答中的每个陈述是否都能在检索到的上下文中找到依据。

**分数含义**：
- 1.0 = 回答中的所有陈述都有上下文支撑
- 0.0 = 回答完全是编造的（幻觉）

**计算方式**：
1. 将回答拆分为多个独立陈述
2. 逐一检查每个陈述是否能从上下文中推导出来
3. Faithfulness = 有依据的陈述数 / 总陈述数

**示例**：

```
上下文："退货需在 7 天内，商品需保持原包装。"

回答 A："退货需在 7 天内完成，商品要保持原包装。"
→ Faithfulness = 1.0（两个陈述都有依据）

回答 B："退货需在 7 天内完成，且需要提供购买发票。"
→ Faithfulness = 0.5（"7 天内"有依据，"购买发票"是编造的）
```

**诊断意义**：Faithfulness 低 → LLM 在"编造"内容，需要优化 Prompt 或降低 temperature。

### 3.2 Answer Relevancy（回答相关性）

**衡量什么**：回答是否真正回答了用户的问题。

**分数含义**：
- 1.0 = 回答完全切题
- 0.0 = 回答完全跑题

**计算方式**：
1. 根据回答反向生成多个可能的问题
2. 计算这些生成问题与原始问题的语义相似度
3. 取平均值作为 Answer Relevancy 分数

**示例**：

```
问题："LangGraph 的 checkpoint 机制怎么用？"

回答 A："LangGraph 通过 SqliteSaver 或 PostgresSaver 实现 checkpoint，
        在编译图时传入 checkpointer 参数即可启用。"
→ Answer Relevancy ≈ 0.95（直接回答了问题）

回答 B："LangGraph 是 LangChain 团队开发的图编排框架，
        支持多种节点类型和边类型。"
→ Answer Relevancy ≈ 0.3（相关但没回答具体问题）
```

**诊断意义**：Answer Relevancy 低 → 回答跑题了，可能是 Prompt 不够聚焦或检索到的上下文不相关。

### 3.3 Context Precision（上下文精确度）

**衡量什么**：检索到的文档中，相关文档是否排在前面。

**分数含义**：
- 1.0 = 所有相关文档都排在最前面
- 0.0 = 相关文档全部排在后面或不存在

**计算方式**：
1. 判断每个检索到的文档是否与问题相关
2. 计算相关文档的排名加权精确度（排名越靠前权重越高）

**示例**：

```
问题："checkpoint 怎么用？"

检索结果 A：[checkpoint 文档, checkpoint 文档, 无关文档]
→ Context Precision ≈ 1.0（相关文档排在前面）

检索结果 B：[无关文档, 无关文档, checkpoint 文档]
→ Context Precision ≈ 0.33（相关文档排在最后）
```

**诊断意义**：Context Precision 低 → 检索器找到了相关文档但排序不好，需要优化 Rerank 或调整检索权重。

### 3.4 Context Recall（上下文召回率）

**衡量什么**：回答问题所需的信息是否都被检索到了。

**分数含义**：
- 1.0 = 回答所需的所有信息都在检索到的上下文中
- 0.0 = 检索到的上下文完全没有覆盖所需信息

**计算方式**：
1. 将参考答案（ground truth）拆分为多个陈述
2. 检查每个陈述是否能从检索到的上下文中找到
3. Context Recall = 被覆盖的陈述数 / 总陈述数

**注意**：Context Recall 需要 ground truth（参考答案），是四个指标中唯一需要标注数据的。

**诊断意义**：Context Recall 低 → 检索器漏掉了重要文档，需要增加 Top-K、优化切分策略或丰富知识库。

---

## 四、指标与问题定位对照表

| 现象 | 可能原因 | 对应指标 | 优化方向 |
|---|---|---|---|
| 回答编造内容 | LLM 幻觉 | Faithfulness 低 | 优化 Prompt，降低 temperature |
| 回答跑题 | 检索不相关或 Prompt 不聚焦 | Answer Relevancy 低 | 优化 Prompt，改进检索 |
| 相关文档排名靠后 | 检索排序不佳 | Context Precision 低 | 添加/优化 Rerank |
| 缺少关键信息 | 检索遗漏 | Context Recall 低 | 增加 Top-K，优化切分 |
| 所有指标都低 | 知识库缺少相关内容 | 全部低 | 补充知识库文档 |

---

## 五、在代码中使用 RAGAS

### 5.1 基本用法

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": ["LangGraph 的 checkpoint 怎么用？"],
    "answer": ["通过 SqliteSaver 实现，编译图时传入 checkpointer 参数。"],
    "contexts": [["LangGraph 支持 SqliteSaver 和 PostgresSaver 两种 checkpoint 方式..."]],
    "ground_truth": ["LangGraph 通过 checkpointer 参数启用状态持久化，支持 SQLite 和 PostgreSQL。"],
}

dataset = Dataset.from_dict(eval_data)

# 运行评估
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.95, 'context_recall': 0.85}
```

### 5.2 批量评估

```python
# 准备多组测试用例
test_cases = [
    {
        "question": "BM25 是什么？",
        "ground_truth": "BM25 是一个基于词频和逆文档频率的经典信息检索算法。"
    },
    {
        "question": "Rerank 的作用是什么？",
        "ground_truth": "Rerank 使用 Cross-Encoder 对初筛结果做精排，提升检索精度。"
    },
    # ... 更多测试用例
]

# 对每个测试用例运行 RAG 并收集结果
for case in test_cases:
    rag_result = rag_engine.query(case["question"])
    case["answer"] = rag_result["answer"]
    case["contexts"] = rag_result["contexts"]

# 批量评估
dataset = Dataset.from_dict({
    "question": [c["question"] for c in test_cases],
    "answer": [c["answer"] for c in test_cases],
    "contexts": [c["contexts"] for c in test_cases],
    "ground_truth": [c["ground_truth"] for c in test_cases],
})

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

---

## 六、构建评估数据集

### 6.1 手动构建

为知识库中的关键主题准备问答对：

```python
eval_dataset = [
    {
        "question": "什么是 Agent？",
        "ground_truth": "Agent 是一个能够感知环境、做出决策并采取行动的自主系统。"
    },
    {
        "question": "RAG 的工作流程是什么？",
        "ground_truth": "RAG 的流程是：用户提问 → 检索相关文档 → 将文档作为上下文 → LLM 生成回答。"
    },
]
```

### 6.2 自动生成（推荐）

用 RAGAS 的 TestsetGenerator 自动从知识库生成测试用例：

```python
from ragas.testset.generator import TestsetGenerator
from langchain_openai import ChatOpenAI

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4"),
    critic_llm=ChatOpenAI(model="gpt-4"),
)

# 从文档自动生成测试集
testset = generator.generate_with_langchain_docs(
    documents=knowledge_base_docs,
    test_size=20,  # 生成 20 个测试用例
)
```

---

## 七、评估驱动的优化循环

```
1. 建立基线 → 用当前 RAG 配置跑一次评估，记录各指标分数
       ↓
2. 定位问题 → 根据指标对照表找到薄弱环节
       ↓
3. 实施优化 → 调整对应环节（切分策略/检索权重/Rerank/Prompt）
       ↓
4. 重新评估 → 跑同一套测试用例，对比分数变化
       ↓
5. 确认改进 → 分数提升则保留，否则回滚
       ↓
   回到步骤 2，持续迭代
```

### 典型优化实验

| 实验 | 调整内容 | 观察指标 |
|---|---|---|
| 增大 chunk_size | 800 → 1200 | Context Recall 是否提升 |
| 增大 chunk_overlap | 100 → 200 | Context Recall 是否提升 |
| 添加 Rerank | 无 → bge-reranker-base | Context Precision 是否提升 |
| 调整 BM25 权重 | 0.4 → 0.5 | 精确关键词查询的 Context Precision |
| 优化 System Prompt | 添加"基于上下文回答" | Faithfulness 是否提升 |
| 增加 Top-K | 3 → 5 | Context Recall 是否提升 |

---

## 八、生产环境持续评估

### 8.1 定期回归测试

```python
# 每次知识库更新后自动运行评估
def post_index_rebuild_evaluation():
    """索引重建后的自动评估"""
    result = evaluate(standard_test_dataset, metrics=[...])

    # 与基线对比
    if result["faithfulness"] < baseline["faithfulness"] - 0.05:
        alert("Faithfulness 下降超过 5%，请检查知识库变更")

    # 记录历史
    save_evaluation_history(result)
```

### 8.2 用户反馈闭环

```python
# 收集用户对回答的评价
@app.post("/feedback")
async def submit_feedback(question: str, answer: str, rating: int):
    """用户反馈：1-5 分"""
    if rating <= 2:
        # 低分回答加入评估数据集，用于后续分析
        add_to_evaluation_dataset(question, answer, rating)
```

---

## 九、注意事项

1. **RAGAS 依赖 LLM 做评判**：评估本身会消耗 LLM API 调用，有成本
2. **评估结果有波动**：同一组数据多次评估分数可能略有不同（LLM 的随机性）
3. **不要过度优化单一指标**：四个指标需要平衡，过度优化一个可能损害另一个
4. **测试集要有代表性**：覆盖知识库的主要主题和不同类型的问题
5. **Context Recall 需要 ground truth**：这是唯一需要人工标注的指标，可以先从少量高质量标注开始

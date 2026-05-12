"""
=== RAG 质量评估模块（基于 RAGAS 框架）===

【知识点】RAGAS（Retrieval Augmented Generation Assessment）
是专门用于评估 RAG 系统质量的开源框架，提供以下核心指标：

1. Faithfulness（忠实度）：回答是否忠实于检索到的上下文？
   - 防止 LLM 在有上下文的情况下仍然"编造"
   - 分数越高越好，1.0 = 完全基于上下文

2. Answer Relevancy（答案相关性）：回答是否切题？
   - 评估回答与用户问题的相关程度
   - 分数越高越好，1.0 = 完全切题

3. Context Precision（上下文精确率）：检索到的文档是否都相关？
   - 评估检索质量，是否有太多无关文档混入
   - 分数越高越好，1.0 = 检索到的全部相关

4. Context Recall（上下文召回率）：相关文档是否都被检索到了？
   - 评估检索是否遗漏了重要信息
   - 需要标注数据（ground truth）才能计算

【企业实战】为什么需要 RAG 评估？
- 调参有依据：chunk_size、Top K、权重等参数的调优需要量化指标
- 质量可追踪：每次修改知识库或调整参数后，评估分数是否提升
- 上线有标准：评估分数达到阈值才允许上线

【使用方式】
  python rag_evaluator.py
"""

import logging
from dataclasses import dataclass, field

from src.config import TEMPERATURE
from src.llm.service import get_llm

logger = logging.getLogger(__name__)


# ============================================================
# 评估数据集定义
# ============================================================
@dataclass
class EvalSample:
    """单条评估样本
    
    【知识点】评估 RAG 需要准备"测试用例"：
    - question：用户问题
    - ground_truth：标准答案（人工标注）
    - contexts：检索到的上下文（由 RAG 引擎提供）
    - answer：RAG 系统生成的回答
    """
    question: str
    ground_truth: str = ""          # 人工标注的标准答案（评估召回率需要）
    contexts: list[str] = field(default_factory=list)  # 检索到的上下文
    answer: str = ""                # RAG 系统生成的回答


# ============================================================
# 预置评估数据集（基于 agent-lab 知识库内容）
# ============================================================
# 【知识点】评估数据集应该覆盖知识库的主要内容和常见问题类型
# 包括：精确查询、语义查询、跨文档查询、知识库外的问题
EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="Transformer 的核心组件有哪些？",
        ground_truth="Tokenization分词、Embedding词嵌入+位置编码、Self-Attention自注意力、Multi-Head Attention多头注意力、Norm & Add归一化与残差连接、FFN前馈网络、Mask与Cross-Attention",
    ),
    EvalSample(
        question="什么是 Temperature 参数？企业场景怎么设置？",
        ground_truth="Temperature控制LLM输出的随机性，0表示确定性输出，0.7表示适度随机，1.5表示高度随机。企业场景中，客服和风控等需要准确性的场景用temperature=0，创意写作用0.7-1.0。",
    ),
    EvalSample(
        question="RAG 的完整流程是什么？",
        ground_truth="RAG的完整流程分为离线阶段和在线阶段。离线阶段：文档加载→文本分块→向量化→存入向量数据库。在线阶段：用户提问→向量检索→重排序→LLM生成回答。",
    ),
    EvalSample(
        question="LangGraph 的核心概念有哪些？",
        ground_truth="LangGraph的核心概念包括State状态、Node节点、Edge边、Conditional Edge条件边。State在节点之间流转数据，Node执行具体逻辑，Edge连接节点决定执行流程，Conditional Edge根据条件决定下一步走哪个节点。",
    ),
    EvalSample(
        question="什么是 LLM 幻觉？怎么解决？",
        ground_truth="LLM幻觉是指模型生成看似合理但实际错误或虚构的内容。分为事实性幻觉、忠实性幻觉和指令性幻觉。解决方案包括RAG检索增强、Prompt约束、答案溯源、Temperature设为0、多模型交叉验证。",
    ),
    EvalSample(
        question="混合检索是什么？为什么需要它？",
        ground_truth="混合检索是将向量语义检索和BM25关键词检索结合使用。纯向量检索对精确关键词匹配差，纯BM25不理解语义，混合检索两者互补，覆盖更全面。",
    ),
    EvalSample(
        question="Python 的最新版本是多少？",  # 知识库外的问题，测试兜底能力
        ground_truth="",  # 知识库中没有这个信息
    ),
]


# ============================================================
# 简易 RAG 评估器（不依赖 ragas 库的轻量实现）
# ============================================================
class RAGEvaluator:
    """RAG 质量评估器
    
    【知识点】评估流程：
    1. 对每个测试问题，调用 RAG 引擎获取检索结果和生成回答
    2. 使用 LLM 作为评判者（LLM-as-Judge）评估各项指标
    3. 汇总所有样本的分数，输出评估报告
    
    【知识点】LLM-as-Judge 是当前主流的自动评估方法：
    - 用一个 LLM 来评判另一个 LLM 的回答质量
    - 比人工评估快得多，比规则匹配灵活得多
    - RAGAS 框架的核心就是这个思路
    """

    def __init__(self) -> None:
        """初始化评估器，创建用于评判的 LLM 实例

        Args:
            无参数

        Returns:
            None
        """
        self._llm = get_llm(temperature=0)  # 评估时用 temperature=0 保证一致性

    def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """评估忠实度：回答是否基于检索到的上下文
        
        【知识点】Faithfulness 评估原理：
        1. 从回答中提取所有"声明"（claims）
        2. 检查每个声明是否能在上下文中找到支持
        3. 忠实度 = 有支持的声明数 / 总声明数

        Args:
            answer: RAG 系统生成的回答文本
            contexts: 检索到的上下文文本列表

        Returns:
            float: 忠实度分数，范围 0.0~1.0；
                   1.0 表示完全基于上下文，0.0 表示完全无关或输入为空
        """
        if not contexts or not answer:
            return 0.0

        context_text = "\n".join(contexts)
        prompt = f"""请评估以下回答是否忠实于给定的参考资料。

参考资料：
{context_text}

回答：
{answer}

评分标准（0-1）：
- 1.0：回答完全基于参考资料，没有编造任何信息
- 0.7：回答大部分基于参考资料，有少量推断但合理
- 0.5：回答部分基于参考资料，部分是模型自己的知识
- 0.3：回答大部分不在参考资料中
- 0.0：回答完全与参考资料无关或编造

请只返回一个 0 到 1 之间的数字，不要返回其他内容。"""

        try:
            response = self._llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception) as e:
            logger.warning(f"忠实度评估失败: {e}")
            return 0.0

    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性：回答是否切题
        
        【知识点】Answer Relevancy 评估原理：
        1. 根据回答反向生成可能的问题
        2. 计算生成的问题与原始问题的相似度
        3. 相似度越高，说明回答越切题
        
        这里简化为直接让 LLM 评分

        Args:
            question: 用户的原始问题
            answer: RAG 系统生成的回答文本

        Returns:
            float: 相关性分数，范围 0.0~1.0；
                   1.0 表示完全切题，0.0 表示完全不相关或 answer 为空
        """
        if not answer:
            return 0.0

        prompt = f"""请评估以下回答与问题的相关程度。

问题：{question}

回答：{answer}

评分标准（0-1）：
- 1.0：回答完全切题，直接回答了问题
- 0.7：回答基本切题，但包含一些无关信息
- 0.5：回答部分相关，但没有直接回答问题
- 0.3：回答与问题关系不大
- 0.0：回答完全不相关

请只返回一个 0 到 1 之间的数字，不要返回其他内容。"""

        try:
            response = self._llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception) as e:
            logger.warning(f"相关性评估失败: {e}")
            return 0.0

    def evaluate_context_precision(self, question: str, contexts: list[str]) -> float:
        """评估上下文精确率：检索到的文档是否都与问题相关
        
        【知识点】Context Precision 反映检索质量：
        - 高精确率 = 检索到的文档大部分都有用
        - 低精确率 = 检索到了很多无关文档（噪声多）

        Args:
            question: 用户的原始问题
            contexts: 检索到的上下文文本列表

        Returns:
            float: 精确率分数，范围 0.0~1.0；
                   1.0 表示所有检索文档都高度相关，0.0 表示完全不相关或 contexts 为空
        """
        if not contexts:
            return 0.0

        prompt = f"""以下是针对问题检索到的多段参考资料，请评估这些资料整体与问题的相关程度。

问题：{question}

检索到的资料：
{chr(10).join(f'[{i+1}] {ctx[:200]}...' for i, ctx in enumerate(contexts))}

评分标准（0-1）：
- 1.0：所有检索到的资料都与问题高度相关
- 0.7：大部分资料相关，少量不太相关
- 0.5：约一半资料相关
- 0.3：大部分资料不相关
- 0.0：检索到的资料完全不相关

请只返回一个 0 到 1 之间的数字，不要返回其他内容。"""

        try:
            response = self._llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception) as e:
            logger.warning(f"上下文精确率评估失败: {e}")
            return 0.0

    def run_evaluation(self, rag_engine, dataset: list[EvalSample] | None = None) -> dict:
        """运行完整评估
        
        【知识点】评估流程：
        1. 遍历测试数据集
        2. 对每个问题调用 RAG 引擎检索
        3. 用 LLM 生成回答
        4. 评估各项指标
        5. 汇总输出报告

        Args:
            rag_engine: RAGEngine 实例，用于对每个测试问题执行检索
            dataset: 评估样本列表，每个样本包含 question 和 ground_truth；
                     如果为 None，则使用模块内预置的 EVAL_DATASET

        Returns:
            dict: 评估报告，包含：
                - samples (list[dict]): 每个样本的详细评估结果
                - avg_faithfulness (float): 平均忠实度
                - avg_relevancy (float): 平均答案相关性
                - avg_precision (float): 平均上下文精确率
        """
        if dataset is None:
            dataset = EVAL_DATASET

        results: list[dict] = []
        total_faithfulness = 0.0
        total_relevancy = 0.0
        total_precision = 0.0
        valid_count = 0

        print("\n" + "=" * 60)
        print("📊 RAG 质量评估报告")
        print("=" * 60)

        for i, sample in enumerate(dataset, 1):
            print(f"\n--- 评估样本 {i}/{len(dataset)} ---")
            print(f"问题: {sample.question}")

            # 调用 RAG 引擎检索
            rag_result = rag_engine.query(sample.question)
            contexts = [rag_result["answer_context"]] if rag_result["found"] else []

            # 用 LLM 基于检索结果生成回答
            if rag_result["found"]:
                gen_prompt = (
                    f"基于以下参考资料回答问题。如果资料中没有相关信息，请说明。\n\n"
                    f"参考资料：\n{rag_result['answer_context']}\n\n"
                    f"问题：{sample.question}"
                )
                answer = self._llm.invoke(gen_prompt).content
            else:
                answer = "知识库中未找到相关信息。"

            sample.contexts = contexts
            sample.answer = answer

            print(f"回答: {answer[:100]}...")
            print(f"检索到文档: {rag_result['doc_count']} 个")

            # 评估各项指标
            if rag_result["found"]:
                faithfulness = self.evaluate_faithfulness(answer, contexts)
                relevancy = self.evaluate_relevancy(sample.question, answer)
                precision = self.evaluate_context_precision(sample.question, contexts)

                total_faithfulness += faithfulness
                total_relevancy += relevancy
                total_precision += precision
                valid_count += 1

                print(f"忠实度: {faithfulness:.2f} | 相关性: {relevancy:.2f} | 检索精确率: {precision:.2f}")
            else:
                faithfulness = relevancy = precision = 0.0
                print("⚠️  未检索到相关文档，跳过评估")

            results.append({
                "question": sample.question,
                "answer": answer,
                "found": rag_result["found"],
                "doc_count": rag_result["doc_count"],
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "precision": precision,
            })

        # 汇总报告
        avg_faithfulness = total_faithfulness / valid_count if valid_count > 0 else 0
        avg_relevancy = total_relevancy / valid_count if valid_count > 0 else 0
        avg_precision = total_precision / valid_count if valid_count > 0 else 0

        print("\n" + "=" * 60)
        print("📈 评估汇总")
        print("=" * 60)
        print(f"评估样本数: {len(dataset)}")
        print(f"有效评估数: {valid_count}")
        print(f"平均忠实度 (Faithfulness):     {avg_faithfulness:.2f}")
        print(f"平均相关性 (Answer Relevancy): {avg_relevancy:.2f}")
        print(f"平均检索精确率 (Context Precision): {avg_precision:.2f}")
        print("=" * 60)

        # 【知识点】评估分数的参考标准
        # > 0.8：优秀，可以上线
        # 0.6-0.8：良好，建议优化后上线
        # < 0.6：需要改进（调整分块策略、检索参数、Prompt 等）
        if avg_faithfulness >= 0.8 and avg_relevancy >= 0.8:
            print("✅ 整体质量优秀，RAG 系统可投入使用")
        elif avg_faithfulness >= 0.6 and avg_relevancy >= 0.6:
            print("⚠️  整体质量良好，建议进一步优化后上线")
        else:
            print("❌ 整体质量需要改进，建议调整分块策略和检索参数")

        return {
            "samples": results,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "avg_precision": avg_precision,
        }


# ============================================================
# 独立运行评估
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("🔧 初始化 RAG 引擎...")
    from rag_engine import RAGEngine
    rag = RAGEngine()

    print("🔧 初始化评估器...")
    evaluator = RAGEvaluator()

    print("🚀 开始评估...\n")
    evaluator.run_evaluation(rag)

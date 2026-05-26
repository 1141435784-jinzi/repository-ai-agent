
"""
RAG 质量评估测试

使用 RAGAS 评估检索质量，包括：
1. 上下文相关性 (Context Relevancy)
2. 答案正确性 (Answer Correctness)
3. 上下文召回率 (Context Recall)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

# RAGAS 评估指标
# pip install ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        context_relevancy,
        context_recall,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS not installed")
class TestRAGASEvaluation:
    """测试RAGAS评估"""

    def test_evaluate_context_relevancy(self):
        """测试上下文相关性评估"""
        # 准备测试数据
        questions = ["北京有哪些著名景点？"]
        contexts = [
            [
                "北京故宫是中国明清两代的皇家宫殿，位于北京市中心。",
                "上海外滩是上海著名的旅游景点。"  # 无关上下文
            ]
        ]
        answers = ["北京故宫是著名景点。"]
        ground_truths = [["北京故宫是著名景点"]]
        
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        })
        
        # 评估
        result = evaluate(
            dataset=dataset,
            metrics=[context_relevancy],
        )
        
        assert result is not None
        # 上下文相关性应该低于1，因为有无关上下文
        assert result["context_relevancy"] < 1.0

    def test_evaluate_answer_correctness(self):
        """测试答案正确性评估"""
        questions = ["中国的首都是哪里？"]
        contexts = [["北京是中华人民共和国的首都。"]]
        answers = ["北京"]
        ground_truths = [["北京"]]
        
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        })
        
        result = evaluate(
            dataset=dataset,
            metrics=[answer_correctness],
        )
        
        assert result is not None
        # 正确答案应该获得高分
        assert result["answer_correctness"] >= 0.8

    def test_evaluate_context_recall(self):
        """测试上下文召回率"""
        questions = ["北京有哪些著名景点？"]
        contexts = [["北京故宫是著名景点。"]]
        answers = ["北京故宫"]
        ground_truths = [["北京故宫", "天安门"]]  # 只召回了部分答案
        
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        })
        
        result = evaluate(
            dataset=dataset,
            metrics=[context_recall],
        )
        
        assert result is not None
        # 召回率应该低于1，因为没有召回全部相关内容
        assert result["context_recall"] < 1.0

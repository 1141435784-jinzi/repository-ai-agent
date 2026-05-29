
"""
测试配置文件

包含 pytest fixtures 和全局配置
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock embeddings to avoid network calls"""
    with patch("src.rag.engine.get_embeddings") as mock:
        mock.return_value = MagicMock(
            embed_documents=MagicMock(return_value=[[0.1] * 384]),
            embed_query=MagicMock(return_value=[0.1] * 384)
        )
        yield mock


@pytest.fixture(autouse=True)
def mock_llm():
    """Mock LLM to avoid network calls"""
    with patch("src.llm.gateway.get_llm") as mock:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Mock response")
        mock_llm_instance.generate.return_value = MagicMock(
            generations=[[MagicMock(text="Mock response")]]
        )
        mock.return_value = mock_llm_instance
        yield mock


@pytest.fixture
def test_documents():
    """创建测试文档"""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="北京故宫是中国明清两代的皇家宫殿，位于北京市中心。",
            metadata={"source_file": "beijing.md", "section": "景点"}
        ),
        Document(
            page_content="上海外滩是上海著名的旅游景点，可以欣赏黄浦江两岸的美景。",
            metadata={"source_file": "shanghai.md", "section": "景点"}
        ),
        Document(
            page_content="杭州西湖是中国著名的风景名胜区，以秀丽的湖光山色闻名。",
            metadata={"source_file": "hangzhou.md", "section": "景点"}
        ),
    ]

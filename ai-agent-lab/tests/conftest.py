"""
=== pytest 配置文件 ===

定义测试共享的 fixtures 和配置
"""

import pytest
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def project_root():
    """项目根目录路径"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="function")
def mock_env():
    """模拟环境变量"""
    original_env = os.environ.copy()
    os.environ["TEST_MODE"] = "true"
    yield
    os.environ.clear()
    os.environ.update(original_env)
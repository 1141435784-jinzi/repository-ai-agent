
"""
单元测试：LLM Gateway 模块

测试覆盖：
1. LLMProviderConfig 配置类
2. get_llm() 函数
3. invoke_with_fallback() 容灾降级
4. 智能路由功能
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.llm.gateway import (
    LLMProviderConfig,
    get_llm,
    invoke_with_fallback,
    get_call_stats,
    _PROVIDERS,
)


class TestLLMProviderConfig:
    """测试 LLMProviderConfig 配置类"""

    def test_config_creation(self):
        """测试配置类创建"""
        config = LLMProviderConfig(
            name="test",
            api_key="test_key",
            base_url="https://api.test.com",
            model="test-model",
            enabled=True,
            priority=1,
            max_retries=2,
            timeout=30,
        )
        
        assert config.name == "test"
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.test.com"
        assert config.model == "test-model"
        assert config.enabled is True
        assert config.priority == 1
        assert config.max_retries == 2
        assert config.timeout == 30

    def test_config_defaults(self):
        """测试配置类默认值"""
        config = LLMProviderConfig(
            name="test",
            api_key="key",
            base_url="https://test.com",
            model="model",
        )
        
        assert config.enabled is True
        assert config.priority == 0
        assert config.max_retries == 2
        assert config.timeout == 30


class TestGetLLM:
    """测试 get_llm() 函数"""

    @patch("src.llm.gateway._create_llm")
    def test_get_llm_with_provider(self, mock_create):
        """测试指定provider获取LLM"""
        mock_llm = Mock()
        mock_create.return_value = mock_llm
        
        llm = get_llm(provider="ollama")
        
        assert llm == mock_llm
        mock_create.assert_called_once()

    @patch("src.llm.gateway._create_llm")
    def test_get_llm_cache(self, mock_create):
        """测试LLM实例缓存"""
        mock_llm = Mock()
        mock_create.return_value = mock_llm
        
        # 第一次调用
        llm1 = get_llm(provider="ollama", temperature=0.7)
        # 第二次调用相同参数
        llm2 = get_llm(provider="ollama", temperature=0.7)
        
        # 应该返回同一个实例
        assert llm1 is llm2
        # create_llm 应该只被调用一次
        assert mock_create.call_count == 1

    def test_get_llm_invalid_provider(self):
        """测试无效provider"""
        with pytest.raises(ValueError):
            get_llm(provider="invalid_provider")


class TestInvokeWithFallback:
    """测试容灾降级功能"""

    @patch("src.llm.gateway.get_llm")
    def test_invoke_success(self, mock_get_llm):
        """测试正常调用成功"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="test response")
        mock_get_llm.return_value = mock_llm
        
        result = invoke_with_fallback([], provider="ollama")
        
        assert result == "test response"
        mock_llm.invoke.assert_called_once()

    @patch("src.llm.gateway.get_llm")
    def test_invoke_with_fallback(self, mock_get_llm):
        """测试降级机制"""
        mock_llm1 = Mock()
        mock_llm1.invoke.side_effect = Exception("Primary failed")
        
        mock_llm2 = Mock()
        mock_llm2.invoke.return_value = Mock(content="fallback response")
        
        # 第一次调用返回失败的LLM，第二次返回成功的LLM
        mock_get_llm.side_effect = [mock_llm1, mock_llm2]
        
        result = invoke_with_fallback([], provider="deepseek")
        
        assert result == "fallback response"
        assert mock_llm1.invoke.call_count == 1
        assert mock_llm2.invoke.call_count == 1


class TestCallStats:
    """测试调用统计"""

    def test_stats_initialization(self):
        """测试统计初始化"""
        stats = get_call_stats()
        
        assert stats.total_calls == 0
        assert stats.success_calls == 0
        assert stats.fallback_calls == 0
        assert stats.error_calls == 0
        assert stats.calls_by_provider == {}

    @patch("src.llm.gateway.get_llm")
    def test_stats_update_on_success(self, mock_get_llm):
        """测试成功调用时统计更新"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="success")
        mock_get_llm.return_value = mock_llm
        
        invoke_with_fallback([], provider="ollama")
        
        stats = get_call_stats()
        assert stats.total_calls == 1
        assert stats.success_calls == 1
        assert stats.calls_by_provider["ollama"] == 1


"""
集成测试：API 层

测试覆盖：
1. 对话接口
2. 会话管理
3. 健康检查
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

from src.api.server import app


class TestAPIIntegration:
    """测试API集成"""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查接口"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    @patch("src.api.routes.chat.create_chat_completion")
    async def test_chat_endpoint(self, mock_create_completion):
        """测试对话接口"""
        mock_response = {
            "session_id": "test-session",
            "response": "Hello! How can I help you?",
            "finish_reason": "completed",
            "thought": None,
            "tool_calls": [],
            "total_tokens": 10,
            "latency_ms": 100
        }
        mock_create_completion.return_value = mock_response
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/chat",
                json={
                    "message": "Hello",
                    "session_id": "test-session"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! How can I help you?"
        assert data["session_id"] == "test-session"

    @pytest.mark.asyncio
    @patch("src.api.routes.chat.create_chat_completion")
    async def test_chat_streaming(self, mock_create_completion):
        """测试流式对话接口"""
        async def mock_stream():
            chunks = [
                {"token": "Hello", "delta": True},
                {"token": "!", "delta": False}
            ]
            for chunk in chunks:
                yield chunk
        
        mock_create_completion.return_value = mock_stream()
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/chat/stream",
                json={
                    "message": "Hello",
                    "session_id": "test-session"
                }
            )
        
        assert response.status_code == 200
        # 验证流式响应


    @pytest.mark.asyncio
    @patch("src.api.routes.chat.create_chat_completion")
    async def test_chat_error_handling(self, mock_create_completion):
        """测试错误处理"""
        mock_create_completion.side_effect = Exception("Internal error")
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/chat",
                json={
                    "message": "Hello",
                    "session_id": "test-session"
                }
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

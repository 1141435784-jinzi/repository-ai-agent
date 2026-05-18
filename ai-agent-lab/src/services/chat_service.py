"""
聊天服务

【功能】：
1. 流式聊天处理
2. 消息分发
3. 响应格式化

【设计原则】：
1. 异步处理：支持流式响应
2. 可扩展性：支持多种聊天模式
3. 错误处理：统一的异常处理
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi.responses import StreamingResponse

from src.config import DEEPSEEK_API_KEY


class ChatService:
    """聊天服务类"""

    async def stream_chat(self, message: str, thread_id: Optional[str] = None):
        """
        流式聊天处理
        
        Args:
            message: 用户消息
            thread_id: 会话线程 ID
            
        Returns:
            StreamingResponse: 流式响应
        """
        from src.agents.workflow import get_async_agent
        from src.memory.manager import get_memory_manager

        if not thread_id:
            thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        memory_manager = get_memory_manager()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "model": "deepseek" if DEEPSEEK_API_KEY else "ollama",
            }
        }

        # 获取异步 Agent 实例
        agent_executor = await get_async_agent()

        async def stream_generator():
            # 使用 ainvoke 获取完整响应，然后分块发送模拟流式输出
            result = await agent_executor.ainvoke(
                {"messages": [("user", message)]},
                config=config
            )
            
            # 从结果中提取消息内容
            content = ""
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        content = last_message.content
            
            # 如果有内容，分块发送（模拟流式输出）
            if content:
                # 按字符分块，每块约50个字符
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    yield f"data: {chunk}\n\n"
                    # 添加微小延迟模拟真实流式
                    await asyncio.sleep(0.01)
            
            # 发送结束标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    async def chat(self, message: str, thread_id: Optional[str] = None):
        """
        非流式聊天处理
        
        Args:
            message: 用户消息
            thread_id: 会话线程 ID
            
        Returns:
            dict: 聊天结果
        """
        from src.agents.workflow import get_async_agent

        if not thread_id:
            thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "model": "deepseek" if DEEPSEEK_API_KEY else "ollama",
            }
        }

        agent_executor = await get_async_agent()
        result = await agent_executor.ainvoke(
            {"messages": [("user", message)]},
            config=config
        )

        return {
            "response": result.get("messages", [{}])[-1].get("content", ""),
            "thread_id": thread_id,
            "sources": [],
            "found_in_kb": False
        }

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
import logging
from datetime import datetime
from typing import Optional

from fastapi.responses import StreamingResponse

from src.config import DEEPSEEK_API_KEY

# 创建日志记录器
logger = logging.getLogger("chat_service")
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
if not logger.handlers:
    logger.addHandler(console_handler)


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
        
        if not thread_id:
            thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ========== 日志：记录对话开始 ==========
        start_time = datetime.now()
        logger.info(f"┌─────────────────────────────────────────────────────────────")
        logger.info(f"│ [对话开始] ThreadID: {thread_id}")
        logger.info(f"│ [用户提问] {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.info(f"│ [开始时间] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"└─────────────────────────────────────────────────────────────")

        try:
            # 获取异步 Agent 实例
            agent_executor = await get_async_agent()

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "model": "deepseek" if DEEPSEEK_API_KEY else "ollama",
                }
            }

            async def stream_generator():
                nonlocal start_time
                
                # 使用 ainvoke 获取完整响应，然后分块发送模拟流式输出
                result = await agent_executor.ainvoke(
                    {
                        "messages": [("user", message)],
                        "execution_plan": [],
                        "iteration_count": 0,
                        "task_errors": []
                    },
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
                
                # ========== 日志：记录最终回答 ==========
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"┌─────────────────────────────────────────────────────────────")
                logger.info(f"│ [对话结束] ThreadID: {thread_id}")
                logger.info(f"│ [回答内容] {content[:200]}{'...' if len(content) > 200 else ''}")
                logger.info(f"│ [回答长度] {len(content)} 字符")
                logger.info(f"│ [耗时] {duration:.2f} 秒")
                logger.info(f"│ [结束时间] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"└─────────────────────────────────────────────────────────────")
                
                # 如果有内容，分块发送（模拟流式输出）
                if content:
                    # 转义内容中的换行符和特殊字符，避免干扰 SSE 协议解析
                    escaped_content = content.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
                    # 按字符分块，每块约50个字符
                    chunk_size = 50
                    for i in range(0, len(escaped_content), chunk_size):
                        chunk = escaped_content[i:i+chunk_size]
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
        
        except Exception as e:
            # ========== 日志：记录错误 ==========
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"┌─────────────────────────────────────────────────────────────")
            logger.error(f"│ [对话异常] ThreadID: {thread_id}")
            logger.error(f"│ [错误信息] {str(e)[:200]}")
            logger.error(f"│ [耗时] {duration:.2f} 秒")
            logger.error(f"└─────────────────────────────────────────────────────────────")
            raise

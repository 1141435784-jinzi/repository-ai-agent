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

    async def stream_chat(self, message: str, user_id: Optional[str] = None, thread_id: Optional[str] = None):
        """
        流式聊天处理
        
        Args:
            message: 用户消息
            user_id: 用户 ID（企业级会话管理必备）
            thread_id: 会话线程 ID
            
        Returns:
            StreamingResponse: 流式响应
        """
        from src.agents.workflow import get_async_graph
        
        # 企业级验证：user_id 和 thread_id 必须都提供
        if not user_id or not thread_id:
            raise ValueError("user_id 和 thread_id 参数必须同时提供")

        # ========== 日志：记录对话开始 ==========
        start_time = datetime.now()
        logger.info(f"┌─────────────────────────────────────────────────────────────")
        logger.info(f"│ [对话开始] UserID: {user_id or 'N/A'}, ThreadID: {thread_id}")
        logger.info(f"│ [用户提问] {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.info(f"│ [开始时间] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"└─────────────────────────────────────────────────────────────")

        try:
            # 获取异步 Graph 实例
            graph_executor = await get_async_graph()

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "model": "deepseek" if DEEPSEEK_API_KEY else "ollama",
                }
            }

            async def stream_generator():
                nonlocal start_time
                
                full_content = ""
                chunk_index = 0
                event_count = 0

                try:
                    async for event in graph_executor.astream_events(
                        {
                            "messages": [("user", message)],
                            "execution_plan": [],
                            "iteration_count": 0,
                            "task_errors": []
                        },
                        config=config,
                        version="v2"
                    ):
                        event_count += 1
                        event_type = event.get("event", "")
                        
                        if event_type == "on_chat_model_stream":
                            chunk_index += 1
                            chunk_data = event.get("data", {}).get("chunk", {})
                            
                            # 调试：输出前3个事件的完整结构
                            if chunk_index <= 3:
                                logger.info(f"🔍 第{chunk_index}个事件结构:")
                                logger.info(f"   类型: {type(chunk_data).__name__}")
                                logger.info(f"   hasattr content: {hasattr(chunk_data, 'content')}")
                                if hasattr(chunk_data, 'content'):
                                    logger.info(f"   content类型: {type(chunk_data.content).__name__}")
                                    logger.info(f"   content值: {repr(chunk_data.content)[:200]}")
                            
                            # 提取内容：优先使用 content 属性（AIMessageChunk 对象）
                            token_text = ""
                            
                            # 方式1：检查是否有 content 属性（AIMessageChunk）
                            if hasattr(chunk_data, 'content'):
                                content = chunk_data.content
                                if isinstance(content, str) and content:
                                    token_text = content
                            
                            # 方式2：检查是否是字典且有 content 字段
                            if not token_text and isinstance(chunk_data, dict):
                                if 'content' in chunk_data and isinstance(chunk_data['content'], str) and chunk_data['content']:
                                    token_text = chunk_data['content']
                                elif 'delta' in chunk_data and isinstance(chunk_data['delta'], dict):
                                    if 'content' in chunk_data['delta'] and isinstance(chunk_data['delta']['content'], str) and chunk_data['delta']['content']:
                                        token_text = chunk_data['delta']['content']
                            
                            # 过滤掉纯空白的内容，但保留首次输出时的空格（可能是标题后的空格）
                            if token_text:
                                # 检查是否是纯空白且不是首次输出
                                is_whitespace_only = len(token_text.strip()) == 0
                                if is_whitespace_only and full_content != "":
                                    # 非首次输出的纯空白，跳过
                                    chunk_index += 1
                                    continue
                                
                                # 首次输出时，过滤掉开头可能的 markdown 标题符号
                                if full_content == "" and token_text.startswith('##'):
                                    # 移除开头的 ##，保留后面的空格
                                    token_text = token_text[2:]
                                
                                full_content += token_text
                                escaped_delta = token_text.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
                                yield f"data: {escaped_delta}\n\n"
                                if chunk_index <= 5 or chunk_index % 20 == 0:
                                    logger.info(f"🤖 Token {chunk_index}: {repr(token_text[:50])}")
                        
                        if event_count % 100 == 0:
                            logger.info(f"📡 已处理 {event_count} 个事件")
                
                except Exception as e:
                    logger.error(f"astream_events 异常: {e}")
                    import traceback
                    logger.error(f"详细错误:\n{traceback.format_exc()}")
                    raise
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"┌─────────────────────────────────────────────────────────────")
                logger.info(f"│ [对话结束] ThreadID: {thread_id}")
                logger.info(f"│ [回答内容] {full_content[:200]}{'...' if len(full_content) > 200 else ''}")
                logger.info(f"│ [回答长度] {len(full_content)} 字符")
                logger.info(f"│ [耗时] {duration:.2f} 秒")
                logger.info(f"│ [结束时间] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"│ [事件总数] {event_count}")
                logger.info(f"│ [Token块数] {chunk_index}")
                logger.info(f"└─────────────────────────────────────────────────────────────")
                
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

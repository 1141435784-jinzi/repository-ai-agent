"""
=== 日志工具模块 ===

提供统一的日志配置和结构化日志输出功能，便于追踪Agent工作流的核心调用流程。

【日志层级设计】：
- INFO: 核心流程节点入口/出口
- DEBUG: 详细参数和返回值
- WARNING: 警告信息
- ERROR: 错误信息

【日志格式】：
YYYY-MM-DD HH:MM:SS [LEVEL] [MODULE] [THREAD_ID] - MESSAGE
"""

import logging
import os
from typing import Optional

# 日志级别颜色
LOG_COLORS = {
    'DEBUG': '\033[0;37m',    # 灰色
    'INFO': '\033[0;34m',     # 蓝色
    'WARNING': '\033[0;33m',  # 黄色
    'ERROR': '\033[0;31m',    # 红色
    'CRITICAL': '\033[1;31m', # 红色加粗
    'RESET': '\033[0m'        # 重置
}


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅控制台输出）"""
    
    def format(self, record):
        level_name = record.levelname
        if level_name in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[level_name]}{level_name}{LOG_COLORS['RESET']}"
        return super().format(record)


def setup_logging(log_dir: str = "log", level: int = logging.INFO):
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录
        level: 日志级别
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建主日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # 清除默认处理器
    
    # 文件处理器（无颜色）
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'agent_lab.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器（带颜色）
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    import sys
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(console_formatter)
    # 确保控制台使用 UTF-8 编码（Windows 兼容）
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    root_logger.addHandler(console_handler)
    
    # 禁用第三方库的DEBUG日志
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    
    logging.info("✅ 日志系统初始化完成")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)


class WorkflowLogger:
    """
    Agent工作流专用日志器
    
    提供结构化的日志输出，便于追踪：
    - Agent路由调度
    - RAG节点执行
    - LLM对话请求响应
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def workflow_start(self, thread_id: str, user_query: str):
        """工作流开始"""
        self.logger.info(f"┌─────────────────────────────────────────────────────────────")
        self.logger.info(f"│ 🚀 工作流启动")
        self.logger.info(f"│   Thread ID: {thread_id[:8]}...")
        self.logger.info(f"│   用户查询: {user_query[:50]}...")
        self.logger.info(f"└─────────────────────────────────────────────────────────────")
    
    def workflow_end(self, thread_id: str, duration: float = None):
        """工作流结束"""
        duration_str = f"耗时: {duration:.2f}s" if duration else ""
        self.logger.info(f"┌─────────────────────────────────────────────────────────────")
        self.logger.info(f"│ ✅ 工作流完成")
        self.logger.info(f"│   Thread ID: {thread_id[:8]}...")
        if duration:
            self.logger.info(f"│   {duration_str}")
        self.logger.info(f"└─────────────────────────────────────────────────────────────")
    
    def node_enter(self, node_name: str, thread_id: str, **kwargs):
        """节点进入"""
        self.logger.info(f"├─▶ [{node_name}] 进入 {thread_id[:8]}...")
    
    def node_exit(self, node_name: str, thread_id: str, result: Optional[str] = None):
        """节点退出"""
        result_str = f" | 结果: {result[:30]}..." if result else ""
        self.logger.info(f"└─◀ [{node_name}] 退出 {thread_id[:8]}...{result_str}")
    
    def route_decision(self, thread_id: str, decision: str, confidence: float = None):
        """路由决策"""
        confidence_str = f" (置信度: {confidence:.2f})" if confidence else ""
        self.logger.info(f"🔀 [{thread_id[:8]}] 路由决策: {decision}{confidence_str}")
    
    def rag_query(self, thread_id: str, question: str, node_name: str):
        """RAG查询"""
        self.logger.debug(f"📚 [{thread_id[:8]}] [{node_name}] RAG查询: {question[:50]}...")
    
    def rag_result(self, thread_id: str, node_name: str, sources: list, context_length: int):
        """RAG结果"""
        self.logger.info(f"📚 [{thread_id[:8]}] [{node_name}] RAG完成 | 来源: {sources} | 上下文长度: {context_length}")
    
    def llm_call(self, thread_id: str, provider: str, model: str = None, prompt_tokens: int = None):
        """LLM调用开始"""
        model_str = f" | 模型: {model}" if model else ""
        tokens_str = f" | 提示词token: {prompt_tokens}" if prompt_tokens else ""
        self.logger.info(f"🤖 [{thread_id[:8]}] LLM调用 | Provider: {provider}{model_str}{tokens_str}")
    
    def llm_response(self, thread_id: str, response_length: int, has_tool_call: bool = False):
        """LLM响应完成"""
        tool_call_str = " | 工具调用" if has_tool_call else ""
        self.logger.info(f"🤖 [{thread_id[:8]}] LLM响应完成 | 响应长度: {response_length}{tool_call_str}")
    
    def tool_execution(self, thread_id: str, tool_name: str, tool_args: dict):
        """工具执行"""
        self.logger.info(f"🛠️ [{thread_id[:8]}] 工具调用: {tool_name} | 参数: {tool_args}")
    
    def tool_result(self, thread_id: str, tool_name: str, success: bool, result: str = None):
        """工具执行结果"""
        status = "✅" if success else "❌"
        result_str = f" | 结果: {result[:50]}..." if result else ""
        self.logger.info(f"🛠️ [{thread_id[:8]}] 工具完成: {status} {tool_name}{result_str}")
    
    def memory_update(self, thread_id: str, memory_context_length: int):
        """记忆更新"""
        self.logger.debug(f"🧠 [{thread_id[:8]}] 记忆更新 | 上下文长度: {memory_context_length}")
    
    def info(self, node_name: str, message: str, thread_id: str = "system"):
        """通用信息记录"""
        self.logger.info(f"ℹ️ [{thread_id[:8]}] [{node_name}] {message}")

    def warning(self, node_name: str, message: str, thread_id: str = "system"):
        """通用警告记录"""
        self.logger.warning(f"⚠️ [{thread_id[:8]}] [{node_name}] {message}")

    def debug(self, node_name: str, message: str, thread_id: str = "system"):
        """通用调试记录"""
        self.logger.debug(f"🔍 [{thread_id[:8]}] [{node_name}] {message}")

    def error(self, thread_id: str, node_name: str, error: Exception):
        """错误记录"""
        self.logger.error(f"❌ [{thread_id[:8]}] [{node_name}] 异常: {str(error)[:100]}...", exc_info=True)


__all__ = ["setup_logging", "get_logger", "WorkflowLogger"]

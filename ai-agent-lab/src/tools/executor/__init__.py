"""
=== 工具执行器层 ===

企业级工具执行引擎，负责统一的工具调用、日志、监控、超时、限流和权限控制。

【功能】：
1. 统一工具调用接口
2. 执行日志记录
3. 性能监控指标
4. 超时控制
5. 限流控制
6. 权限校验
7. 重试机制

【设计原则】：
1. 统一入口：所有工具调用都经过此执行器
2. 可观测性：完整的日志和监控支持
3. 安全性：权限控制和参数校验
4. 可靠性：超时、限流和重试机制
"""

import asyncio
import time
from typing import Any, Dict, Optional, Type, List
from datetime import datetime
import logging
from functools import wraps

from src.tools.base import BaseTool, ToolOutput
from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# 延迟导入，避免循环依赖
_skill_manager = None
_skill_adapter = None

def _get_skill_manager():
    """获取 Skill 管理器（延迟加载）"""
    global _skill_manager
    if _skill_manager is None:
        try:
            from src.tools.skills import SkillManager
            _skill_manager = SkillManager()
            _skill_manager.initialize()
        except ImportError:
            pass
    return _skill_manager

def _get_skill_adapter():
    """获取 Skill LangChain 适配器（延迟加载）"""
    global _skill_adapter
    if _skill_adapter is None:
        try:
            from agent_skills_sdk.adapters.langchain import LangChainAdapter
            skill_manager = _get_skill_manager()
            if skill_manager:
                _skill_adapter = LangChainAdapter(skill_paths=[skill_manager.skills_dir])
        except ImportError:
            pass
    return _skill_adapter


class ExecutionContext:
    """
    执行上下文
    
    包含工具执行的所有上下文信息
    """
    
    def __init__(self, tool_name: str, params: Dict[str, Any]):
        self.tool_name = tool_name
        self.params = params
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
        self.status = "running"  # running, completed, failed, timeout
        self.result = None
        self.error = None
        self.retries = 0
    
    def complete(self, result: Any):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "completed"
        self.result = result
    
    def fail(self, error: Exception):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "failed"
        self.error = error
    
    def timeout(self):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "timeout"


class ExecutionResult:
    """
    执行结果封装
    """
    
    def __init__(self, context: ExecutionContext):
        self.context = context
        self.success = context.status == "completed"
        self.result = context.result
        self.error = context.error
        self.duration = context.duration
        self.retries = context.retries
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.context.tool_name,
            "success": self.success,
            "result": self.result.model_dump() if hasattr(self.result, 'model_dump') else self.result,
            "error": str(self.error) if self.error else None,
            "duration": self.duration,
            "retries": self.retries,
            "timestamp": self.context.start_time.isoformat()
        }


class RateLimiter:
    """
    限流控制器
    
    基于令牌桶算法实现
    """
    
    def __init__(self, max_calls: int, period: int = 1):
        """
        Args:
            max_calls: 周期内最大调用次数
            period: 周期时间（秒），默认为 1 秒
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        获取调用许可
        
        Returns:
            bool: 是否获取到许可
        """
        async with self._lock:
            now = time.time()
            
            # 移除过期的调用记录
            self.calls = [c for c in self.calls if now - c < self.period]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False


class ToolExecutor:
    """
    工具执行器
    
    负责工具的统一调用、监控和控制
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._rate_limiters: Dict[str, RateLimiter] = {}  # 工具名称 -> 限流器
        self._middleware = []  # 中间件列表
        self._initialized = True
        logger.info("工具执行器已初始化")
    
    def add_middleware(self, middleware):
        """
        添加中间件
        
        Args:
            middleware: 中间件对象，需实现 before_execute 和 after_execute 方法
        """
        self._middleware.append(middleware)
        logger.info(f"中间件 {middleware.__class__.__name__} 已添加")
    
    def remove_middleware(self, middleware_type: Type):
        """
        移除中间件
        
        Args:
            middleware_type: 中间件类型
        """
        self._middleware = [m for m in self._middleware if not isinstance(m, middleware_type)]
        logger.info(f"中间件类型 {middleware_type.__name__} 已移除")
    
    async def _get_rate_limiter(self, tool_name: str) -> Optional[RateLimiter]:
        """
        获取工具的限流器
        
        Args:
            tool_name: 工具名称
        
        Returns:
            Optional[RateLimiter]: 限流器，如果没有配置则返回 None
        """
        if tool_name in self._rate_limiters:
            return self._rate_limiters[tool_name]
        
        # 从工具元数据获取限流配置
        tool_class = tool_registry.get_tool(tool_name)
        if tool_class:
            rate_limit = tool_class.get_metadata().rate_limit
            if rate_limit:
                self._rate_limiters[tool_name] = RateLimiter(rate_limit)
                return self._rate_limiters[tool_name]
        
        return None
    
    async def _check_permissions(self, tool_name: str, **kwargs) -> bool:
        """
        检查权限
        
        Args:
            tool_name: 工具名称
            **kwargs: 其他参数（可能包含用户信息等）
        
        Returns:
            bool: 是否有权限
        """
        # 检查工具是否需要认证
        tool_class = tool_registry.get_tool(tool_name)
        if tool_class and tool_class.get_metadata().requires_auth:
            # 检查是否提供了认证信息
            if "user_id" not in kwargs and "api_key" not in kwargs:
                logger.warning(f"工具 {tool_name} 需要认证，但未提供认证信息")
                return False
        
        return True
    
    async def execute(self, tool_name: str, **kwargs) -> ExecutionResult:
        """
        执行工具（统一入口）
        
        支持执行：
        1. 注册到 tool_registry 的工具
        2. 通过 agent_skills_sdk 安装的 Skill
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
        
        Returns:
            ExecutionResult: 执行结果
        """
        context = ExecutionContext(tool_name, kwargs)
        
        try:
            # 1. 权限检查
            if not await self._check_permissions(tool_name, **kwargs):
                raise PermissionError(f"没有权限调用工具 {tool_name}")
            
            # 2. 限流检查
            rate_limiter = await self._get_rate_limiter(tool_name)
            if rate_limiter and not await rate_limiter.acquire():
                raise RuntimeError(f"工具 {tool_name} 调用过于频繁，请稍后再试")
            
            # 3. 获取工具实例（先尝试注册的工具，再尝试 Skill）
            tool_instance = tool_registry.get_tool_instance(tool_name)
            is_skill = False
            
            if not tool_instance:
                # 尝试从 Skill 管理器获取
                skill_manager = _get_skill_manager()
                if skill_manager:
                    tool_instance = await self._get_skill_tool(tool_name, skill_manager)
                    is_skill = True
            
            if not tool_instance:
                raise ValueError(f"工具 {tool_name} 不存在")
            
            # 4. 获取超时配置（Skill 使用默认超时）
            timeout = 30  # 默认超时
            if not is_skill and hasattr(tool_instance, 'metadata') and tool_instance.metadata:
                timeout = getattr(tool_instance.metadata, 'timeout', 30)
            
            # 5. 执行前中间件
            for middleware in self._middleware:
                if hasattr(middleware, 'before_execute'):
                    await middleware.before_execute(context)
            
            # 6. 执行工具（带超时）
            try:
                async with asyncio.timeout(timeout):
                    if is_skill:
                        # Skill 执行 - 使用 ainvoke
                        result = await tool_instance.ainvoke(kwargs)
                    else:
                        # 注册工具执行 - 使用 ainvoke
                        result = await tool_instance.ainvoke(kwargs)
            except asyncio.TimeoutError:
                context.timeout()
                raise TimeoutError(f"工具 {tool_name} 执行超时（{timeout}秒）")
            
            # 7. 完成执行（设置 duration）
            context.complete(result)
            
            # 8. 执行后中间件（此时 context.duration 已设置）
            for middleware in reversed(self._middleware):
                if hasattr(middleware, 'after_execute'):
                    result = await middleware.after_execute(context, result)
            
            return ExecutionResult(context)
        
        except Exception as e:
            context.fail(e)
            
            # 执行后中间件（即使失败）
            for middleware in reversed(self._middleware):
                if hasattr(middleware, 'on_error'):
                    await middleware.on_error(context, e)
            
            return ExecutionResult(context)
    
    async def _get_skill_tool(self, tool_name: str, skill_manager) -> Optional[BaseTool]:
        """
        从 Skill 管理器获取工具

        Args:
            tool_name: 工具名称
            skill_manager: SkillManager 实例

        Returns:
            Optional[BaseTool]: Skill 工具实例，如果未找到返回 None
        """
        try:
            adapter = _get_skill_adapter()
            if not adapter:
                return None

            skills_tools = adapter.as_langchain_tools()

            # 查找匹配的工具
            for tool in skills_tools:
                if tool.name == tool_name or tool.name.lower() == tool_name.lower():
                    return tool

            return None
        except Exception as e:
            logger.error(f"获取 Skill 工具失败: {e}")
            return None
    
    async def execute_with_retry(self, tool_name: str, max_retries: int = 3, **kwargs) -> ExecutionResult:
        """
        带重试的工具执行
        
        Args:
            tool_name: 工具名称
            max_retries: 最大重试次数
            **kwargs: 工具参数
        
        Returns:
            ExecutionResult: 执行结果
        """
        result = None
        
        for attempt in range(max_retries + 1):
            result = await self.execute(tool_name, **kwargs)
            
            if result.success:
                break
            
            if attempt < max_retries:
                logger.warning(f"工具 {tool_name} 执行失败，第 {attempt + 1} 次尝试，将重试")
                result.context.retries += 1
                await asyncio.sleep(2 ** attempt)  # 指数退避
            
        return result
    
    async def batch_execute(self, requests: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """
        批量执行工具
        
        Args:
            requests: 请求列表，每个请求包含 tool_name 和参数
        
        Returns:
            List[ExecutionResult]: 执行结果列表
        """
        tasks = []
        for req in requests:
            tool_name = req.get("tool_name")
            params = req.get("params", {})
            tasks.append(self.execute(tool_name, **params))
        
        return await asyncio.gather(*tasks)


# 全局单例实例
tool_executor = ToolExecutor()


# 便捷函数
async def call_tool(tool_name: str, **kwargs) -> ExecutionResult:
    """
    调用工具（便捷接口）
    
    Args:
        tool_name: 工具名称
        **kwargs: 工具参数
    
    Returns:
        ExecutionResult: 执行结果
    """
    return await tool_executor.execute(tool_name, **kwargs)


async def call_tool_with_retry(tool_name: str, max_retries: int = 3, **kwargs) -> ExecutionResult:
    """
    带重试调用工具（便捷接口）
    
    Args:
        tool_name: 工具名称
        max_retries: 最大重试次数
        **kwargs: 工具参数
    
    Returns:
        ExecutionResult: 执行结果
    """
    return await tool_executor.execute_with_retry(tool_name, max_retries, **kwargs)


# 导出列表
__all__ = [
    "ToolExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "RateLimiter",
    "tool_executor",
    "call_tool",
    "call_tool_with_retry",
]

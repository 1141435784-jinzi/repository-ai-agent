"""
=== Agent Middleware 模块 ===

【职责】：
1. 提供企业级中间件实现
2. 支持审计日志、RAG注入、成本控制等横切关注点
3. 统一的中间件注册和管理机制

【设计原则】：
1. 可插拔：中间件可以自由组合
2. 无侵入：不修改核心业务逻辑
3. 可扩展：支持自定义中间件
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Type
from langchain.agents.factory import AgentMiddleware
from langchain_core.messages import BaseMessage, SystemMessage

# 配置日志
logger = logging.getLogger("agent_middleware")
logger.setLevel(logging.INFO)


class AuditMiddleware(AgentMiddleware):
    """企业级审计日志中间件 —— 记录 Agent 的每一步操作"""

    async def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs):
        """工具调用前：记录操作意图"""
        logger.info(
            f"[AUDIT] Tool Start | tool={tool_name} | "
            f"input={tool_input} | time={datetime.now().isoformat()}"
        )

    async def on_tool_end(self, tool_name: str, tool_output: Any, **kwargs):
        """工具调用后：记录执行结果"""
        output_str = str(tool_output)[:200] if tool_output else ""
        logger.info(
            f"[AUDIT] Tool End | tool={tool_name} | "
            f"output={output_str} | time={datetime.now().isoformat()}"
        )

    async def on_model_start(self, messages: List[BaseMessage], **kwargs):
        """模型调用前：记录请求信息"""
        user_query = ""
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == 'user':
                user_query = msg.content[:100]
                break
        logger.info(
            f"[AUDIT] Model Start | query={user_query} | "
            f"time={datetime.now().isoformat()}"
        )
        return messages

    async def on_model_end(self, response, **kwargs):
        """模型调用后：记录响应信息"""
        if hasattr(response, 'content'):
            content = response.content[:100]
        else:
            content = str(response)[:100]
        logger.info(
            f"[AUDIT] Model End | response={content} | "
            f"time={datetime.now().isoformat()}"
        )


class CostControlMiddleware(AgentMiddleware):
    """成本控制中间件 —— 控制 Agent 的 Token 消耗，防止失控"""

    def __init__(self, max_tokens: int = 100_000, max_iterations: int = 20):
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.total_tokens = 0
        self.iteration_count = 0

    async def on_model_end(self, response, **kwargs):
        """模型调用后：统计 Token 消耗"""
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            self.total_tokens += usage.get("total_tokens", 0)

        self.iteration_count += 1

        if self.total_tokens > self.max_tokens:
            raise RuntimeError(
                f"Token 消耗超限: {self.total_tokens}/{self.max_tokens}"
            )
        if self.iteration_count > self.max_iterations:
            raise RuntimeError(
                f"迭代次数超限: {self.iteration_count}/{self.max_iterations}"
            )

        logger.info(
            f"[COST] Token消耗: {self.total_tokens}/{self.max_tokens} | "
            f"迭代次数: {self.iteration_count}/{self.max_iterations}"
        )

    def reset(self):
        """重置计数器"""
        self.total_tokens = 0
        self.iteration_count = 0


class PermissionMiddleware(AgentMiddleware):
    """权限校验中间件 —— 在工具调用前检查用户权限"""

    def __init__(self, required_permissions: Dict[str, List[str]] = None):
        """
        Args:
            required_permissions: 工具名到权限列表的映射
                例如: {"restart_service": ["ops_admin"], "query_logs": ["viewer", "admin"]}
        """
        self.required_permissions = required_permissions or {}

    async def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs):
        """工具调用前：检查权限"""
        required_perms = self.required_permissions.get(tool_name, [])
        if not required_perms:
            return  # 无权限要求，直接通过

        # 获取当前用户权限（从配置或上下文中获取）
        current_permissions = kwargs.get('config', {}).get(
            'configurable', {}).get('permissions', [])

        # 检查是否有任一必需权限
        has_permission = any(perm in current_permissions for perm in required_perms)
        if not has_permission:
            raise PermissionError(
                f"无权执行工具 {tool_name}，需要权限: {required_perms}，当前权限: {current_permissions}"
            )

        logger.info(f"[PERMISSION] 权限校验通过: {tool_name}")


class SummarizationMiddleware(AgentMiddleware):
    """响应摘要中间件 —— 对长响应进行自动摘要，便于日志记录和审计"""

    def __init__(self, max_length: int = 200, summary_trigger_length: int = 500):
        """
        Args:
            max_length: 摘要的最大长度（字符数）
            summary_trigger_length: 触发摘要的响应长度阈值
        """
        self.max_length = max_length
        self.summary_trigger_length = summary_trigger_length
        self._summary_history: Dict[str, List[str]] = {}  # 按agent name记录历史摘要

    async def on_model_end(self, response, **kwargs):
        """模型调用后：对长响应进行摘要"""
        if not hasattr(response, 'content'):
            return response

        content = response.content
        if not content or len(content) < self.summary_trigger_length:
            return response

        # 生成摘要
        summary = self._summarize(content)
        agent_name = kwargs.get('config', {}).get('configurable', {}).get('agent_name', 'unknown')

        # 记录到历史
        if agent_name not in self._summary_history:
            self._summary_history[agent_name] = []
        self._summary_history[agent_name].append(summary)

        # 保持历史记录在合理范围
        if len(self._summary_history[agent_name]) > 100:
            self._summary_history[agent_name] = self._summary_history[agent_name][-100:]

        logger.info(
            f"[SUMMARY] 响应摘要 | agent={agent_name} | "
            f"原始长度={len(content)} | 摘要={summary} | time={datetime.now().isoformat()}"
        )

        return response

    def _summarize(self, content: str) -> str:
        """生成响应摘要"""
        if len(content) <= self.max_length:
            return content

        # 简单截断并添加省略号
        return content[:self.max_length] + "..."

    def get_summary_history(self, agent_name: str) -> List[str]:
        """获取指定Agent的摘要历史"""
        return self._summary_history.get(agent_name, [])

    def clear_history(self, agent_name: Optional[str] = None):
        """清除摘要历史"""
        if agent_name:
            self._summary_history.pop(agent_name, None)
        else:
            self._summary_history.clear()


class ToolSelectionMiddleware(AgentMiddleware):
    """工具选择中间件 —— 根据上下文动态选择可用工具，实现精细化权限控制"""

    def __init__(
        self,
        enabled_tools: Optional[List[str]] = None,
        disabled_tools: Optional[List[str]] = None,
        tool_selector: Optional[callable] = None,
        max_tools_per_invocation: Optional[int] = None
    ):
        """
        Args:
            enabled_tools: 启用的工具名称列表（白名单），为空表示不限制
            disabled_tools: 禁用的工具名称列表（黑名单）
            tool_selector: 自定义工具选择函数，签名为 (query: str, tools: List) -> List[str]
                返回值为允许使用的工具名称列表
            max_tools_per_invocation: 每次调用最多使用的工具数量
        """
        self._enabled_tools = set(enabled_tools) if enabled_tools else None
        self._disabled_tools = set(disabled_tools) if disabled_tools else set()
        self._tool_selector = tool_selector
        self._max_tools_per_invocation = max_tools_per_invocation

    async def on_model_start(self, messages: List[BaseMessage], tools: List[Any] = None, **kwargs):
        """模型调用前：动态过滤工具列表"""
        if not tools:
            return messages

        # 获取用户查询用于自定义选择
        user_query = ""
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == 'user':
                user_query = msg.content
                break

        # 应用白名单过滤
        filtered_tools = tools
        if self._enabled_tools:
            filtered_tools = [t for t in filtered_tools if getattr(t, 'name', str(t)) in self._enabled_tools]
            logger.info(f"[TOOL_SELECT] 白名单过滤后可用工具: {[getattr(t, 'name', str(t)) for t in filtered_tools]}")

        # 应用黑名单过滤
        if self._disabled_tools:
            filtered_tools = [t for t in filtered_tools if getattr(t, 'name', str(t)) not in self._disabled_tools]
            logger.info(f"[TOOL_SELECT] 黑名单过滤后可用工具: {[getattr(t, 'name', str(t)) for t in filtered_tools]}")

        # 应用自定义选择器
        if self._tool_selector and user_query:
            selected_names = self._tool_selector(user_query, [getattr(t, 'name', str(t)) for t in filtered_tools])
            if selected_names:
                filtered_tools = [t for t in filtered_tools if getattr(t, 'name', str(t)) in selected_names]
                logger.info(f"[TOOL_SELECT] 自定义选择后可用工具: {[getattr(t, 'name', str(t)) for t in filtered_tools]}")

        # 应用最大工具数量限制
        if self._max_tools_per_invocation and len(filtered_tools) > self._max_tools_per_invocation:
            filtered_tools = filtered_tools[:self._max_tools_per_invocation]
            logger.info(f"[TOOL_SELECT] 数量限制后可用工具: {[getattr(t, 'name', str(t)) for t in filtered_tools]}")

        # 将过滤后的工具列表存入config供后续使用
        config = kwargs.get('config', {})
        if 'configurable' not in config:
            config['configurable'] = {}
        config['configurable']['_filtered_tools'] = filtered_tools
        kwargs['config'] = config

        logger.info(f"[TOOL_SELECT] 最终可用工具数量: {len(filtered_tools)}")
        return messages

    def enable_tool(self, tool_name: str):
        """启用指定工具"""
        if self._enabled_tools is None:
            self._enabled_tools = set()
        self._enabled_tools.add(tool_name)
        self._disabled_tools.discard(tool_name)

    def disable_tool(self, tool_name: str):
        """禁用指定工具"""
        if self._disabled_tools is None:
            self._disabled_tools = set()
        self._disabled_tools.add(tool_name)
        if self._enabled_tools:
            self._enabled_tools.discard(tool_name)

    def get_available_tools(self) -> Optional[set]:
        """获取当前启用的工具集合"""
        return self._enabled_tools.copy() if self._enabled_tools else None

    def get_disabled_tools(self) -> set:
        """获取当前禁用的工具集合"""
        return self._disabled_tools.copy()


class RetryMiddleware(AgentMiddleware):
    """工具调用重试中间件 —— 为工具调用提供统一的重试机制"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 10.0,
        retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始重试延迟（秒）
            backoff_factor: 指数退避因子
            max_delay: 最大重试延迟（秒）
            retry_exceptions: 需要重试的异常类型元组
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_exceptions = retry_exceptions
        self._retry_state: Dict[str, int] = {}  # 记录每个工具的重试次数

    async def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs):
        """工具调用前：初始化重试状态"""
        thread_id = kwargs.get('config', {}).get('configurable', {}).get('thread_id', 'default')
        key = f"{thread_id}_{tool_name}"
        self._retry_state[key] = 0

    async def on_tool_error(self, tool_name: str, error: Exception, **kwargs) -> bool:
        """工具调用出错时：判断是否重试并执行重试逻辑
        
        Returns:
            bool: True 表示需要重试，False 表示不再重试
        """
        # 检查异常类型是否在重试列表中
        if not isinstance(error, self.retry_exceptions):
            logger.info(f"[RETRY] 异常类型不在重试列表中: {type(error).__name__}")
            return False

        thread_id = kwargs.get('config', {}).get('configurable', {}).get('thread_id', 'default')
        key = f"{thread_id}_{tool_name}"
        retry_count = self._retry_state.get(key, 0)

        if retry_count >= self.max_retries:
            logger.error(f"[RETRY] 工具调用失败，已达最大重试次数 {self.max_retries}: {tool_name}")
            self._retry_state.pop(key, None)
            return False

        # 计算重试延迟（指数退避）
        delay = min(
            self.initial_delay * (self.backoff_factor ** retry_count),
            self.max_delay
        )

        logger.warning(
            f"[RETRY] 工具调用失败，第 {retry_count + 1}/{self.max_retries} 次重试，"
            f"延迟 {delay:.2f}s: {tool_name}"
        )

        await asyncio.sleep(delay)
        self._retry_state[key] = retry_count + 1

        return True

    async def on_tool_end(self, tool_name: str, tool_output: Any, **kwargs):
        """工具调用成功后：清除重试状态"""
        thread_id = kwargs.get('config', {}).get('configurable', {}).get('thread_id', 'default')
        key = f"{thread_id}_{tool_name}"
        self._retry_state.pop(key, None)

    def get_retry_count(self, tool_name: str, thread_id: str = 'default') -> int:
        """获取指定工具的重试次数"""
        key = f"{thread_id}_{tool_name}"
        return self._retry_state.get(key, 0)


class HITLMiddleware(AgentMiddleware):
    """人工干预中间件 (Human-in-the-Loop) —— 在关键决策点暂停执行，等待人工确认"""

    def __init__(
        self,
        enabled: bool = True,
        pause_on_tools: Optional[List[str]] = None,
        pause_on_high_cost: bool = True,
        cost_threshold: float = 0.01,
        approval_callback: Optional[callable] = None,
        rejection_callback: Optional[callable] = None
    ):
        """
        Args:
            enabled: 是否启用 HITL
            pause_on_tools: 触发暂停的工具名称列表（为空表示所有工具）
            pause_on_high_cost: 是否在高成本操作时暂停
            cost_threshold: 成本阈值（单位：美元），超过此阈值触发暂停
            approval_callback: 人工批准时的回调函数，签名为 (tool_name, tool_input, context) -> bool
            rejection_callback: 人工拒绝时的回调函数，签名为 (tool_name, tool_input, context) -> None
        """
        self.enabled = enabled
        self._pause_on_tools = set(pause_on_tools) if pause_on_tools else None
        self._pause_on_high_cost = pause_on_high_cost
        self._cost_threshold = cost_threshold
        self._approval_callback = approval_callback
        self._rejection_callback = rejection_callback
        self._pending_approvals: Dict[str, dict] = {}
        self._approval_results: Dict[str, bool] = {}

    async def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs) -> dict:
        """工具调用前：检查是否需要人工干预"""
        if not self.enabled:
            return {"hitl_approved": True}

        config = kwargs.get('config', {})
        thread_id = config.get('configurable', {}).get('thread_id', 'default')
        approval_key = f"{thread_id}_{tool_name}_{id(tool_input)}"

        needs_approval = False
        approval_reason = ""

        if self._pause_on_tools and tool_name in self._pause_on_tools:
            needs_approval = True
            approval_reason = f"工具 '{tool_name}' 需要人工确认"
        elif self._pause_on_tools is None:
            needs_approval = True
            approval_reason = f"所有工具调用需要人工确认"

        if not needs_approval and self._pause_on_high_cost:
            estimated_cost = self._estimate_cost(tool_name, tool_input)
            if estimated_cost > self._cost_threshold:
                needs_approval = True
                approval_reason = f"预估成本 ${estimated_cost:.4f} 超过阈值 ${self._cost_threshold:.4f}"

        if needs_approval:
            logger.info(f"[HITL] ⏸️ 暂停等待人工确认: {tool_name} - {approval_reason}")

            self._pending_approvals[approval_key] = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "reason": approval_reason,
                "thread_id": thread_id
            }

            if self._approval_callback:
                result = self._approval_callback(tool_name, tool_input, {"reason": approval_reason})
                if result is True:
                    logger.info(f"[HITL] ✅ 回调批准: {tool_name}")
                    self._approval_results[approval_key] = True
                    self._pending_approvals.pop(approval_key, None)
                    return {"hitl_approved": True}
                elif result is False:
                    logger.info(f"[HITL] ❌ 回调拒绝: {tool_name}")
                    self._approval_results[approval_key] = False
                    self._pending_approvals.pop(approval_key, None)
                    if self._rejection_callback:
                        self._rejection_callback(tool_name, tool_input, {"reason": approval_reason})
                    return {"hitl_approved": False, "hitl_rejected": True}

            return {
                "hitl_pending": True,
                "approval_key": approval_key,
                "approval_reason": approval_reason
            }

        return {"hitl_approved": True}

    async def on_tool_end(self, tool_name: str, tool_output: Any, **kwargs):
        """工具调用后：清理待审批状态"""
        config = kwargs.get('config', {})
        thread_id = config.get('configurable', {}).get('thread_id', 'default')
        approval_key_prefix = f"{thread_id}_{tool_name}_"

        keys_to_remove = [k for k in self._pending_approvals.keys() if k.startswith(approval_key_prefix)]
        for k in keys_to_remove:
            self._pending_approvals.pop(k, None)

    def approve(self, approval_key: str) -> bool:
        """批准待处理的请求
        
        Args:
            approval_key: 审批密钥
            
        Returns:
            bool: 是否批准成功
        """
        if approval_key in self._pending_approvals:
            logger.info(f"[HITL] ✅ 人工批准: {approval_key}")
            self._approval_results[approval_key] = True
            return True
        return False

    def reject(self, approval_key: str) -> bool:
        """拒绝待处理的请求
        
        Args:
            approval_key: 审批密钥
            
        Returns:
            bool: 是否拒绝成功
        """
        if approval_key in self._pending_approvals:
            logger.info(f"[HITL] ❌ 人工拒绝: {approval_key}")
            pending = self._pending_approvals[approval_key]
            self._approval_results[approval_key] = False
            if self._rejection_callback:
                self._rejection_callback(pending["tool_name"], pending["tool_input"], {})
            return True
        return False

    def get_pending_approvals(self) -> List[dict]:
        """获取所有待审批请求"""
        return list(self._pending_approvals.values())

    def is_approved(self, approval_key: str) -> Optional[bool]:
        """检查审批结果"""
        return self._approval_results.get(approval_key)

    def _estimate_cost(self, tool_name: str, tool_input: dict) -> float:
        """估算工具调用的成本（简化版本）"""
        high_cost_tools = {
            "search": 0.001,
            "query_knowledge_base": 0.002,
            "browser": 0.005,
            "execute_code": 0.01
        }
        return high_cost_tools.get(tool_name, 0.0001)

    def set_enabled(self, enabled: bool):
        """启用/禁用 HITL"""
        self.enabled = enabled
        logger.info(f"[HITL] HITL {'启用' if enabled else '禁用'}")


__all__ = [
    "AuditMiddleware",
    "CostControlMiddleware",
    "PermissionMiddleware",
    "SummarizationMiddleware",
    "ToolSelectionMiddleware",
    "RetryMiddleware",
    "HITLMiddleware"
]

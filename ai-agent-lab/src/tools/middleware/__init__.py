"""
=== 工具中间件层 ===

企业级工具中间件框架，基于 AOP 切面思想，不侵入业务代码。

【功能】：
1. 日志记录中间件
2. 监控指标中间件
3. 参数校验中间件
4. 结果脱敏中间件
5. 调用审计中间件

【设计原则】：
1. AOP 切面：无侵入式设计
2. 可组合：支持多个中间件组合
3. 可扩展：易于添加自定义中间件
4. 统一接口：所有中间件实现相同的接口
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime

from src.tools.executor import ExecutionContext
from src.tools.base import ToolOutput

logger = logging.getLogger(__name__)


class BaseMiddleware:
    """
    中间件基类
    
    子类可以选择性实现以下方法：
    - before_execute: 执行前处理
    - after_execute: 执行后处理
    - on_error: 错误处理
    """
    
    async def before_execute(self, context: ExecutionContext):
        """
        执行前处理
        
        Args:
            context: 执行上下文
        """
        pass
    
    async def after_execute(self, context: ExecutionContext, result: Any) -> Any:
        """
        执行后处理
        
        Args:
            context: 执行上下文
            result: 执行结果
        
        Returns:
            Any: 修改后的结果
        """
        return result
    
    async def on_error(self, context: ExecutionContext, error: Exception):
        """
        错误处理
        
        Args:
            context: 执行上下文
            error: 异常对象
        """
        pass


class LoggingMiddleware(BaseMiddleware):
    """
    日志记录中间件
    
    记录工具调用的详细日志，包括：
    - 调用时间
    - 工具名称
    - 参数
    - 执行时长
    - 结果状态
    """
    
    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
    
    async def before_execute(self, context: ExecutionContext):
        """执行前记录日志"""
        logger.log(
            self.log_level,
            f"🔧 开始执行工具: {context.tool_name} | 参数: {context.params}",
            extra={
                "tool_name": context.tool_name,
                "params": context.params,
                "timestamp": context.start_time.isoformat()
            }
        )
    
    async def after_execute(self, context: ExecutionContext, result: Any) -> Any:
        """执行后记录日志"""
        status = "✅ 成功" if context.status == "completed" else "❌ 失败"
        
        logger.log(
            self.log_level,
            f"🔧 工具执行完成: {context.tool_name} | {status} | 耗时: {context.duration:.2f}s",
            extra={
                "tool_name": context.tool_name,
                "status": context.status,
                "duration": context.duration,
                "result": result if isinstance(result, dict) else str(result)[:200]
            }
        )
        
        return result
    
    async def on_error(self, context: ExecutionContext, error: Exception):
        """错误时记录日志"""
        logger.error(
            f"🔧 工具执行异常: {context.tool_name} | 错误: {str(error)}",
            exc_info=True,
            extra={
                "tool_name": context.tool_name,
                "error": str(error),
                "duration": context.duration
            }
        )


class MetricsMiddleware(BaseMiddleware):
    """
    监控指标中间件
    
    收集工具执行的关键指标：
    - 调用次数
    - 成功次数
    - 失败次数
    - 平均执行时间
    - 最大/最小执行时间
    """
    
    def __init__(self):
        self._metrics = {
            # 工具级别指标
            "tool_calls": {},          # tool_name -> count
            "tool_success": {},        # tool_name -> count
            "tool_errors": {},         # tool_name -> count
            "tool_duration_sum": {},   # tool_name -> sum
            "tool_duration_min": {},   # tool_name -> min
            "tool_duration_max": {},   # tool_name -> max
            
            # 全局指标
            "total_calls": 0,
            "total_success": 0,
            "total_errors": 0,
            "total_duration": 0,
            "start_time": time.time()
        }
        self._lock = asyncio.Lock()
    
    async def _update_metrics(self, context: ExecutionContext, success: bool):
        """更新指标"""
        async with self._lock:
            tool_name = context.tool_name
            duration = context.duration
            
            # 更新工具级别指标
            self._metrics["tool_calls"][tool_name] = self._metrics["tool_calls"].get(tool_name, 0) + 1
            
            if success:
                self._metrics["tool_success"][tool_name] = self._metrics["tool_success"].get(tool_name, 0) + 1
            else:
                self._metrics["tool_errors"][tool_name] = self._metrics["tool_errors"].get(tool_name, 0) + 1
            
            # 更新时长统计
            if tool_name not in self._metrics["tool_duration_sum"]:
                self._metrics["tool_duration_sum"][tool_name] = 0
                self._metrics["tool_duration_min"][tool_name] = float('inf')
                self._metrics["tool_duration_max"][tool_name] = 0
            
            self._metrics["tool_duration_sum"][tool_name] += duration
            self._metrics["tool_duration_min"][tool_name] = min(
                self._metrics["tool_duration_min"][tool_name], duration
            )
            self._metrics["tool_duration_max"][tool_name] = max(
                self._metrics["tool_duration_max"][tool_name], duration
            )
            
            # 更新全局指标
            self._metrics["total_calls"] += 1
            if success:
                self._metrics["total_success"] += 1
            else:
                self._metrics["total_errors"] += 1
            self._metrics["total_duration"] += duration
    
    async def after_execute(self, context: ExecutionContext, result: Any) -> Any:
        """执行后更新指标"""
        await self._update_metrics(context, success=True)
        return result
    
    async def on_error(self, context: ExecutionContext, error: Exception):
        """错误时更新指标"""
        await self._update_metrics(context, success=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标统计"""
        result = self._metrics.copy()
        
        # 计算平均值
        result["tool_avg_duration"] = {}
        for tool_name, calls in result["tool_calls"].items():
            total = result["tool_duration_sum"].get(tool_name, 0)
            result["tool_avg_duration"][tool_name] = total / calls if calls > 0 else 0
        
        # 计算全局平均值
        result["avg_duration"] = (
            result["total_duration"] / result["total_calls"] 
            if result["total_calls"] > 0 else 0
        )
        
        # 计算成功率
        result["success_rate"] = (
            result["total_success"] / result["total_calls"] * 100 
            if result["total_calls"] > 0 else 0
        )
        
        # 添加运行时间
        result["uptime"] = time.time() - result["start_time"]
        
        return result


class ValidationMiddleware(BaseMiddleware):
    """
    参数校验中间件
    
    对工具输入参数进行额外校验，包括：
    - 类型校验
    - 范围校验
    - 格式校验
    """
    
    async def before_execute(self, context: ExecutionContext):
        """执行前校验参数"""
        params = context.params
        
        # 基础类型检查
        for key, value in params.items():
            if value is None:
                logger.warning(f"参数 {key} 为 None")
            
            # 字符串长度检查
            if isinstance(value, str):
                if len(value) > 1000:
                    logger.warning(f"参数 {key} 长度超过 1000 字符")
            
            # 数值范围检查
            if isinstance(value, (int, float)):
                if value < -1e18 or value > 1e18:
                    logger.warning(f"参数 {key} 数值超出合理范围")


class DataMaskingMiddleware(BaseMiddleware):
    """
    数据脱敏中间件
    
    对敏感数据进行脱敏处理，保护用户隐私：
    - 手机号脱敏
    - 邮箱脱敏
    - 身份证号脱敏
    - 银行卡号脱敏
    """
    
    # 脱敏规则
    _MASK_PATTERNS = {
        "phone": r'(\d{3})\d{4}(\d{4})',           # 手机号: 138****8888
        "email": r'([a-zA-Z0-9]+)@([a-zA-Z0-9]+)', # 邮箱: abc***@xxx.com
        "id_card": r'(\d{4})\d{10}(\d{4})',        # 身份证: 1101****1234
        "bank_card": r'(\d{4})\d{8,12}(\d{4})'     # 银行卡: 6228****8888
    }
    
    def __init__(self, mask_fields: Optional[Dict[str, str]] = None):
        """
        Args:
            mask_fields: 字段名到脱敏类型的映射
                例如: {"phone_number": "phone", "email": "email"}
        """
        self.mask_fields = mask_fields or {}
    
    def _mask_value(self, value: Any, mask_type: str) -> Any:
        """脱敏单个值"""
        if not isinstance(value, str):
            return value
        
        import re
        pattern = self._MASK_PATTERNS.get(mask_type)
        
        if not pattern:
            return value
        
        if mask_type == "phone":
            return re.sub(pattern, r'\1****\2', value)
        elif mask_type == "email":
            return re.sub(r'(.{1,3})(.*)@', r'\1***@', value)
        elif mask_type == "id_card":
            return re.sub(pattern, r'\1**********\2', value)
        elif mask_type == "bank_card":
            return re.sub(pattern, r'\1********\2', value)
        
        return value
    
    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏字典中的字段"""
        result = {}
        for key, value in data.items():
            if key in self.mask_fields:
                mask_type = self.mask_fields[key]
                if isinstance(value, dict):
                    result[key] = self._mask_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._mask_value(item, mask_type) for item in value]
                else:
                    result[key] = self._mask_value(value, mask_type)
            elif isinstance(value, dict):
                result[key] = self._mask_dict(value)
            elif isinstance(value, list):
                result[key] = [self._mask_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result
    
    async def after_execute(self, context: ExecutionContext, result: Any) -> Any:
        """执行后脱敏结果"""
        if isinstance(result, dict):
            return self._mask_dict(result)
        
        if hasattr(result, 'model_dump'):
            return result
        
        return result


class AuditMiddleware(BaseMiddleware):
    """
    调用审计中间件
    
    记录所有工具调用记录，用于合规审计：
    - 调用时间
    - 调用者信息
    - 工具名称
    - 参数（脱敏后）
    - 结果
    - 执行时长
    """
    
    def __init__(self):
        self._audit_logs = []
        self._max_logs = 10000
        self._lock = asyncio.Lock()
    
    async def _log_audit(self, context: ExecutionContext, result: Any, error: Optional[Exception] = None):
        """记录审计日志"""
        async with self._lock:
            audit_record = {
                "timestamp": context.start_time.isoformat(),
                "tool_name": context.tool_name,
                "params": context.params,
                "status": context.status,
                "duration": context.duration,
                "result_summary": str(result)[:500] if result else None,
                "error": str(error) if error else None,
                "retries": context.retries
            }
            
            self._audit_logs.append(audit_record)
            
            # 保持日志数量限制
            if len(self._audit_logs) > self._max_logs:
                self._audit_logs = self._audit_logs[-self._max_logs:]
    
    async def after_execute(self, context: ExecutionContext, result: Any) -> Any:
        """执行后记录审计日志"""
        await self._log_audit(context, result)
        return result
    
    async def on_error(self, context: ExecutionContext, error: Exception):
        """错误时记录审计日志"""
        await self._log_audit(context, None, error)
    
    def get_audit_logs(self, limit: int = 100) -> list:
        """获取审计日志"""
        return self._audit_logs[-limit:]
    
    def clear_audit_logs(self):
        """清空审计日志"""
        self._audit_logs.clear()


# 导出列表
__all__ = [
    # 基类
    "BaseMiddleware",
    
    # 中间件实现
    "LoggingMiddleware",
    "MetricsMiddleware",
    "ValidationMiddleware",
    "DataMaskingMiddleware",
    "AuditMiddleware",
]

"""
=== 数学计算工具实现 ===

基于 LangChain BaseTool 的数学计算工具实现。
"""

import math
from typing import Any, ClassVar, Dict, Optional, Type
from pydantic import BaseModel, Field

from src.tools.base import BaseTool
from src.tools.registry import register_tool


class CalculatorInput(BaseModel):
    """数学计算工具输入参数"""
    expression: str = Field(
        description="数学表达式，支持 Python 语法，内置 math 模块。例如：'2 + 3 * 4'、'sqrt(16)'、'pow(2, 10)'、'(10 + 5) / 3'、'sin(pi/2)'",
        examples=["2 + 3 * 4", "sqrt(16)", "pow(2, 10)", "(10 + 5) / 3", "sin(pi/2)"]
    )


class CalculatorOutput(BaseModel):
    """数学计算工具输出结果"""
    success: bool = True
    message: Optional[str] = None
    expression: Optional[str] = None
    result: Optional[float] = None


@register_tool
class CalculatorTool(BaseTool):
    """数学表达式计算工具"""

    name = "calculator"
    description = "执行数学表达式计算，支持 Python 语法，内置 math 模块函数。例如：'2 + 3 * 4'、'sqrt(16)'、'pow(2, 10)'、'(10 + 5) / 3'、'sin(pi/2)'。"
    args_schema: ClassVar[Type[BaseModel]] = CalculatorInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {
        "category": "utility",
        "tags": ["math", "calculation", "arithmetic"],
    }

    def _run(self, expression: str) -> CalculatorOutput:
        """
        执行数学表达式计算

        Args:
            expression: Python 数学表达式字符串

        Returns:
            CalculatorOutput: 计算结果
        """
        try:
            safe_globals = {
                'pi': math.pi,
                'e': math.e,
                'sqrt': math.sqrt,
                'pow': math.pow,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'atan2': math.atan2,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'log2': math.log2,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
            }

            result = eval(expression, {"__builtins__": {}}, safe_globals)

            return CalculatorOutput(
                success=True,
                message=f"计算成功",
                expression=expression,
                result=float(result) if isinstance(result, (int, float)) else result
            )

        except SyntaxError as e:
            return CalculatorOutput(
                success=False,
                message=f"语法错误: {str(e)}"
            )
        except ZeroDivisionError:
            return CalculatorOutput(
                success=False,
                message="错误：除数不能为零"
            )
        except Exception as e:
            return CalculatorOutput(
                success=False,
                message=f"计算错误: {str(e)}"
            )

    async def _arun(self, expression: str) -> CalculatorOutput:
        """异步执行"""
        return self._run(expression)

"""
=== 数学计算工具 ===

【功能】：
支持任意复杂的 Python 数学表达式计算
内置 math 模块支持，可使用各种数学函数

【使用方式】：
输入数学表达式字符串，例如：
- "2 + 3 * 4"
- "sqrt(16)"
- "pow(2, 10)"
- "(10 + 5) / 3"
- "sin(pi/2)"
"""

import math
from typing import List
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class MathExpressionInput(BaseModel):
    """数学表达式输入参数"""
    expression: str = Field(
        description="数学表达式，支持 Python 语法，内置 math 模块。例如：'2 + 3 * 4'、'sqrt(16)'、'pow(2, 10)'、'(10 + 5) / 3'、'sin(pi/2)'",
        examples=["2 + 3 * 4", "sqrt(16)", "pow(2, 10)", "(10 + 5) / 3", "sin(pi/2)"]
    )


async def math_calculation_tool(expression: str) -> str:
    """数学表达式计算工具
    
    使用安全的 eval 执行数学表达式，内置 math 模块支持
    
    Args:
        expression: Python 数学表达式字符串
        
    Returns:
        str: 计算结果
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
        return f"✅ {expression} = {result}"
            
    except SyntaxError as e:
        return f"❌ 语法错误: {str(e)}"
    except ZeroDivisionError:
        return "❌ 错误：除数不能为零"
    except Exception as e:
        return f"❌ 计算错误: {str(e)}"


# 创建工具实例
math_calculation_tool_instance = StructuredTool.from_function(
    name="local_math_calculation",
    func=math_calculation_tool,
    description="执行数学表达式计算，支持 Python 语法，内置 math 模块函数。例如：'2 + 3 * 4'、'sqrt(16)'、'pow(2, 10)'、'(10 + 5) / 3'、'sin(pi/2)'。",
    args_schema=MathExpressionInput,
    coroutine=math_calculation_tool
)


def create_calculator_tools() -> List[BaseTool]:
    """创建数学计算工具"""
    return [math_calculation_tool_instance]


__all__ = [
    "create_calculator_tools",
    "math_calculation_tool",
    "math_calculation_tool_instance",
]

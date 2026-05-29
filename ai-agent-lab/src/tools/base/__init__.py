"""
=== 工具基类层 ===

基于 LangChain BaseTool 的企业级工具系统。

【设计原则】：
1. 使用 LangChain BaseTool 作为基类
2. 类型安全：使用 Pydantic 进行参数验证
3. 异常处理：统一的错误处理机制
4. 可扩展性：支持同步和异步工具
"""

from typing import Any, Dict, Optional, Type, List, Callable, ClassVar
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool as LangChainBaseTool


class ToolMetadata(BaseModel):
    """工具元数据"""
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    author: Optional[str] = None
    tags: List[str] = []
    requires_auth: bool = False
    rate_limit: Optional[int] = None
    timeout: int = 30


class ToolInput(BaseModel):
    """工具输入参数基类"""
    pass


class ToolOutput(BaseModel):
    """工具输出结果基类"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None

    @classmethod
    def success_result(cls, data: Any = None, message: str = "执行成功") -> 'ToolOutput':
        return cls(success=True, data=data, message=message)

    @classmethod
    def error_result(cls, message: str, data: Any = None) -> 'ToolOutput':
        return cls(success=False, data=data, message=message)


class BaseTool(LangChainBaseTool):
    """
    工具基类 - 基于 LangChain BaseTool

    所有工具必须继承此类。

    子类必须定义：
    - name: 工具名称（类属性）
    - description: 工具描述（类属性）
    - args_schema: 输入参数 Pydantic 模型

    子类应实现：
    - _run: 同步执行逻辑
    - _arun: 异步执行逻辑（可选）
    """

    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "基础工具"
    args_schema: ClassVar[Optional[Type[BaseModel]]] = ToolInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = None

    def _run(self, **kwargs: Any) -> Any:
        """
        同步执行（默认实现）

        Args:
            **kwargs: 工具参数

        Returns:
            Any: 执行结果
        """
        raise NotImplementedError("子类必须实现 _run 方法")

    async def _arun(self, **kwargs: Any) -> Any:
        """
        异步执行（默认实现调用同步方法）

        Args:
            **kwargs: 工具参数

        Returns:
            Any: 执行结果
        """
        return self._run(**kwargs)

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        return ToolMetadata(name=cls.name, description=cls.description)

    @classmethod
    def to_function_calling_schema(cls) -> Dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式的 schema

        Returns:
            Dict[str, Any]: Function Calling 格式的 schema
        """
        input_model = cls.args_schema
        if input_model is None:
            input_model = ToolInput

        properties = {}
        required = []

        for field_name, field_info in input_model.model_fields.items():
            field_type = field_info.annotation
            json_type = "string"

            if field_type is int:
                json_type = "integer"
            elif field_type is float:
                json_type = "number"
            elif field_type is bool:
                json_type = "boolean"
            elif field_type is list:
                json_type = "array"
            elif field_type is dict:
                json_type = "object"

            properties[field_name] = {
                "type": json_type,
                "description": field_info.description or ""
            }

            if field_info.is_required():
                required.append(field_name)

        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class AsyncTool(BaseTool):
    """
    异步工具基类

    子类只需实现 _arun 方法
    """

    async def _arun(self, **kwargs: Any) -> Any:
        """
        异步执行逻辑

        Args:
            **kwargs: 工具参数

        Returns:
            Any: 执行结果
        """
        raise NotImplementedError("子类必须实现 _arun 方法")


class SyncTool(BaseTool):
    """
    同步工具基类

    子类只需实现 _run 方法
    """

    def _run(self, **kwargs: Any) -> Any:
        """
        同步执行逻辑

        Args:
            **kwargs: 工具参数

        Returns:
            Any: 执行结果
        """
        raise NotImplementedError("子类必须实现 _run 方法")


def create_tool(
    name: str,
    description: str,
    func: Callable,
    args_schema: Optional[Type[BaseModel]] = None
) -> BaseTool:
    """
    动态创建工具的工厂函数

    Args:
        name: 工具名称
        description: 工具描述
        func: 执行函数
        args_schema: 参数 schema（可选）

    Returns:
        BaseTool: 创建的工具实例
    """
    return type(
        name,
        (BaseTool,),
        {
            "name": name,
            "description": description,
            "args_schema": args_schema or ToolInput,
            "_run": lambda self, **kwargs: func(**kwargs)
        }
    )()


__all__ = [
    "BaseTool",
    "AsyncTool",
    "SyncTool",
    "ToolMetadata",
    "ToolInput",
    "ToolOutput",
    "create_tool",
]
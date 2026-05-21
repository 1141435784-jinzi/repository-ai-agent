"""
=== 第一层：工具基类层 ===

企业级工具系统的基础抽象层，定义所有工具必须遵循的标准接口。

【设计原则】：
1. 统一接口：所有工具继承 BaseTool
2. 类型安全：使用 Pydantic 进行参数验证
3. 异常处理：统一的错误处理机制
4. 可扩展性：支持同步和异步工具

【架构】：
BaseTool (抽象基类)
    ├── SyncTool (同步工具基类)
    └── AsyncTool (异步工具基类)
"""

from typing import Any, Dict, Optional, Type, TypeVar, Generic, List
from abc import ABC, abstractmethod
from pydantic import BaseModel, create_model
import json

# 工具类型变量
T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=BaseModel)


class ToolMetadata(BaseModel):
    """工具元数据"""
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    author: Optional[str] = None
    tags: List[str] = []
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # 每秒最大调用次数
    timeout: int = 30  # 超时时间（秒）


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


class BaseTool(ABC, Generic[T, U]):
    """
    工具基类（抽象）
    
    所有工具必须继承此类，实现统一的接口标准。
    
    泛型参数：
    - T: 输入参数类型（继承自 BaseModel）
    - U: 输出结果类型（继承自 BaseModel）
    
    子类必须实现：
    - name: 工具名称（类属性）
    - description: 工具描述（类属性）
    - input_schema: 输入参数 schema
    - _execute: 执行逻辑（同步或异步）
    """
    
    # 必须在子类中定义
    name: str = "base_tool"
    description: str = "基础工具"
    input_schema: Type[T] = ToolInput
    output_schema: Type[U] = ToolOutput
    metadata: ToolMetadata = ToolMetadata(name="base_tool", description="基础工具")
    
    @classmethod
    def get_name(cls) -> str:
        """获取工具名称"""
        return cls.name
    
    @classmethod
    def get_description(cls) -> str:
        """获取工具描述"""
        return cls.description
    
    @classmethod
    def get_input_schema(cls) -> Type[T]:
        """获取输入参数 schema"""
        return cls.input_schema
    
    @classmethod
    def get_output_schema(cls) -> Type[U]:
        """获取输出结果 schema"""
        return cls.output_schema
    
    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """获取工具元数据"""
        return cls.metadata
    
    @classmethod
    def to_function_calling_schema(cls) -> Dict[str, Any]:
        """
        转换为 OpenAI Function Calling 格式的 schema
        
        Returns:
            Dict[str, Any]: Function Calling 格式的 schema
        """
        input_model = cls.input_schema
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
    
    @abstractmethod
    async def execute(self, **kwargs) -> U:
        """
        执行工具（统一入口）
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        pass
    
    def validate_input(self, **kwargs) -> T:
        """
        验证输入参数
        
        Args:
            **kwargs: 输入参数
        
        Returns:
            T: 验证后的参数对象
        
        Raises:
            ValidationError: 参数验证失败
        """
        return self.input_schema(**kwargs)
    
    async def _execute_with_validation(self, **kwargs) -> U:
        """
        带参数验证的执行
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        try:
            validated_input = self.validate_input(**kwargs)
            return await self.execute(**validated_input.model_dump())
        except Exception as e:
            return self.output_schema.error_result(str(e))


class SyncTool(BaseTool[T, U], ABC):
    """
    同步工具基类
    
    子类只需实现 sync_execute 方法
    """
    
    @abstractmethod
    def sync_execute(self, **kwargs) -> U:
        """
        同步执行逻辑
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        pass
    
    async def execute(self, **kwargs) -> U:
        """
        执行工具（统一入口）
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        try:
            validated_input = self.validate_input(**kwargs)
            result = self.sync_execute(**validated_input.model_dump())
            return result
        except Exception as e:
            return self.output_schema.error_result(str(e))


class AsyncTool(BaseTool[T, U], ABC):
    """
    异步工具基类
    
    子类只需实现 async_execute 方法
    """
    
    @abstractmethod
    async def async_execute(self, **kwargs) -> U:
        """
        异步执行逻辑
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        pass
    
    async def execute(self, **kwargs) -> U:
        """
        执行工具（统一入口）
        
        Args:
            **kwargs: 工具参数
        
        Returns:
            U: 执行结果
        """
        try:
            validated_input = self.validate_input(**kwargs)
            result = await self.async_execute(**validated_input.model_dump())
            return result
        except Exception as e:
            return self.output_schema.error_result(str(e))


# 导出列表
__all__ = [
    # 基础类
    "BaseTool",
    "SyncTool",
    "AsyncTool",
    
    # 数据模型
    "ToolMetadata",
    "ToolInput",
    "ToolOutput",
]

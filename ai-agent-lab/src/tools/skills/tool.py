"""
自定义技能工具类 - 完全替代 agent-skills-sdk 的 SkillTool

兼容 Pydantic v2，所有字段必须提前声明
"""

from typing import Optional, Dict, Any, ClassVar, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SkillToolInput(BaseModel):
    """技能工具输入模型"""
    query: Optional[str] = Field(None, description="查询内容")


class SkillTool(BaseTool):
    """完全替代 agent-skills-sdk 的 SkillTool"""
    skill_id: str
    client: Any = Field(exclude=True)  # ✅ Pydantic v2 必须提前声明
    args_schema: ClassVar[Type[BaseModel]] = SkillToolInput

    def _run(self, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """实际执行技能"""
        return self.client.invoke(
            skill_id=self.skill_id,
            query=query,
            **kwargs
        )

    async def _arun(self, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return await self.client.ainvoke(
            skill_id=self.skill_id,
            query=query,
            **kwargs
        )

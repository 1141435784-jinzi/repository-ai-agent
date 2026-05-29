"""
Skill 管理系统 - 使用自定义实现，完全替代 agent-skills-sdk

提供 Skill 的安装、卸载、更新、搜索等功能。
兼容 Pydantic v2，避免版本兼容性问题。

核心组件：
- SkillManager: 主入口，提供完整的 Skill 生命周期管理
- SkillClient: 技能客户端，负责发现和执行技能
- SkillTool: LangChain 工具封装，兼容 Pydantic v2

使用示例：
```python
from src.tools.skills import SkillManager

# 初始化
manager = SkillManager()
manager.initialize()

# 列出所有已安装的技能
skills = manager.list_skills()

# 模糊搜索
results = manager.search_skills("数据分析")

# 检查是否安装
is_installed = manager.is_installed("my-skill")

# 转换为 LangChain 工具
tools = manager.to_langchain_tools()
```
"""

from .manager import SkillManager
from .models import SkillInfo, SkillStatus
from .client import SkillClient, SkillMetadata
from .tool import SkillTool

__all__ = [
    "SkillManager",
    "SkillInfo",
    "SkillStatus",
    "SkillClient",
    "SkillMetadata",
    "SkillTool",
]

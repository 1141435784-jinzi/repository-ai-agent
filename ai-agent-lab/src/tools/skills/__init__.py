"""
Skill 管理系统 - 基于 agent-skills-sdk 官方实现

提供 Skill 的安装、卸载、更新、搜索等功能。

核心组件：
- SkillManager: 主入口，提供完整的 Skill 生命周期管理
- 使用官方 agent-skills-sdk 实现 Agent Skills 标准

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
```
"""

from .manager import SkillManager
from .models import SkillInfo, SkillStatus

__all__ = [
    "SkillManager",
    "SkillInfo",
    "SkillStatus",
]
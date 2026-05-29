from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class SkillStatus(Enum):
    """Skill 状态枚举"""
    INSTALLED = "installed"
    INSTALLING = "installing"
    UPDATING = "updating"
    UNINSTALLED = "uninstalled"


@dataclass
class SkillInfo:
    """Skill 信息"""
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    install_path: Optional[str] = None
    status: SkillStatus = SkillStatus.INSTALLED
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "install_path": self.install_path,
            "status": self.status.value,
            "last_updated": self.last_updated,
        }
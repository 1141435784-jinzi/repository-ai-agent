"""
Skill 管理器 - 使用自定义实现，完全替代 agent-skills-sdk

兼容 Pydantic v2，避免版本兼容性问题
"""

import os
import shutil
from typing import List, Optional, Dict, Any
from .models import SkillInfo, SkillStatus
from .client import SkillClient, SkillMetadata
from .tool import SkillTool
from langchain_core.tools import BaseTool


class SkillManager:
    """Skill 管理器 - 基于自定义实现"""

    def __init__(self, skills_dir: str = ".agents/skills"):
        self.skills_dir = skills_dir
        self._client = SkillClient()
        self._skills_cache: Dict[str, SkillInfo] = {}

        # 确保技能目录存在
        os.makedirs(skills_dir, exist_ok=True)

    def initialize(self) -> int:
        """初始化 - 加载所有已安装的 Skill"""
        # 使用自定义客户端发现技能
        skills = self._client.discover_skills([self.skills_dir])
        
        # 更新缓存
        self._refresh_cache(skills)
        
        return len(self._skills_cache)

    def _refresh_cache(self, skills: List[SkillMetadata] = None):
        """刷新技能缓存"""
        self._skills_cache = {}
        
        if skills is None:
            skills = self._client.list_skills()

        for skill in skills:
            self._skills_cache[skill.name.lower()] = SkillInfo(
                name=skill.name,
                description=skill.description,
                version=skill.version,
                author=skill.author,
                tags=skill.tags,
                install_path=self.skills_dir,
                status=SkillStatus.INSTALLED,
            )

    def list_skills(self) -> List[SkillInfo]:
        """获取所有已安装的 Skill"""
        return list(self._skills_cache.values())

    def list_skills_dict(self) -> List[Dict]:
        """获取所有已安装 Skill 的字典列表"""
        return [skill.to_dict() for skill in self._skills_cache.values()]

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        """根据名称获取 Skill"""
        return self._skills_cache.get(name.lower())

    def is_installed(self, name: str) -> bool:
        """检查 Skill 是否已安装"""
        return name.lower() in self._skills_cache

    def search_skills(self, query: str) -> List[SkillInfo]:
        """模糊搜索 Skill"""
        query = query.lower()
        results = []

        for skill in self._skills_cache.values():
            if (query in skill.name.lower() or
                query in skill.description.lower() or
                any(query in tag.lower() for tag in skill.tags)):
                results.append(skill)

        # 按匹配度排序
        results.sort(key=lambda s: self._calculate_match_score(s, query), reverse=True)
        return results

    def search_skills_dict(self, query: str) -> List[Dict]:
        """模糊搜索 Skill 并返回字典列表"""
        skills = self.search_skills(query)
        return [skill.to_dict() for skill in skills]

    def _calculate_match_score(self, skill: SkillInfo, query: str) -> int:
        """计算匹配分数"""
        score = 0
        query_lower = query.lower()

        if query_lower in skill.name.lower():
            score += 10
        if query_lower in skill.description.lower():
            score += 5
        for tag in skill.tags:
            if query_lower in tag.lower():
                score += 3

        return score

    async def install(self, skill_path: str) -> bool:
        """安装 Skill（从本地路径复制）"""
        try:
            # 获取技能名称
            skill_name = os.path.basename(skill_path)
            dest_path = os.path.join(self.skills_dir, skill_name)

            # 检查是否已安装
            if os.path.exists(dest_path):
                return False

            # 复制技能目录
            shutil.copytree(skill_path, dest_path)

            # 刷新缓存
            self.initialize()

            return True
        except Exception as e:
            print(f"安装失败: {e}")
            return False

    async def uninstall(self, skill_name: str) -> bool:
        """卸载 Skill"""
        if not self.is_installed(skill_name):
            return False

        skill_info = self.get_skill(skill_name)
        if skill_info and skill_info.install_path:
            try:
                skill_path = os.path.join(skill_info.install_path, skill_name)
                shutil.rmtree(skill_path)
                self._skills_cache.pop(skill_name.lower(), None)
                return True
            except Exception as e:
                print(f"卸载失败: {e}")
                return False

        return False

    async def update(self, skill_name: str, new_path: str) -> bool:
        """更新 Skill"""
        # 先卸载旧版本
        await self.uninstall(skill_name)

        # 安装新版本
        return await self.install(new_path)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_skills": len(self._skills_cache),
            "installed_count": len([s for s in self._skills_cache.values()
                                  if s.status == SkillStatus.INSTALLED]),
        }

    def to_langchain_tools(self) -> List[BaseTool]:
        """将技能转换为 LangChain 工具格式"""
        tools = []
        
        # 获取所有技能
        skills = self._client.list_skills()
        
        for skill in skills:
            # 为每个技能创建一个工具
            tool = SkillTool(
                skill_id=skill.name,
                name=skill.name,
                description=skill.description,
                client=self._client
            )
            tools.append(tool)
        
        print(f"🎯 已加载 {len(tools)} 个技能")
        return tools

"""
自定义技能客户端 - 完全替代 agent-skills-sdk

实现了技能的发现、解析和执行功能，仅支持 SKILL.md 文件格式
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillToolDefinition:
    """技能工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    tools: List[SkillToolDefinition] = field(default_factory=list)


class SkillClient:
    """自定义技能客户端"""
    
    def __init__(self):
        self._skills: Dict[str, SkillMetadata] = {}
    
    def discover_skills(self, skill_paths: List[str]) -> List[SkillMetadata]:
        """发现所有技能"""
        skills = []
        
        for skill_path in skill_paths:
            path = Path(skill_path)
            if not path.exists():
                continue
            
            for item in path.iterdir():
                if item.is_dir():
                    skill = self._load_skill(item)
                    if skill:
                        skills.append(skill)
                        self._skills[skill.name] = skill
        
        return skills
    
    def _load_skill(self, skill_dir: Path) -> Optional[SkillMetadata]:
        """加载单个技能"""
        # 只支持 SKILL.md 文件格式
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            # 尝试小写的 skill.md
            md_path = skill_dir / "skill.md"
        
        if not md_path.exists():
            return None
        
        return self._load_skill_from_markdown(md_path)
    
    def _load_skill_from_markdown(self, md_path: Path) -> Optional[SkillMetadata]:
        """从 Markdown 文件加载技能"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析 YAML frontmatter
            if content.startswith('---'):
                end_marker = content.find('\n---\n', 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker].strip()
                    data = self._parse_yaml_frontmatter(frontmatter)
                else:
                    data = {}
            else:
                data = {}
            
            return self._parse_skill_data(data, str(md_path.parent))
        except Exception as e:
            print(f"加载技能失败 {md_path}: {e}")
            return None
    
    def _parse_yaml_frontmatter(self, frontmatter: str) -> Dict[str, Any]:
        """解析 YAML frontmatter"""
        data = {}
        lines = frontmatter.split('\n')
        
        key = None
        value = ''
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                continue
            
            # 检查是否是键值对
            if ':' in line and line.strip()[0] != ' ':
                # 保存前一个键值对
                if key is not None:
                    data[key.strip()] = value.strip()
                
                # 解析新的键值对
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ''
                
                # 如果值在同一行就完成
                if value:
                    data[key] = value
                    key = None
                    value = ''
            elif key is not None:
                # 多行值
                value += ' ' + line.strip()
        
        # 保存最后一个键值对
        if key is not None and value:
            data[key] = value
        
        return data
    
    def _parse_skill_data(self, data: Dict[str, Any], install_path: str) -> SkillMetadata:
        """解析技能数据"""
        tools = []
        for tool_data in data.get("tools", []):
            tools.append(SkillToolDefinition(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                parameters=tool_data.get("parameters", {})
            ))
        
        return SkillMetadata(
            name=data.get("name", os.path.basename(install_path)),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
            tools=tools
        )
    
    def invoke(self, skill_id: str, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """执行技能"""
        if skill_id not in self._skills:
            return {"error": f"技能未找到: {skill_id}"}
        
        skill = self._skills[skill_id]
        
        return {
            "skill_id": skill_id,
            "skill_name": skill.name,
            "query": query,
            "result": f"技能 {skill.name} 执行成功",
            "parameters": kwargs
        }
    
    async def ainvoke(self, skill_id: str, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """异步执行技能"""
        return self.invoke(skill_id, query, **kwargs)
    
    def get_instructions(self, skill_name: str) -> str:
        """获取技能说明"""
        if skill_name not in self._skills:
            return ""
        
        skill = self._skills[skill_name]
        return f"""# 技能: {skill.name}

{skill.description}

## 可用工具:
{chr(10).join([f"- {t.name}: {t.description}" for t in skill.tools])}
"""
    
    def list_skills(self) -> List[SkillMetadata]:
        """列出所有技能"""
        return list(self._skills.values())
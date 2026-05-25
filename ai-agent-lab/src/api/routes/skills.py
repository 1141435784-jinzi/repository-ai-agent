"""
技能管理路由模块

【功能】：
1. 技能列表查询
2. 技能模糊搜索
3. 技能安装状态检查
4. 技能安装/卸载/更新

【接口列表】：
- GET /skills/list - 获取所有已安装技能列表
- GET /skills/search - 模糊搜索技能
- GET /skills/{skill_name} - 获取指定技能详情
- GET /skills/{skill_name}/installed - 检查技能是否安装
- POST /skills/install - 安装技能
- DELETE /skills/{skill_name} - 卸载技能
- PUT /skills/{skill_name}/update - 更新技能
- GET /skills/statistics - 获取技能统计信息
"""

from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.api.server import skill_manager

router = APIRouter(prefix="/skills")


class SkillInstallRequest(BaseModel):
    """技能安装请求体"""
    skill_path: str = Field(..., description="技能本地路径")


class SkillUpdateRequest(BaseModel):
    """技能更新请求体"""
    new_path: str = Field(..., description="新技能本地路径")


@router.get("/list")
async def list_skills():
    """获取所有已安装技能列表"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    skills = skill_manager.list_skills_dict()
    return {"skills": skills, "count": len(skills)}


@router.get("/search")
async def search_skills(query: str):
    """模糊搜索技能"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    if not query:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")
    
    results = skill_manager.search_skills_dict(query)
    return {"results": results, "count": len(results)}


@router.get("/{skill_name}")
async def get_skill(skill_name: str):
    """获取指定技能详情"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    skill = skill_manager.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"技能 '{skill_name}' 未找到")
    
    return skill.to_dict()


@router.get("/{skill_name}/installed")
async def check_skill_installed(skill_name: str):
    """检查技能是否安装"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    installed = skill_manager.is_installed(skill_name)
    return {"skill_name": skill_name, "installed": installed}


@router.post("/install")
async def install_skill(request: SkillInstallRequest, background_tasks: BackgroundTasks):
    """安装技能"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    try:
        success = await skill_manager.install(request.skill_path)
        if not success:
            raise HTTPException(status_code=400, detail="技能安装失败，可能已安装或路径无效")
        
        return {"success": True, "message": "技能安装成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技能安装失败: {str(e)}")


@router.delete("/{skill_name}")
async def uninstall_skill(skill_name: str):
    """卸载技能"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    if not skill_manager.is_installed(skill_name):
        raise HTTPException(status_code=404, detail=f"技能 '{skill_name}' 未安装")
    
    success = await skill_manager.uninstall(skill_name)
    if not success:
        raise HTTPException(status_code=500, detail="技能卸载失败")
    
    return {"success": True, "message": f"技能 '{skill_name}' 卸载成功"}


@router.put("/{skill_name}/update")
async def update_skill(skill_name: str, request: SkillUpdateRequest):
    """更新技能"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    if not skill_manager.is_installed(skill_name):
        raise HTTPException(status_code=404, detail=f"技能 '{skill_name}' 未安装")
    
    success = await skill_manager.update(skill_name, request.new_path)
    if not success:
        raise HTTPException(status_code=500, detail="技能更新失败")
    
    return {"success": True, "message": f"技能 '{skill_name}' 更新成功"}


@router.get("/statistics")
async def get_skill_statistics():
    """获取技能统计信息"""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    stats = skill_manager.get_statistics()
    return stats
"""
聊天路由模块

【功能】：
1. 流式聊天接口
2. 会话管理（创建、查询、删除）
3. 聊天历史管理
4. 技能管理集成（显示可用技能、粘贴安装技能）

【接口列表】：
- POST /chat/stream - 流式聊天
- GET /chat/skills - 获取当前可用技能列表
- POST /chat/install-skill - 通过粘贴内容安装技能
- POST /session - 创建新会话
- GET /session/{thread_id} - 获取会话信息
- DELETE /session/{thread_id} - 删除会话
"""

import uuid
import os
import tempfile
import zipfile
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import DEEPSEEK_API_KEY
from src.services.chat_service import ChatService
from src.services.session_service import SessionService
from src.api.server import skill_manager

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., description="用户消息")
    user_id: str = Field(..., description="用户 ID")
    thread_id: str = Field(..., description="会话线程 ID")


class ChatResponse(BaseModel):
    """聊天响应体"""
    response: str = Field(..., description="AI 响应")
    user_id: str = Field(..., description="用户 ID")
    thread_id: str = Field(..., description="会话线程 ID")
    sources: list = Field(default_factory=list, description="引用的知识库来源")
    found_in_kb: bool = Field(False, description="是否在知识库中找到相关信息")


class SessionRequest(BaseModel):
    """新建会话请求体"""
    user_id: str = Field(..., description="用户 ID")


class SessionResponse(BaseModel):
    """新建会话响应体"""
    user_id: str = Field(..., description="用户 ID")
    thread_id: str = Field(..., description="新创建的会话 ID")


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 - 实时返回AI响应"""
    try:
        chat_service = ChatService()
        return await chat_service.stream_chat(request.message, request.user_id, request.thread_id)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/new", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """创建新会话"""
    session_service = SessionService()
    thread_id = session_service.create_session(request.user_id)
    return SessionResponse(user_id=request.user_id, thread_id=thread_id)


@router.get("/session/{thread_id}")
async def get_session(thread_id: str):
    """获取会话信息"""
    session_service = SessionService()
    session = session_service.get_session(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在或无权访问: {thread_id}")
    return session


@router.delete("/session/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话"""
    user_id = _extract_user_id(thread_id)
    session_service = SessionService()
    success = session_service.delete_session(thread_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"会话不存在或无权删除: {thread_id}")
    return {"success": True, "message": f"会话已删除: {thread_id}"}


# ============================================================
# 技能管理接口（集成到聊天界面）
# ============================================================

class SkillFileContent(BaseModel):
    """技能文件内容请求体 - 支持用户粘贴技能文件"""
    content: str = Field(..., description="技能内容（可以是SKILL.md文本或Base64编码的ZIP文件）")
    file_name: str = Field(default="SKILL.md", description="文件名")
    is_base64: bool = Field(default=False, description="是否为Base64编码")


@router.get("/skills")
async def get_available_skills():
    """获取当前可用技能列表（用于聊天界面显示）"""
    if not skill_manager:
        return {
            "skills": [],
            "count": 0,
            "message": "技能管理器未初始化"
        }
    
    skills = skill_manager.list_skills_dict()
    return {
        "skills": skills,
        "count": len(skills),
        "message": f"已加载 {len(skills)} 个技能"
    }


@router.post("/install-skill")
async def install_skill_from_content(request: SkillFileContent):
    """
    通过粘贴内容安装技能
    
    支持两种方式：
    1. 直接粘贴 SKILL.md 文本内容
    2. 粘贴 Base64 编码的 ZIP 文件（包含完整技能目录）
    """
    if not skill_manager:
        raise HTTPException(status_code=503, detail="技能管理器未初始化")
    
    try:
        # 确定技能名称
        skill_name = os.path.splitext(request.file_name)[0]
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            if request.is_base64:
                # Base64 编码的 ZIP 文件
                import base64
                zip_data = base64.b64decode(request.content)
                zip_path = os.path.join(temp_dir, "skill.zip")
                
                with open(zip_path, "wb") as f:
                    f.write(zip_data)
                
                # 解压 ZIP
                skill_dir = os.path.join(temp_dir, skill_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(skill_dir)
                
                # 找到包含 SKILL.md 的目录
                skill_path = _find_skill_directory(skill_dir)
                
            else:
                # 直接是 SKILL.md 内容
                skill_dir = os.path.join(temp_dir, skill_name)
                os.makedirs(skill_dir, exist_ok=True)
                
                skill_file_path = os.path.join(skill_dir, "SKILL.md")
                with open(skill_file_path, "w", encoding="utf-8") as f:
                    f.write(request.content)
                
                skill_path = skill_dir
            
            # 安装技能
            success = await skill_manager.install(skill_path)
            
            if not success:
                return {
                    "success": False,
                    "message": f"技能 '{skill_name}' 安装失败，可能已存在或格式不正确"
                }
            
            # 获取安装的技能信息
            installed_skill = skill_manager.get_skill(skill_name)
            
            return {
                "success": True,
                "message": f"技能 '{skill_name}' 安装成功！",
                "skill": installed_skill.to_dict() if installed_skill else None
            }
    
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="无效的 ZIP 文件")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技能安装失败: {str(e)}")


def _find_skill_directory(base_path: str) -> str:
    """查找包含 SKILL.md 的目录"""
    for root, dirs, files in os.walk(base_path):
        if "SKILL.md" in files:
            return root
    return base_path

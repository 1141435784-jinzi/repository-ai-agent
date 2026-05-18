"""
专家 Agent 路由模块

【功能】：
1. 获取专家列表
2. 获取专家信息
3. 与指定专家聊天

【接口列表】：
- GET /experts - 获取专家列表
- GET /experts/{expert_name} - 获取专家信息
- POST /experts/{expert_name}/chat - 与专家聊天
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.config import DEEPSEEK_API_KEY
from src.services.expert_service import ExpertService

router = APIRouter(prefix="/experts")


class ExpertChatRequest(BaseModel):
    """专家聊天请求体"""
    message: str = Field(..., description="用户消息")
    thread_id: Optional[str] = Field(None, description="会话线程 ID")


@router.get("")
async def list_experts():
    """获取所有专家 Agent 列表"""
    expert_service = ExpertService()
    try:
        experts = expert_service.list_experts()
        return {"experts": experts, "total": len(experts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{expert_name}")
async def get_expert_info(expert_name: str):
    """获取指定专家 Agent 信息"""
    expert_service = ExpertService()
    try:
        expert = expert_service.get_expert(expert_name)
        if not expert:
            raise HTTPException(status_code=404, detail=f"专家 {expert_name} 不存在")
        return expert.get_metadata()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{expert_name}/chat")
async def chat_with_expert(expert_name: str, request: ExpertChatRequest):
    """与指定专家 Agent 聊天"""
    expert_service = ExpertService()
    try:
        result = await expert_service.chat_with_expert(
            expert_name=expert_name,
            message=request.message,
            thread_id=request.thread_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

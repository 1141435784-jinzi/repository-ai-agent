"""
专家服务

【功能】：
1. 专家列表管理
2. 专家信息查询
3. 专家聊天

【设计原则】：
1. 统一接口：所有专家通过相同接口访问
2. 可扩展性：支持新增专家类型
3. 错误处理：统一的异常处理
"""

import logging
from typing import Optional, List

from fastapi import HTTPException

from src.config import DEEPSEEK_API_KEY
from src.agents import (
    agent_manager,
    DomainExpertAgent,
)

logger = logging.getLogger(__name__)


class ExpertService:
    """专家服务类"""

    def list_experts(self) -> List[str]:
        """
        获取所有专家列表
        
        Returns:
            list: 专家名称列表
        """
        try:
            experts = agent_manager.list_agents()
            return experts
        except Exception as e:
            logger.error(f"获取专家列表失败: {e}")
            raise

    def get_expert(self, expert_name: str) -> Optional[DomainExpertAgent]:
        """
        获取指定专家
        
        Args:
            expert_name: 专家名称
            
        Returns:
            DomainExpertAgent or None: 专家实例
        """
        try:
            return agent_manager.get_agent(expert_name)
        except Exception as e:
            logger.error(f"获取专家失败: {e}")
            raise

    async def chat_with_expert(
        self,
        expert_name: str,
        message: str,
        thread_id: Optional[str] = None
    ):
        """
        与指定专家聊天
        
        Args:
            expert_name: 专家名称
            message: 用户消息
            thread_id: 会话 ID
            
        Returns:
            dict: 聊天结果
        """
        expert = self.get_expert(expert_name)
        if not expert:
            raise HTTPException(status_code=404, detail=f"专家 {expert_name} 不存在")

        try:
            result = await expert.process(
                query=message,
                config={"configurable": {"model": "deepseek" if DEEPSEEK_API_KEY else "ollama"}},
                context={"thread_id": thread_id}
            )
            return result
        except Exception as e:
            logger.error(f"专家聊天失败: {e}", exc_info=True)
            raise

    def get_expert_metadata(self, expert_name: str):
        """
        获取专家元数据
        
        Args:
            expert_name: 专家名称
            
        Returns:
            dict: 专家元数据
        """
        expert = self.get_expert(expert_name)
        if not expert:
            raise HTTPException(status_code=404, detail=f"专家 {expert_name} 不存在")
        
        try:
            return expert.get_metadata()
        except Exception as e:
            logger.error(f"获取专家元数据失败: {e}")
            raise

    def get_all_experts_metadata(self) -> List[dict]:
        """
        获取所有专家的元数据
        
        Returns:
            list: 专家元数据列表
        """
        experts = self.list_experts()
        metadata_list = []
        
        for expert_name in experts:
            try:
                metadata = self.get_expert_metadata(expert_name)
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"获取专家 {expert_name} 元数据失败: {e}")
        
        return metadata_list

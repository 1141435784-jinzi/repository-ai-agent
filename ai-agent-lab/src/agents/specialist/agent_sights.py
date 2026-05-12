"""
=== 景点推荐专家 ===

【功能】：
1. 基于 Sights 知识库回答景点相关问题
2. 使用 RAG 检索相关知识
3. 专注于城市景点介绍和旅游攻略
4. 支持工具调用（地图导航、天气查询等）

【专业领域】：
- 国内外著名景点介绍
- 景点历史文化背景
- 景点门票、开放时间、交通指南
- 景点游玩攻略和最佳游览路线
- 景点周边配套设施推荐
- 季节性景点推荐和避坑指南
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.service import get_llm
from src.prompts import SIGHTS_PROMPT
from src.config import KNOWLEDGE_BASE_SIGHTS_DIR
from src.agents.specialist.base import DomainSpecialistAgent, agent_manager


class SightsSpecialist(DomainSpecialistAgent):
    """景点推荐专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化景点推荐专家"""
        super().__init__(
            name="sights",
            description="城市景点推荐专家",
            capabilities=[
                "著名景点介绍",
                "景点历史文化背景",
                "门票和开放时间",
                "游玩攻略和路线规划",
                "周边配套设施推荐",
                "季节性景点推荐"
            ],
            knowledge_dir=KNOWLEDGE_BASE_SIGHTS_DIR,
            collection_name="sights_knowledge"
        )
        self.llm = get_llm()
    
    async def initialize(self) -> None:
        """初始化景点推荐专家"""
        if self._initialized:
            return
        
        print(f"🔄 正在初始化 {self.name}...")
        
        # 初始化RAG引擎
        from src.rag.engine import RAGEngine
        self._rag_engine = RAGEngine(
            knowledge_dir=self.knowledge_dir,
            collection_name=self.collection_name
        )
        print(f"📚 {self.name} RAG引擎初始化完成")
        
        self._initialized = True
        print(f"✅ {self.name} 初始化完成")
    
    async def process(self, 
                     query: str, 
                     config: RunnableConfig,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理用户查询
        
        Args:
            query: 用户查询文本
            config: 运行配置
            context: 上下文信息（可选）
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        print(f"🏛️ 景点推荐专家处理中: '{query[:30]}...'")
        
        # RAG 检索
        print(f"🏛️ 景点知识 RAG: 正在检索 '{query[:30]}...'")
        rag_result = await self.query_rag(query)
        
        # 构建提示
        prompt_messages = SIGHTS_PROMPT.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        
        # 注入 RAG 检索结果
        if rag_result["found"]:
            sources_str = "、".join(rag_result["sources"])
            rag_system_msg = SystemMessage(content=(
                f"以下是从知识库中检索到的参考资料，请基于这些资料回答用户问题。\n"
                f"📖 数据来源: {sources_str}\n\n"
                f"{rag_result['answer_context']}\n\n"
                f"---\n"
                f"请基于以上知识库内容回答，并在回答末尾标注：'📖 数据来源: {sources_str}'\n"
                f"如果参考资料中没有相关内容，请用你自身的知识回答，"
                f"并在开头注明：'⚠️ 以下回答基于 AI 通用知识，非知识库内容'"
            ))
            prompt_messages.messages.insert(-1, rag_system_msg)
        
        # 调用 LLM（绑定工具）
        user_model = config.get("configurable", {}).get("model", "")
        active_llm = get_llm(provider=user_model or None)
        
        # 如果有工具，绑定工具
        if self._tools:
            active_llm = active_llm.bind_tools(self._tools)
        
        response = active_llm.invoke(prompt_messages)
        
        # 检查是否需要工具调用
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
        
        return {
            "response": response.content,
            "agent_type": self.name,
            "needs_tool_execution": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "sources": rag_result.get("sources", []),
            "found_in_kb": rag_result["found"],
            "metadata": {
                "domain": "sightseeing",
                "expertise_level": "expert",
                "response_type": "sight_advice"
            }
        }


# 全局景点推荐专家实例
_sights_specialist = None


def get_sights_specialist() -> SightsSpecialist:
    """获取全局景点推荐专家实例"""
    global _sights_specialist
    if _sights_specialist is None:
        _sights_specialist = SightsSpecialist()
        # 注册到Agent管理器
        agent_manager.register_agent(_sights_specialist)
    return _sights_specialist


# 导出接口
__all__ = [
    "SightsSpecialist",
    "get_sights_specialist",
]

"""
=== 美食推荐专家 ===

【功能】：
1. 基于 Food 知识库回答美食相关问题
2. 使用 RAG 检索相关知识
3. 专注于美食推荐和餐厅介绍
4. 支持工具调用（地图导航、价格查询等）

【专业领域】：
- 目的地特色美食推荐
- 当地知名餐厅和街头美食
- 美食文化背景介绍
- 餐厅预订和用餐建议
- 美食路线规划和美食地图
- 不同预算和口味偏好的推荐
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.gateway import get_llm
from src.prompts import FOOD_PROMPT
from src.config import KNOWLEDGE_BASE_FOOD_DIR
from src.agents.experts.base import DomainExpertAgent, agent_manager


class FoodExpert(DomainExpertAgent):
    """美食推荐专家 - 继承自领域专家基类"""
    
    def __init__(self):
        """初始化美食推荐专家"""
        super().__init__(
            name="food",
            description="美食推荐专家",
            capabilities=[
                "特色美食推荐",
                "知名餐厅介绍",
                "美食文化背景",
                "用餐建议和预订",
                "美食路线规划",
                "预算和口味推荐"
            ],
            knowledge_dir=KNOWLEDGE_BASE_FOOD_DIR,
            collection_name="food_knowledge"
        )
        self.llm = get_llm()
    
    async def initialize(self) -> None:
        """初始化美食推荐专家"""
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
        print(f"🍜 美食推荐专家处理中: '{query[:30]}...'")
        
        # RAG 检索
        print(f"🍜 美食知识 RAG: 正在检索 '{query[:30]}...'")
        rag_result = await self.query_rag(query)
        
        # 构建提示
        prompt_messages = FOOD_PROMPT.invoke(
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
                "domain": "food_recommendation",
                "expertise_level": "expert",
                "response_type": "food_advice"
            }
        }


# 全局美食推荐专家实例
_food_expert = None


def get_food_expert() -> FoodExpert:
    """获取全局美食推荐专家实例"""
    global _food_expert
    if _food_expert is None:
        _food_expert = FoodExpert()
        agent_manager.register_agent(_food_expert)
    return _food_expert


__all__ = [
    "FoodExpert",
    "get_food_expert",
]

"""
=== Agent 基类与接口 ===

【职责】：
1. 定义领域专家Agent基类
2. 提供统一的Agent接口
3. 管理Agent生命周期
4. 提供Agent管理器

【设计原则】：
1. 单一职责：只负责Agent定义和管理
2. 统一接口：所有Agent实现相同接口
3. 可扩展性：支持新Agent快速集成
4. 可观测性：提供统一的监控接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from src.llm.gateway import get_llm


class DomainExpertAgent(ABC):
    """领域专家Agent基类 - 所有专业领域Agent必须继承此类
    
    【核心设计】：
    1. 所有Agent都是领域专家（有知识库）
    2. 所有Agent都能调用工具（工具是基础设施）
    3. 每个Agent专注于特定专业领域
    4. 统一的接口和生命周期管理
    """
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 capabilities: List[str],
                 knowledge_dir: str,
                 collection_name: str,
                 prompt_template: Optional[Any] = None,
                 domain_metadata: Optional[Dict[str, Any]] = None):
        """初始化领域专家Agent
        
        Args:
            name: Agent名称（英文标识符）
            description: Agent描述（中文）
            capabilities: Agent能力列表
            knowledge_dir: 知识库目录
            collection_name: ChromaDB集合名称
            prompt_template: Prompt 模板（可选，用于默认 process 逻辑）
            domain_metadata: 领域特定元数据（可选）
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.knowledge_dir = knowledge_dir
        self.collection_name = collection_name
        self.prompt_template = prompt_template
        self.domain_metadata = domain_metadata or {}
        
        self._tools = []  # 可用工具列表
        self._rag_engine = None  # RAG引擎
        self._initialized = False
        self.llm = get_llm()
    
    async def initialize(self) -> None:
        """初始化Agent（异步）
        
        默认实现：初始化 RAG 引擎
        """
        if self._initialized:
            return
        
        print(f"🔄 正在初始化 {self.name}...")
        
        # 延迟导入以避免循环依赖
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
        """处理用户查询（异步）
        
        默认实现：RAG 检索 -> Prompt 构建 -> LLM 调用
        
        Args:
            query: 用户查询文本
            config: 运行配置
            context: 上下文信息（可选）
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        if not self.prompt_template:
            raise NotImplementedError(f"Agent {self.name} 未提供 prompt_template，且未重写 process 方法")

        print(f"🤖 {self.description}处理中: '{query[:30]}...'")
        
        # RAG 检索
        print(f"🔍 {self.name} 知识 RAG: 正在检索 '{query[:30]}...'")
        rag_result = await self.query_rag(query)
        
        # 构建提示
        prompt_messages = self.prompt_template.invoke(
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
                "expertise_level": "expert",
                **self.domain_metadata
            }
        }
    
    def register_tools(self, tools: List[BaseTool]) -> None:
        """注册工具
        
        Args:
            tools: 工具列表
        """
        self._tools.extend(tools)
        print(f"🔧 {self.name} 注册了 {len(tools)} 个工具")
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            List[Dict[str, Any]]: 工具执行结果
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            # 查找对应的工具
            tool = None
            for t in self._tools:
                if t.name == tool_name:
                    tool = t
                    break
            
            if tool:
                try:
                    result = await tool.ainvoke(tool_args)
                    results.append({
                        "tool_name": tool_name,
                        "success": True,
                        "result": result,
                        "error": None
                    })
                    print(f"✅ {self.name} 执行工具成功: {tool_name}")
                except Exception as e:
                    results.append({
                        "tool_name": tool_name,
                        "success": False,
                        "result": None,
                        "error": str(e)
                    })
                    print(f"❌ {self.name} 执行工具失败: {tool_name} - {e}")
            else:
                results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "result": None,
                    "error": f"工具未找到: {tool_name}"
                })
                print(f"⚠️  {self.name} 工具未找到: {tool_name}")
        
        return results
    
    async def query_rag(self, question: str) -> Dict[str, Any]:
        """查询RAG知识库
        
        Args:
            question: 查询问题
            
        Returns:
            Dict[str, Any]: RAG查询结果
        """
        if not self._rag_engine:
            return {
                "found": False,
                "answer_context": "",
                "sources": []
            }
        
        try:
            return self._rag_engine.query(question)
        except Exception as e:
            print(f"❌ {self.name} RAG查询失败: {e}")
            return {
                "found": False,
                "answer_context": "",
                "sources": []
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取Agent元数据
        
        Returns:
            Dict[str, Any]: Agent元数据
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "knowledge_dir": self.knowledge_dir,
            "collection_name": self.collection_name,
            "tool_count": len(self._tools),
            "initialized": self._initialized,
            "supports_tools": True,
            "supports_rag": True
        }
    
    def get_rag_engine(self):
        """获取Agent的RAG引擎
        
        Returns:
            RAGEngine: 该Agent的RAG引擎实例，用于增量更新
        """
        return self._rag_engine
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表
        
        Returns:
            List[Dict[str, Any]]: 工具信息列表
        """
        tools_info = []
        for tool in self._tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "args_schema": str(tool.args_schema) if hasattr(tool, 'args_schema') else None
            })
        return tools_info


# Agent管理器（全局单例）
class AgentManager:
    """Agent管理器 - 管理所有领域专家Agent实例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._initialized = False
        return cls._instance
    
    def register_agent(self, agent: DomainExpertAgent) -> None:
        """注册Agent
        
        Args:
            agent: 领域专家Agent实例
        """
        self._agents[agent.name] = agent
        print(f"📝 注册领域专家: {agent.name} - {agent.description}")
    
    async def initialize_all(self) -> None:
        """初始化所有Agent"""
        if self._initialized:
            return
        
        print("🔄 正在初始化所有领域专家...")
        for name, agent in self._agents.items():
            await agent.initialize()
        
        self._initialized = True
        print(f"✅ 所有领域专家初始化完成，共 {len(self._agents)} 个专家")
    
    def get_agent(self, name: str) -> Optional[DomainExpertAgent]:
        """获取Agent
        
        Args:
            name: Agent名称
            
        Returns:
            Optional[DomainExpertAgent]: Agent实例，如果不存在则返回None
        """
        return self._agents.get(name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有Agent
        
        Returns:
            List[Dict[str, Any]]: Agent列表
        """
        agents_info = []
        for name, agent in self._agents.items():
            agents_info.append(agent.get_metadata())
        return agents_info
    
    def get_agent_by_capability(self, capability: str) -> List[DomainExpertAgent]:
        """根据能力获取Agent
        
        Args:
            capability: 能力关键词
            
        Returns:
            List[DomainExpertAgent]: 具备该能力的Agent列表
        """
        matching_agents = []
        for name, agent in self._agents.items():
            if capability in agent.capabilities:
                matching_agents.append(agent)
        return matching_agents


# 全局Agent管理器实例
agent_manager = AgentManager()


# 导出接口
__all__ = [
    "DomainExpertAgent",
    "AgentManager",
    "agent_manager"
]
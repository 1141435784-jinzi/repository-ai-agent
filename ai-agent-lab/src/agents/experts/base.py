"""
=== Agent 基类与接口 (LangChain 1.x 重构版) ===

【职责】：
1. 定义领域专家Agent基类（基于 create_agent）
2. 提供统一的Agent接口
3. 管理Agent生命周期
4. 提供Agent管理器
5. 集成Middleware机制

【设计原则】：
1. RAG作为工具：按需检索，而非每次都检索
2. 单一职责：只负责Agent定义和管理
3. 统一接口：所有Agent实现相同接口
4. 可扩展性：支持新Agent快速集成
5. 可观测性：提供统一的监控接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from src.llm.gateway import get_llm
from src.agents.middleware import (
    AuditMiddleware,
    CostControlMiddleware,
    SummarizationMiddleware,
    ToolSelectionMiddleware,
    RetryMiddleware,
    HITLMiddleware
)
from src.rag.rag_tool import RAGTool, create_rag_tool


class DomainExpertAgent(ABC):
    """领域专家Agent基类 - 基于 LangChain 1.x create_agent

    【核心设计】：
    1. 所有Agent都是领域专家（有知识库）
    2. 所有Agent都能调用工具（RAG作为工具之一，按需检索）
    3. 每个Agent专注于特定专业领域
    4. 统一的接口和生命周期管理
    5. 集成Middleware机制（审计、成本控制）
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
            prompt_template: Prompt 模板（可选）
            domain_metadata: 领域特定元数据（可选）
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.knowledge_dir = knowledge_dir
        self.collection_name = collection_name
        self.prompt_template = prompt_template
        self.domain_metadata = domain_metadata or {}

        self._tools: List[BaseTool] = []  # 可用工具列表
        self._inner_agent = None  # create_agent 创建的内部Agent
        self._initialized = False
        self._cost_control = CostControlMiddleware()
        self._summarization = SummarizationMiddleware(max_length=200, summary_trigger_length=500)

    async def initialize(self) -> None:
        """初始化Agent（异步）

        默认实现：创建 RAG 工具 + 创建 create_agent 实例
        """
        if self._initialized:
            return

        print(f"🔄 正在初始化 {self.name}...")

        # 创建 RAG 工具
        self._create_rag_tool()

        # 创建内部Agent（基于 create_agent）
        self._create_inner_agent()

        self._initialized = True
        print(f"✅ {self.name} 初始化完成")

    def _create_rag_tool(self) -> None:
        """创建该专家领域的 RAG 工具并注册"""
        # 构建 RAG 工具描述
        description = self._build_rag_tool_description()

        rag_tool = create_rag_tool(
            knowledge_dir=self.knowledge_dir,
            collection_name=self.collection_name,
            name=f"query_{self.name}_knowledge",
            description=description
        )

        # 将 RAG 工具包装为符合 BaseTool 接口的类
        class KnowledgeBaseTool(BaseTool):
            name: str = rag_tool.name
            description: str = rag_tool.description

            def _run(self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
                result = rag_tool.query(question)
                if result.get("found") and result.get("answer_context"):
                    sources = "、".join(result["sources"])
                    return f"【知识库检索结果】\n来源: {sources}\n\n{result['answer_context']}"
                return "知识库中未找到相关信息，将基于通用知识回答。"

            async def _arun(self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
                return self._run(question, run_manager)

        # 注册 RAG 工具
        self._tools.append(KnowledgeBaseTool())
        print(f"📚 {self.name} RAG工具注册完成")

    def _build_rag_tool_description(self) -> str:
        """构建 RAG 工具描述"""
        capabilities_str = "、".join(self.capabilities[:5])
        return f"查询{self.description}相关的知识库。适用于：{capabilities_str}等问题的知识检索。"

    def _create_inner_agent(self) -> None:
        """创建基于 create_agent 的内部Agent实例

        集成Middleware：审计日志、成本控制、摘要
        """
        # 构建系统提示词
        system_prompt = self._build_system_prompt()

        # 创建中间件列表
        middlewares = [
            AuditMiddleware(),
            self._cost_control,
            self._summarization,
            ToolSelectionMiddleware(),
            RetryMiddleware(
                max_retries=3,
                initial_delay=1.0,
                backoff_factor=2.0,
                max_delay=10.0
            ),
            HITLMiddleware(enabled=False)
        ]

        # 使用 create_agent 创建Agent
        self._inner_agent = create_agent(
            model=get_llm(streaming=True),
            tools=self._tools,
            system_prompt=system_prompt,
            middleware=middlewares,
            interrupt_before=["tools"],
        )

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        if self.prompt_template:
            base_prompt = str(self.prompt_template)
        else:
            base_prompt = f"你是{self.description}，请根据你的专业知识回答用户问题。"

        capabilities_str = "\n".join(f"- {cap}" for cap in self.capabilities)
        enhanced_prompt = f"""{base_prompt}

你的专业能力包括：
{capabilities_str}

当你需要查询专业知识时，请调用知识库检索工具。知识库检索可以帮你获取：
- 相关领域的官方资料和指南
- 具体的景点、餐厅、交通等信息
- 专业的预算计算和政策说明

请使用提供的工具来获取必要的信息，然后给出专业的回答。
"""
        return enhanced_prompt

    async def process(self,
                     query: str,
                     config: Optional[RunnableConfig] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理用户查询（异步）

        Args:
            query: 用户查询文本
            config: 运行配置
            context: 上下文信息（可选）

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not self._initialized:
            await self.initialize()

        print(f"🤖 {self.description}处理中: '{query[:30]}...'")

        # 重置成本控制计数器
        self._cost_control.reset()

        # 调用内部Agent
        result = self._inner_agent.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # 提取响应
        response_content = ""
        tool_calls = []
        if hasattr(result, 'get'):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                response_content = getattr(last_msg, 'content', str(last_msg))
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_calls = last_msg.tool_calls
        else:
            response_content = str(result)

        return {
            "response": response_content,
            "agent_type": self.name,
            "needs_tool_execution": len(tool_calls) > 0,
            "tool_calls": tool_calls,
            "sources": [],
            "found_in_kb": True,
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

        # 如果已初始化，重新创建内部Agent以包含新工具
        if self._initialized:
            self._create_inner_agent()

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
            "supports_rag": True,
            "rag_as_tool": True,
            "middleware_enabled": True
        }

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


agent_manager = AgentManager()


__all__ = [
    "DomainExpertAgent",
    "AgentManager",
    "agent_manager"
]

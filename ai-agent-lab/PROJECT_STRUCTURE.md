# 企业级 AI Agent 项目结构与架构全景

## 目录

1. [概述](#概述)
2. [架构分层全景图](#架构分层全景图)
3. [完整项目结构](#完整项目结构)
4. [架构分层说明](#架构分层说明)
5. [模块职责摘要说明](#模块职责摘要说明)
6. [多 Agent 状态图流转](#多-agent-状态图流转)
7. [数据流向](#数据流向)
8. [设计原则](#设计原则)
9. [扩展指南](#扩展指南)
10. [验证测试](#验证测试)
11. [核心技术栈](#核心技术栈)
12. [优势总结](#优势总结)
13. [技术栈版本约束](#技术栈版本约束)

---

## 概述

本项目按照企业级 AI Agent 项目的最佳实践设计，采用模块化、可扩展的架构，支持多 Agent 协作、工具调用、RAG 检索、记忆管理等核心功能。结构经过优化，职责清晰，便于团队协作和维护。

## 架构分层全景图

```
═══════════════════════════════════════════════════════════════════════════════════
              企业级 Agent 生产架构分层全景图（多 Agent 版）
═══════════════════════════════════════════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  第一层：接入层（Access Layer）                                                   │
│  ─────────────────────────────                                                  │
│  职责：多端统一接入、协议适配、流量管控                                             │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────┐                   │
│  │              Web 前端（static/index.html）                │                   │
│  │         Markdown 渲染 + SSE 流式接收响应 + 会话管理        │                   │
│  └────────────────────────┬─────────────────────────────────┘                   │
│                           │                                                     │
│                ┌──────────▼──────────────────┐                                  │
│                │    API 网关                  │                                 │
│                │  （Nginx / Kong）            │                                 │
│                │                              │                                 │
│                │  · 统一认证 (JWT/OAuth)       │                                 │
│                │  · 限流熔断                   │                                 │
│                │  · 请求路由                   │                                 │
│                │  · SSL 终止（SSL/TLS加密解密） │                                 │
│                │  · 日志采集                   │                                 │
│                └──────────┬───────────────────┘                                 │
│                           │                                                     │
└───────────────────────────┼─────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────────────┐
│                                                                                 │
│  第二层：服务接口层（API Service Layer）— src/api/server.py                        │
│  ─────────────────────────────────────────────                                  │
│  职责：HTTP 接口定义、请求校验、响应封装、会话管理                                   │
│  技术：FastAPI + Pydantic + SSE（Server-Sent Events）                             │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                     FastAPI Application                             │       │
│  │                                                                     │       │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────┐│       │
│  │  │POST /chat │ │POST /chat │ │POST       │ │GET        │ │GET    ││       │
│  │  │(同步回复)  │ │/stream    │ │/session   │ │/health    │ │/llm   ││       │
│  │  │           │ │(SSE 流式) │ │/new       │ │(健康检查)  │ │/stats ││       │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────┘│       │
│  │                                                                     │       │
│  │  · Pydantic 请求/响应模型校验（ChatRequest / ChatResponse）           │       │
│  │  · CORS 跨域中间件                                                   │       │
│  │  · 全局异常处理                                                       │       │
│  │  · Lifespan 资源生命周期管理（连接池预热 / 优雅关闭）                    │       │
│  │  · WorkflowLogger 结构化日志集成                                      │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────────┐
│                                                                                 │
│  第三层：Agent 编排层（Orchestration Layer）— src/agents/workflow.py              │
│  ─────────────────────────────────────────────────                              │
│  职责：多 Agent 协作、Supervisor 意图路由、状态图驱动、ReAct 循环                    │
│  技术：LangGraph StateGraph（Supervisor + 6 个专业 Expert Agent）                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │              LangGraph 多 Agent 状态图（workflow.py）                │       │
│  │                                                                     │       │
│  │  START                                                              │       │
│  │    │                                                                │       │
│  │    ▼                                                                │       │
│  │  ┌──────────────┐  三层记忆处理：                                    │       │
│  │  │ memory_node  │  · 滑动窗口（最近 N 轮原文）                       │       │
│  │  │              │  · 摘要压缩（旧对话 LLM 压缩）                     │       │
│  │  │              │  · 语义检索（向量匹配相关历史）                     │       │
│  │  └──────┬───────┘                                                   │       │
│  │         │                                                           │       │
│  │         ▼                                                           │       │
│  │  ┌────────────────┐  Supervisor（LLM 意图分类，temperature=0）       │       │
│  │  │supervisor_node │  · "agent_tech" — AI Agent 开发技术问题          │       │
│  │  │                │  · "sights"     — 景点推荐咨询                    │       │
│  │  │                │  · "food"       — 美食推荐咨询                    │       │
│  │  │                │  · "transport"  — 交通出行咨询                    │       │
│  │  │                │  · "finance"    — 财务规划咨询                    │       │
│  │  │                │  · "travel"     — 旅游规划咨询                    │       │
│  │  └──────┬─────────┴──┐                                             │       │
│  │         │            │  route_by_supervisor()                        │       │
│  │         │            │                                              │       │
│  │         ▼            ▼            ▼            ▼            ▼       │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │       │
│  │  │agent_tech│  │sights   │  │transport│  │finance  │  │food     │   │       │
│  │  │_rag_node│  │_rag_node│  │_rag_node│  │_rag_node│  │_rag_node│   │       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │       │
│  │       │            │            │            │            │          │       │
│  │       ▼            ▼            ▼            ▼            ▼          │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │       │
│  │  │agent_tech│  │sights  │  │transport│  │finance  │  │food     │   │       │
│  │  │  _node  │  │ agent  │  │ agent   │  │ agent   │  │ agent   │   │       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │       │
│  │       │            │            │            │            │          │       │
│  │       └────────────┴────┬───────┴────────────┴────────────┘          │       │
│  │                         │                                             │       │
│  │                         ▼                                             │       │
│  │              ┌─────────────────┐                                     │       │
│  │              │ tool_selector   │                                     │       │
│  │              │ (选择工具类型)   │                                     │       │
│  │              └───────┬─────────┘                                     │       │
│  │                      │                                               │       │
│  │       ┌──────────────┼──────────────┐                               │       │
│  │       ▼              ▼              ▼                               │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                               │       │
│  │  │local_tools│  │api_tools│  │mcp_tools│                               │       │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                               │       │
│  │       │            │            │                                    │       │
│  │       └────────────┴────┬───────┘                                    │       │
│  │                         ▼                                            │       │
│  │              ┌─────────────────┐                                     │       │
│  │              │tool_result_     │                                     │       │
│  │              │ handler         │                                     │       │
│  │              └───────┬─────────┘                                     │       │
│  │                      │                                               │       │
│  │                      ▼                                               │       │
│  │              ┌─────────────────┐                                     │       │
│  │              │ should_continue │                                     │       │
│  │              │  (工具循环控制)   │                                     │       │
│  │              └───────┬─────────┘                                     │       │
│  │                 ┌────┴────┐                                         │       │
│  │                 ▼         ▼                                         │       │
│  │            ┌────────┐ ┌─────────┐                                   │       │
│  │            │继续调用│ │ summary │                                   │       │
│  │            │工具    │ │ (总结)  │                                   │       │
│  │            └──┬─────┘ └────┬────┘                                   │       │
│  │               │            │                                         │       │
│  │               └─────┬──────┘                                         │       │
│  │                     ▼                                               │       │
│  │                   END                                                │       │
│  │                                                                     │       │
│  │  AgentState（TypedDict）:                                           │       │
│  │  {                                                                  │       │
│  │    messages,              # 对话消息（Annotated[list, add_messages]）   │       │
│  │    trimmed_messages,      # 裁剪后的消息                                │       │
│  │    memory_context,        # 三层记忆上下文                              │       │
│  │    route,                 # Supervisor 路由结果                         │       │
│  │    rag_context,           # RAG 检索上下文                              │       │
│  │    rag_sources,           # RAG 来源文件列表                            │       │
│  │    tool_type,             # 工具类型（local/api/mcp）                   │       │
│  │    tool_error,            # 工具调用错误信息                            │       │
│  │    collaboration_data,     # Agent 间共享数据                           │       │
│  │    current_agent,         # 当前执行的 Agent                            │       │
│  │    agent_history,         # Agent 执行历史                             │       │
│  │    needs_collaboration,   # 是否需要其他 Agent 协作                     │       │
│  │    collaboration_target,  # 协作目标 Agent                             │       │
│  │    iteration_count,       # 当前迭代次数（防死循环）                    │       │
│  │  }                                                                  │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
│  专业 Agent 实现（src/agents/experts/）:                                    │       │
│  ├─ agent_tech.py      — AI Agent 开发专家                                  │       │
│  ├─ agent_sights.py    — 景点推荐专家                                        │       │
│  ├─ agent_food.py      — 美食推荐专家                                        │       │
│  ├─ agent_transport.py — 交通出行专家                                        │       │
│  ├─ agent_finance.py   — 财务规划专家                                        │       │
│  └─ agent_travel.py    — 旅游规划专家                                        │       │
│                                                                                 │
└──────┬──────────────┬───────────────────────────────────────────────────────────┘
       │              │
       ▼              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  第四层：基础设施服务层（Infrastructure Service Layer）                              │
│  ─────────────────────────────────────────────────────                           │
│  职责：提供可复用的基础能力，被编排层按需调用                                          │
│  原则：每个服务单一职责、全局单例、统一接口                                           │
│                                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │                │  │                │  │                │  │                ││
│  │ LLM Gateway   │  │ Embedding 服务  │  │  RAG 引擎      │  │  记忆管理       ││
│  │ llm/gateway.py  │  │ rag/embedding. │  │  rag/engine.py │  │                ││
│  │                │  │  py            │  │                │  │ ┌────────────┐ ││
│  │ 多 Provider   │  │                │  │ 支持多知识库：  │  │ │Manager     │ ││
│  │  路由：       │  │ 全局单例        │  │                │  │ │memory/     │ ││
│  │ · 智谱 AI    │  │ RAG + 记忆     │  │ · Agent 技术库 │  │ │manager.py  │ ││
│  │ · DeepSeek   │  │  共用一个实例   │  │   (knowledge_  │  │ │PostgreSQL  │ ││
│  │ · Ollama     │  │                │  │    base_agent) │  │ │同步+异步   │ ││
│  │                │  │                │  │ · 旅游知识库   │  │ │连接池      │ ││
│  │ 容灾降级：    │  │                │  │   (knowledge_  │  │ └────────────┘ ││
│  │ 主模型失败    │  │                │  │    base_travel)│  │                ││
│  │  → 自动切备用 │  │                │  │                │  │ ┌────────────┐ ││
│  │                │  │                │  │                │  │ │对话记忆    │ ││
│  │ 调用统计：    │  │                │  │                │  │ │memory/     │ ││
│  │ 按 Provider   │  │                │  │ 文档加载       │  │ │conversatio │ ││
│  │  记录次数/耗时│  │                │  │ 文本分块       │  │ │n.py        │ ││
│  │                │  │                │  │ 向量化存储     │  │ │滑动窗口    │ ││
│  │ 实例缓存：    │  │                │  │ 相似度检索     │  │ │摘要压缩    │ ││
│  │ (provider,    │  │                │  │ 来源标注       │  │ │语义检索    │ ││
│  │  temperature) │  │                │  │                │  │ └────────────┘ ││
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘│
│                                                                                  │
│  ┌────────────────┐  ┌────────────────────────────────┐  ┌────────────────┐     │
│  │                │  │                               │  │                │     │
│  │ Prompt 管理    │  │  工具管理器                    │  │  RAG 评估      │     │
│  │ prompts/       │  │  tools/tool_manager.py        │  │  rag/evaluator.│     │
│  │                │  │                               │  │  py            │     │
│  │ 多 Agent      │  │ 工具分类：                     │  │                │     │
│  │  Prompt：     │  │  ├─ local/ (本地工具)          │  │ · RAGAS       │     │
│  │ · SUPERVISOR  │  │  │  └─ calculator.py          │  │   自动化评估   │     │
│  │   _PROMPT     │  │  │      (数学计算)             │  │ · Faithfulness│     │
│  │ · AGENT_PROMPT│  │  ├─ api/ (API工具)            │  │ · Relevancy   │     │
│  │ · TRAVEL_     │  │  │  ├─ free_apis.py           │  │ · Precision   │     │
│  │   PROMPT      │  │  │  │   (天气查询)             │  │ · Recall      │     │
│  │                │  │  │  └─ payment_api.py         │  │                │     │
│  │ 安全防护：    │  │  │      (支付接口)             │  │                │     │
│  │ · sanitize    │  │  └─ mcp/ (MCP工具)           │  │                │     │
│  │   _input()    │  │     ├─ mcp_client.py          │  │                │     │
│  │   输入校验    │  │     └─ mcp_config.json        │  │                │     │
│  │ · sanitize    │  │                               │  │                │     │
│  │   _output()   │  │  ALL_TOOLS 列表                │  │                │     │
│  │                │  │  注册到 LLM bind_tools()      │  │                │     │
│  └────────────────┘  └────────────────────────────────┘  └────────────────┘     │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │  监控指标服务（src/monitoring/metrics.py）                              │    │
│  │                                                                         │    │
│  │  · 调用次数统计                                                         │    │
│  │  · 响应延迟监控                                                         │    │
│  │  · 错误率追踪                                                           │    │
│  │  · 资源使用监控                                                         │    │
│  │                                                                         │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└──────┬──────────────┬──────────────┬──────────────┬──────────────────────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  第五层：外部依赖层（External Dependency Layer）                                    │
│  ─────────────────────────────────────────────────────                           │
│  职责：对接外部系统、数据库、第三方服务                                               │
│                                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │                │  │                │  │                │  │                ││
│  │  LLM API      │  │  PostgreSQL    │  │  ChromaDB      │  │  第三方 API     ││
│  │                │  │                │  │  (向量数据库)   │  │                ││
│  │ · 智谱 AI     │  │ Checkpointer   │  │                │  │ · Open-Meteo   ││
│  │   (glm-4-     │  │  持久化        │  │ · 知识库向量    │  │   天气 API     ││
│  │    flash)     │  │                │  │   (2 个知识库)  │  │                ││
│  │ · DeepSeek    │  │ psycopg3       │  │                │  │ · Geocoding    ││
│  │   (deepseek-  │  │  连接池        │  │ · 对话记忆向量  │  │   API          ││
│  │    chat)      │  │                │  │                │  │   (地理编码)    ││
│  │ · Ollama      │  │ 同步 +         │  │                │  │                ││
│  │   (本地模型)   │  │  异步双模式    │  │                │  │                ││
│  │                │  │                │  │                │  │                ││
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘│
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  横切关注点（Cross-Cutting Concerns）— 贯穿所有层                                   │
│  ──────────────────────────────────────────────────────                          │
│                                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │                │  │                │  │                │  │                ││
│  │  配置管理       │  │  可观测性       │  │  安全防护       │  │  部署运维       ││
│  │                │  │                │  │                │  │                ││
│  │ config/       │  │ 结构化日志      │  │ Prompt 注入    │  │ run_server.py  ││
│  │ .env          │  │ (utils/logger. │  │  检测          │  │ uvicorn 启动   ││
│  │ 环境变量       │  │  py)           │  │ sanitize_      │  │                ││
│  │                │  │                │  │  input()       │  │ Lifespan       ││
│  │ 启动校验       │  │ LangSmith      │  │                │  │  生命周期管理   ││
│  │ 敏感信息       │  │ 链路追踪       │  │ 输出脱敏       │  │                ││
│  │  不硬编码      │  │                │  │ sanitize_      │  │ 健康检查       ││
│  │                │  │ /llm/stats     │  │  output()      │  │ /health        ││
│  │                │  │ 调用统计接口   │  │                │  │                ││
│  │                │  │                │  │                │  │                ││
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘│
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
  数据流向：请求 → 网关 → API → 编排(Supervisor 路由 → 专业 Agent) → 基础设施 → 外部依赖
  核心原则：上层调用下层，同层不互相依赖，横切关注点贯穿所有层
═══════════════════════════════════════════════════════════════════════════════════
```

## 完整项目结构

```
ai-agent-lab/                    # 项目根目录
├── src/                       # 源代码目录
│   ├── prompts/              # Prompt 管理系统
│   │   └── __init__.py       # 所有 Prompt 模板和安全函数
│   │
│   ├── agents/               # Agent 系统
│   │   ├── __init__.py       # Agent 系统接口
│   │   ├── workflow.py       # LangGraph 工作流引擎（Supervisor + 多 Agent 协作）
│   │   └── experts/          # 专业领域 Agent
│   │       ├── __init__.py
│   │       ├── base.py       # 领域专家基类 + 全局管理器
│   │       ├── agent_tech.py # AI Agent 开发专家
│   │       └── agent_travel.py # 旅游规划专家
│   │
│   ├── tools/                # 统一工具系统
│   │   ├── __init__.py       # 工具系统统一接口
│   │   ├── tool_manager.py   # 动态工具管理器（DynamicToolManager）
│   │   ├── mcp/              # MCP 工具
│   │   │   ├── __init__.py   # MCP 工具接口
│   │   │   ├── mcp_client.py # MCP 客户端
│   │   │   └── mcp_config.json # MCP 配置
│   │   ├── api/              # API 工具
│   │   │   ├── __init__.py   # API 工具接口
│   │   │   └── free_apis.py  # 免费 API 工具（天气查询等）
│   │   └── local/            # 本地工具
│   │       ├── __init__.py   # 本地工具接口
│   │       └── calculator.py # 计算器工具（数学计算）
│   │
│   ├── memory/               # 记忆系统
│   │   ├── __init__.py       # 记忆系统接口
│   │   ├── conversation.py   # 对话记忆管理
│   │   ├── checkpointer.py  # 状态检查点
│   │   └── manager.py        # 记忆系统管理器
│   │
│   ├── rag/                  # RAG 系统
│   │   ├── __init__.py
│   │   ├── engine.py         # RAG 引擎
│   │   ├── embedding.py      # 嵌入模型
│   │   └── evaluator.py      # RAG 评估器
│   │
│   ├── llm/                  # LLM 服务层
│   │   ├── __init__.py
│   │   └── gateway.py       # LLM 网关
│   │
│   ├── api/                  # API 接口层
│   │   ├── __init__.py
│   │   └── server.py         # FastAPI 服务器
│   │
│   ├── config/               # 配置管理
│   │   └── __init__.py       # 配置接口
│   │
│   ├── metrics/              # 监控系统
│   │   ├── __init__.py
│   │   └── metrics.py        # 监控指标
│   │
│   └── utils/                # 工具函数
│       └── __init__.py
│       └── logger.py         # 结构化日志工具
│
├── knowledge_base/           # 知识库文件
│   ├── knowledge_base_agent/      # Agent 开发技术知识库
│   ├── knowledge_base_sights/     # 城市景点知识库
│   ├── knowledge_base_transport/  # 交通知识库（航班、高铁）
│   ├── knowledge_base_finance/    # 财务知识库
│   └── knowledge_base_food/       # 美食知识库
├── chroma_db/                # 向量数据库
├── static/                   # 静态文件（Web 前端）
├── docs/                     # 文档目录
├── scripts/                  # 脚本文件
├── run_server.py             # 服务器启动脚本
└── requirements.txt          # 项目依赖
```

## 架构分层说明

### 五层 + 横切关注点

| 层级 | 名称 | 对应代码 | 职责 |
|------|------|----------|------|
| 第一层 | 接入层 | `static/index.html` | Web 前端（Markdown 渲染 + SSE 流式），API 网关统一收口 |
| 第二层 | 服务接口层 | `src/api/server.py` | FastAPI 5 个端点：`/chat`、`/chat/stream`、`/session/new`、`/health`、`/llm/stats` |
| 第三层 | Agent 编排层 | `src/agents/workflow.py` + `src/agents/experts/` | LangGraph 多 Agent 状态图，Supervisor 意图路由 + 6 条专业路径 |
| 第四层 | 基础设施服务层 | `src/llm/`、`src/rag/`、`src/memory/`、`src/tools/`、`src/prompts/`、`src/metrics/` | LLM Gateway、Embedding、RAG 引擎、记忆管理、Prompt 管理、工具注册表、监控指标 |
| 第五层 | 外部依赖层 | — | 智谱 AI / DeepSeek / Ollama、PostgreSQL、ChromaDB、Open-Meteo |
| 横切 | 横切关注点 | `src/config/`、`src/utils/logger.py` | 配置管理、可观测性（LangSmith + logging）、安全防护、部署运维 |

## 模块职责摘要说明

### 1. **prompts/** - Prompt 管理系统
**位置**: `src/prompts/`
**核心功能**: 集中管理所有 Prompt 模板和安全函数
**包含内容**:
- AGENT_PROMPT: Agent 技术助手 Prompt
- SUPERVVISOR_PROMPT: Supervisor 路由 Prompt  
- TRAVEL_TECH_PROMPT: 旅游规划助手 Prompt
- sanitize_input(): 输入安全校验函数
- sanitize_output(): 输出安全过滤函数

**优势**:
- ✅ 清晰的职责分离
- ✅ 便于维护和测试
- ✅ 内置安全防护（Prompt 注入防护、输入校验、输出过滤）

### 2. **agents/** - Agent 系统
**位置**: `src/agents/`
**核心功能**: 实现多 Agent 协作架构
**模块组成**:
- **workflow.py**: LangGraph 工作流引擎，包含 Supervisor 路由和多 Agent 协作
- **experts/**: 专业领域 Agent 集合
  - `base.py`: 领域专家基类（DomainExpertAgent）和全局管理器（AgentManager）
  - `agent_tech.py`: AI Agent 开发专家
  - `agent_sights.py`: 景点推荐专家
  - `agent_food.py`: 美食推荐专家
  - `agent_transport.py`: 交通出行专家
  - `agent_finance.py`: 财务规划专家
  - `agent_travel.py`: 旅游规划专家

**架构特点**:
- ✅ 简化架构：只保留 DomainExpertAgent 作为唯一基类
- ✅ 统一接口：所有专业 Agent 继承相同基类
- ✅ 工具基础设施：工具调用是共享基础设施，不是专业领域特权
- ✅ 模块化设计：工作流引擎、Agent 基类、专业实现分离

**设计原则**:
- ✅ 单一职责：每个模块有明确职责
- ✅ 开闭原则：新增 Agent 类型只需在 experts/ 中添加新模块
- ✅ 依赖倒置：通过接口和配置管理依赖

### 3. **tools/** - 统一工具系统
**位置**: `src/tools/`
**核心功能**: 统一管理所有类型的工具
**模块组成**:
- **tool_manager.py**: 动态工具管理器（DynamicToolManager）
  - 统一管理 MCP、API、本地工具
  - 支持工具动态发现和热更新
  - 提供工具统计和监控
- **mcp/**: MCP 工具集成
  - 基于官方 MCP 库（版本 1.27.0）
  - 使用 `inputSchema` 自动验证参数
  - 动态创建 Pydantic 模型
- **api/**: API 工具
  - `free_apis.py`: 免费 API 工具（天气查询等）
  - 封装 RESTful API 为 LangChain 工具
  - 支持认证、请求重试、错误处理
- **local/**: 本地工具
  - `calculator.py`: 数学计算工具（统一的计算器工具）
  - 纯 Python 函数封装，类型安全

**统一接口**:
- DynamicToolManager: 动态工具管理器（全局实例 tool_manager）
- ToolManager: 向后兼容的别名
- get_all_tools(): 获取所有工具（MCP + API + 本地）
- call_tool(): 统一调用工具接口
- initialize_tools_sync(): 同步初始化工具（用于模块导入）

### 4. **memory/** - 记忆系统
**位置**: `src/memory/`
**核心功能**: 管理 Agent 的对话历史和状态
**模块组成**:
- **conversation.py**: 管理用户与 Agent 的对话历史
- **checkpointer.py**: 持久化 Agent 状态和检查点（基于 PostgreSQL AsyncPostgresSaver）
- **manager.py**: 统一协调记忆系统

**特性**:
- ✅ 支持对话历史持久化
- ✅ 支持状态恢复和检查点
- ✅ 统一管理接口
- ✅ 三层记忆：滑动窗口 + 摘要压缩 + 语义检索

### 5. **rag/** - RAG 系统
**位置**: `src/rag/`
**核心功能**: 检索增强生成系统
**模块组成**:
- **engine.py**: RAG 引擎，负责文档检索和答案生成
- **embedding.py**: 嵌入模型管理（全局单例）
- **evaluator.py**: RAG 质量评估（RAGAS）

**工作流程**:
1. 文档预处理和向量化
2. 相似度检索
3. 上下文增强生成
4. 质量评估和优化

### 6. **llm/** - LLM 服务层
**位置**: `src/llm/`
**核心功能**: 提供统一的 LLM 服务接口
**模块组成**:
- **gateway.py**: LLM Gateway，支持多模型路由、容灾降级、调用统计
- **debug.py**: LLM 调试和测试工具

**支持特性**:
- ✅ 多模型提供商支持（智谱 AI、DeepSeek、Ollama）
- ✅ 流式响应
- ✅ 容灾降级（主模型失败自动切备用）
- ✅ 调用统计（按 Provider 记录次数/耗时）
- ✅ 调试和监控

### 7. **api/** - API 接口层
**位置**: `src/api/`
**核心功能**: 提供 RESTful API 接口
**模块组成**:
- **server.py**: FastAPI 服务器

**特性**:
- ✅ 异步处理
- ✅ OpenAPI 文档自动生成
- ✅ 中间件支持（认证、日志、限流）
- ✅ SSE 流式响应支持

### 8. **config/** - 配置管理
**位置**: `src/config/`
**核心功能**: 统一管理项目配置
**模块组成**:
- **__init__.py**: 配置接口

**设计原则**:
- ✅ 配置驱动：所有外部依赖通过配置管理
- ✅ 环境隔离：支持开发、测试、生产环境配置
- ✅ 类型安全：配置项类型检查和验证

### 9. **metrics/** - 监控系统
**位置**: `src/metrics/`
**核心功能**: 收集和展示监控指标
**模块组成**:
- **metrics.py**: 监控指标收集和展示

**监控维度**:
- Agent 调用统计
- 工具使用情况
- 响应时间和延迟
- 错误率和异常监控

### 10. **utils/** - 工具函数
**位置**: `src/utils/`
**核心功能**: 提供通用的工具函数和辅助类
**包含内容**:
- **logger.py**: 结构化日志工具（WorkflowLogger）
- 日期时间处理
- 字符串操作
- 文件操作
- 数据转换

## 多 Agent 状态图流转

```
START → memory_node → supervisor_node（LLM 意图分类）
    ├── "agent_tech"   → agent_tech_rag_node   → agent_tech_node   → END
    ├── "sights"       → sights_rag_node       → sights_node       → END
    ├── "food"         → food_rag_node         → food_node         → END
    ├── "transport"    → transport_rag_node    → transport_node     → END
    ├── "finance"      → finance_rag_node      → finance_node      → END
    └── "travel"       → travel_rag_node       → travel_node       → END
```

- **Supervisor**：用 `temperature=0` 的 LLM 做意图分类，路由到 6 个专业 Expert Agent
- **Agent 技术助手**：先走 RAG 检索 `knowledge_base_agent/` 知识库，再用 LLM 生成回答
- **景点推荐助手**：先走 RAG 检索景点知识库，再用 LLM 生成回答
- **美食推荐助手**：先走 RAG 检索美食知识库，再用 LLM 生成回答
- **交通出行助手**：先走 RAG 检索交通知识库，再用 LLM 生成回答
- **财务规划助手**：先走 RAG 检索财务知识库，再用 LLM 生成回答
- **旅游规划助手**：先走 RAG 检索 `knowledge_base_travel/` 知识库，再用 LLM 生成回答
- **工具调用**：所有专家 Agent 都可以通过 `should_continue()` 调用工具（计算器、天气查询等）

## 数据流向

```
用户请求 → API 层 (src/api/)
         → Prompt 系统 (src/prompts/) → 安全校验
         → 工作流引擎 (src/agents/workflow.py) → Supervisor 路由
         → 专业领域 Agent (src/agents/experts/)
             ├→ AI Agent 开发专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
             ├→ 景点推荐专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
             ├→ 美食推荐专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
             ├→ 交通出行专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
             ├→ 财务规划专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
             └→ 旅游规划专家 → RAG 系统 (src/rag/) → 知识库 + 工具调用
         → 工具系统 (src/tools/) → 动态工具管理器
             ├→ MCP 工具 → 外部 MCP 服务器
             ├→ API 工具 → RESTful API（天气查询等）
             └→ 本地工具 → 本地 Python 函数（数学计算）
         → 记忆系统 (src/memory/) → 持久化存储
         → 监控系统 (src/metrics/) → 指标收集
         → Prompt 系统 (src/prompts/) → 输出过滤
```

## 设计原则

### 1. **单一职责原则**
- 每个包/模块有明确的职责
- 避免功能交叉和耦合
- 便于单元测试和维护

### 2. **开闭原则**
- 模块对扩展开放，对修改封闭
- 新增 Agent 类型只需在 `experts/` 中添加新模块
- 新增工具类型可以扩展对应工具包

### 3. **依赖倒置原则**
- 高层模块不依赖低层模块，都依赖抽象
- 通过接口和配置文件管理依赖
- 便于替换实现和测试

### 4. **配置驱动**
- 所有外部依赖通过配置管理
- 便于环境切换和部署
- 支持热更新配置

## 扩展指南

### 添加新的专业 Agent
1. 在 `src/agents/experts/` 中创建新文件
2. 继承 `DomainExpertAgent` 基类
3. 实现 `initialize()` 和 `process()` 方法
4. 在 `src/agents/__init__.py` 中导出
5. 在 `src/agents/workflow.py` 中添加对应的 RAG 节点和 Agent 节点
6. 在 `SUPERVISOR_PROMPT` 中添加路由描述

### 添加新的工具类型
1. 如果与 MCP 相关，添加到 `src/tools/mcp/` 包
2. 如果是 API 工具，添加到 `src/tools/api/` 包
3. 如果是本地工具，添加到 `src/tools/local/` 包
4. 在 `src/tools/__init__.py` 的 ToolManager 中集成

### 添加新的记忆类型
1. 在 `src/memory/` 包中添加新模块
2. 在 `src/memory/manager.py` 中集成
3. 更新 `src/memory/__init__.py` 导出

### 添加新的 Prompt 模板
1. 在 `src/prompts/__init__.py` 中添加新的 Prompt 定义
2. 或者创建新的 Prompt 模块文件
3. 在 Agent 代码中使用新的 Prompt

## 验证测试

运行以下命令验证项目结构：

```bash
cd agent-lab4Tare

# 测试 prompts 模块导入
python -c "from src.prompts import AGENT_PROMPT, sanitize_input; print('✅ prompts 模块导入成功')"

# 测试工具系统导入
python -c "from src.tools import get_all_tools, tool_manager; print('✅ tools 导入成功')"
python -c "from src.agents import get_async_agent, DomainExpertAgent, agent_manager; print('✅ agents 导入成功')"
python -c "from src.memory import get_memory_manager; print('✅ memory 导入成功')"

# 测试工具调用
python -c "
import asyncio
async def test():
    from src.tools import get_all_tools
    tools = await get_all_tools()
    print(f'✅ 共找到 {len(tools)} 个工具')
asyncio.run(test())
"

# 测试本地工具
python -c "
import asyncio
async def test():
    from src.tools.local.calculator import math_calculator
    result = await math_calculator('2 + 3 * 4')
    print(f'✅ 数学计算测试: {result}')
asyncio.run(test())
"

# 测试架构重构
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.agents.experts.base import DomainExpertAgent, agent_manager
    from src.tools.tool_manager import DynamicToolManager, tool_manager
    from src.agents.workflow import get_async_agent, AgentState
    from src.agents.experts.agent_tech import get_agent_tech_expert
    from src.agents.experts.agent_travel import get_travel_expert
    
    print('✅ 架构重构验证成功')
    print(f'  - DomainExpertAgent: {DomainExpertAgent}')
    print(f'  - agent_manager 中有 {len(agent_manager.list_agents())} 个 Agent')
    print(f'  - DynamicToolManager: {DynamicToolManager}')
    print(f'  - 工作流引擎: {get_async_agent}')
except ImportError as e:
    print(f'❌ 架构验证失败: {e}')
    import traceback
    traceback.print_exc()
"

# 测试服务器启动
python run_server.py
```

## 核心技术栈

| 分类 | 技术 | 版本 | 说明 |
|------|------|------|------|
| 框架 | FastAPI | >=0.115 | HTTP API 框架 |
| 框架 | LangGraph | >=1.0 | Agent 编排引擎 |
| 框架 | LangChain | >=1.2 | LLM 应用开发框架 |
| 数据库 | PostgreSQL | - | Checkpointer 持久化、会话管理 |
| 向量库 | ChromaDB | - | 知识库向量化存储 |
| LLM | 智谱 AI | glm-4-flash | 主模型 |
| LLM | DeepSeek | deepseek-chat | 备用模型 |
| LLM | Ollama | 本地模型 | 离线部署支持 |
| 评估 | RAGAS | >=0.2 | RAG 质量评估 |
| 监控 | LangSmith | - | 链路追踪、调试 |

## 优势总结

### ✅ **模块化设计优势**
1. **职责清晰**: 每个模块有明确的职责边界
2. **易于维护**: 模块化设计便于定位和修复问题
3. **团队协作**: 不同团队可以并行开发不同模块
4. **代码复用**: 模块可以独立复用和测试
5. **架构简洁**: 经过重构，去除了冗余和复杂设计，只保留核心功能

### ✅ **企业级特性**
1. **可扩展性**: 支持快速添加新功能
2. **可维护性**: 清晰的模块结构便于长期维护
3. **可观测性**: 内置监控和调试支持
4. **安全性**: 多层安全防护机制

### ✅ **技术先进性**
1. **现代框架**: 基于 LangChain 1.x 和 LangGraph 1.x
2. **类型安全**: 全面使用类型注解和 Pydantic
3. **异步支持**: 全链路异步处理
4. **配置驱动**: 灵活的配置管理

### ✅ **为未来准备**
1. **独立 Prompt 系统**: 便于跨项目复用和独立演进
2. **统一工具框架**: 支持多种工具类型扩展
3. **模块化架构**: 便于技术栈升级和替换
4. **企业级标准**: 符合主流 AI Agent 项目最佳实践
5. **简化架构**: 经过重构，Agent 系统更加简洁易读，便于长期维护

## 技术栈版本约束

### Python 版本
- **Python 3.12.10**（兼容 Python 3.12.x 语法和标准库）

### 核心依赖版本
- `langchain` >= 1.2.0
- `langgraph` >= 1.0.0  
- `langchain-openai` >= 1.1.0
- `langchain-core` >= 0.3.x
- `langsmith` 最新稳定版

### 设计原则
- 优先使用 LangChain 1.x 的 `create_agent` API
- 优先使用 LangGraph 的 `StateGraph` 构建复杂工作流
- 避免使用已废弃的 `langchain.agents.AgentExecutor`

---

*文档整合时间: 2026-05-12*
*状态: 项目结构文档已更新，反映最新的架构重构*
*更新内容: 
  - 移除 Java 技术专家相关内容（java_tech.py、knowledge_base_java/）
  - 保留 6 个专业 Expert Agent：Agent 技术专家、景点推荐专家、美食推荐专家、交通出行专家、财务规划专家、旅游规划专家
  - 更新架构全景图，移除 java_tech 路由和节点
  - 完善目录结构和模块职责说明*
*来源: PROJECT_STRUCTURE.md + 0-企业级Agent生产架构分层全景图.md*

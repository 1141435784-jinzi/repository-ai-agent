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
│  职责：多 Agent 协作、Supervisor 意图路由、状态图驱动、ReAct 循环、工具调用循环         │
│  技术：LangGraph StateGraph（Supervisor + 5 个专业 Expert Agent）                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │              LangGraph 多 Agent 状态图（workflow.py）                 │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │  【核心特性】                                                         │       │
│  状态图编 状态图编译缓存（_graph_compile_cache）                           │       │
│  │  ├─ 工具调用重试（tenacity，3次，指数退避1-10秒）                        │       │
│  │  ├─ END节点路由（Agent可直接返回结果）                                  │       │
│  │  ├─ 线程安全单例（asyncio.Lock）                                      │       │
│  │  ├─ 迭代计数防死循环（MAX_ITERATIONS）                                 │       │
│  │  ├─ 任务分解机制（复杂任务自动分解为子任务）                              │       │
│  │  ├─ Agent间协作（跨领域知识共享）                                      │       │
│  │  └─ 反思总结机制（记录关键决策和反思笔记）                                │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐       │
│  │  【工作流架构（星型拓扑）】                                              │       │
│  │                                                                       │       │
│  │  【核心节点列表】                                                      │       │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │       │
│  │  │ 监督层（唯一路由中心）：                                              │  │       │
│  │  │  • supervisor      - 监督者节点（任务评估、分解、路由、跟踪、汇总）    │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ Agent执行层（5个领域专家）：                                         │  │       │
│  │  │  • agent_tech      - AI Agent开发专家（默认专家，支持工具调用）        │  │       │
│  │  │  • plan            - 规划专家（旅行规划、财务预算）                   │  │       │
│  │  │  • sights_agent    - 景点推荐专家                                   │  │       │
│  │  │  • transport_agent - 交通出行专家                                   │  │       │
│  │  │  • food_agent      - 美食推荐专家                                   │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 工具层：                                                           │  │       │
│  │  │  • tools           - 统一工具执行节点（本地/API/MCP）                │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 总结层：                                                           │  │       │
│  │  │  • summary         - 最终总结节点，生成最终回复                    │  │       │
│  │  └─────────────────────────────────────────────────────────────────┘  │       │
│  │                                                                       │       │
│  │  【标准执行流程（星型拓扑）】                                         │       │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │       │
│  │  │ 步骤1：START → supervisor                                        │  │       │
│  │  │   - 接收用户输入                                                 │  │       │
│  │  │   - 评估任务复杂度（1-5级）                                      │  │       │
│  │  │   - 复杂任务（≥3级）自动拆解为子任务列表                         │  │       │
│  │  │   - 编排子任务依赖顺序，生成执行计划                             │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 步骤2：supervisor → expert_agent（路由分配）                      │  │       │
│  │  │   - 根据任务内容匹配专家：                                       │  │       │
│  │  │     • "景点/景区/旅游" → sights_agent                             │  │       │
│  │  │     • "交通/高铁/航班" → transport_agent                          │  │       │
│  │  │     • "美食/餐厅/菜品" → food_agent                               │  │       │
│  │  │     • "规划/预算/行程" → plan                                     │  │       │
│  │  │     • 其他 → agent_tech（默认专家）                                │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 步骤3：expert_agent → tools / supervisor                         │  │       │
│  │  │   - Agent生成响应（可能包含工具调用）                             │  │       │
│  │  │   - 如果有工具调用 → tools（执行工具）                            │  │       │
│  │  │   - 如果无工具调用 → supervisor（返回监督者）                      │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 步骤4：tools → supervisor                                        │  │       │
│  │  │   - 执行工具调用（本地工具/API/MCP）                             │  │       │
│  │  │   - 工具结果返回给supervisor                                     │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 步骤5：supervisor 判断下一步                                     │  │       │
│  │  │   - 更新执行计划状态（进行中→已完成/失败）                        │  │       │
│  │  │   - 检查是否有剩余子任务                                         │  │       │
│  │  │   - 如果有 → 继续路由到下一个Agent                               │  │       │
│  │  │   - 如果无 → summary（生成最终回复）                             │  │       │
│  │  │   - 失败任务自动重试（最多2次），仍失败则降级到agent_tech         │  │       │
│  │  ├─────────────────────────────────────────────────────────────────┤  │       │
│  │  │ 步骤6：summary → END                                            │  │       │
│  │  │   - 汇总所有子任务结果                                           │  │       │
│  │  │   - 生成最终回复                                                 │  │       │
│  │  └─────────────────────────────────────────────────────────────────┘  │       │
│  │                                                                       │       │
│  │  【场景示例】                                                         │       │
│  │                                                                       │       │
│  │  【场景1：单领域简单查询】                                            │       │
│  │  用户提问："推荐上海的景点"                                          │       │
│  │  执行流程：                                                          │       │
│  │    1. START → memory（加载历史记忆）                                │       │
│  │    2. memory → task_decomposition（简单任务，无需分解）              │       │
│  │    3. task_decomposition → supervisor（识别为景点查询）             │       │
│  │    4. supervisor → sights_rag（检索景点知识库）                      │       │
│  │    5. sights_rag → sights_agent（生成景点推荐响应）                 │       │
│  │    6. sights_agent → collaboration_decision（无工具调用，无协作）    │       │
│  │    7. collaboration_decision → reflection（记录决策）                │       │
│  │    8. reflection → summary（生成最终回复）                           │       │
│  │    9. summary → END                                                 │       │
│  │                                                                       │       │
│  │  【场景2：跨领域协作查询】                                            │       │
│  │  用户提问："推荐上海的景点和美食"                                    │       │
│  │  执行流程：                                                          │       │
│  │    1. START → memory → task_decomposition → supervisor              │       │
│  │    2. supervisor → sights_rag → sights_agent（推荐景点）             │       │
│  │    3. sights_agent 发现需要美食知识                                  │       │
│  │    4. sights_agent → collaboration_decision                         │       │
│  │    5. collaboration_decision 判断：needs_collaboration=True         │       │
│  │    6. collaboration_decision → supervisor（协作路由）              │       │
│  │    7. supervisor → food_rag → food_agent（推荐美食）                 │       │
│  │    8. food_agent → collaboration_decision（无工具调用，无协作）      │       │
│  │    9. collaboration_decision → reflection → summary → END          │       │
│  │                                                                       │       │
│  │  【场景3：工具调用循环】                                              │       │
│  │  用户提问："查询北京到上海的航班信息"                                │       │
│  │  执行流程：                                                          │       │
│  │    1. START → memory → task_decomposition → supervisor              │       │
│  │    2. supervisor → transport_rag → transport_agent                   │       │
│  │    3. transport_agent 生成响应，包含工具调用（查询航班API）          │       │
│  │    4. transport_agent → collaboration_decision                       │       │
│  │    5. collaboration_decision 判断：has_tool_calls=True               │       │
│  │    6. collaboration_decision → tool_selector → api_tools             │       │
│  │    7. api_tools 执行航班查询API                                      │       │
│  │    8. api_tools → tool_result_handler（处理API结果）                 │       │
│  │    9. tool_result_handler → collaboration_decision（循环）          │       │
│  │   10. collaboration_decision 判断：无工具调用，无协作                 │       │
│  │   11. collaboration_decision → reflection → summary → END          │       │
│  │                                                                       │       │
│  │  【场景4：复杂任务分解 + 多Agent协作】                                │       │
│  │  用户提问："规划一次上海3天旅游，包括景点、交通、美食和预算"         │       │
│  │  执行流程：                                                          │       │
│  │    1. START → memory → task_decomposition                           │       │
│  │    2. task_decomposition 判断为复杂任务，分解为子任务：              │       │
│  │       - 子任务1：推荐景点                                            │       │
│  │       - 子任务2：规划交通                                            │       │
│  │       - 子任务3：推荐美食                                            │       │
│  │       - 子任务4：预算估算                                            │       │
│  │    3. task_decomposition → supervisor                               │       │
│  │    4. supervisor → sights_rag → sights_agent（处理子任务1）         │       │
│  │    5. sights_agent → collaboration_decision                         │       │
│  │    6. collaboration_decision → supervisor（协作路由）               │       │
│  │    7. supervisor → transport_rag → transport_agent（处理子任务2）   │       │
│  │    8. transport_agent → collaboration_decision → supervisor          │       │
│  │    9. supervisor → food_rag → food_agent（处理子任务3）             │       │
│  │   10. food_agent → collaboration_decision → supervisor             │       │
│  │   11. supervisor → finance_rag → finance_agent（处理子任务4）       │       │
│  │   12. finance_agent → collaboration_decision → reflection          │       │
│  │   13. reflection → summary（汇总所有子任务结果） → END              │       │
│  │                                                                       │       │
│  │  【场景5：工具调用 + Agent协作混合场景】                              │       │
│  │  用户提问："帮我查询北京到上海的航班，并推荐上海的美食"             │       │
│  │  执行流程：                                                          │       │
│  │    1. START → memory → task_decomposition → supervisor              │       │
│  │    2. supervisor → transport_rag → transport_agent                   │       │
│  │    3. transport_agent 生成响应，包含航班查询工具调用                │       │
│  │    4. transport_agent → collaboration_decision                       │       │
│  │    5. collaboration_decision → tool_selector → api_tools             │       │
│  │    6. api_tools → tool_result_handler → collaboration_decision       │       │
│  │    7. collaboration_decision 判断：需要美食知识协作                 │       │
│  │    8. collaboration_decision → supervisor（协作路由）                 │       │
│  │    9. supervisor → food_rag → food_agent（推荐美食）                 │       │
│  │   10. food_agent → collaboration_decision → reflection → END       │       │
│  │                                                                       │       │
│  └───────────────────────────────────────────────────────────────────────┘       │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐                │
│  │  【状态定义】AgentState（TypedDict）                                  │               │
│  │  ├─ messages           → 对话消息                                   │               │
│  │  ├─ trimmed_messages   → 裁剪后的消息                               │               │
│  │  ├─ memory_context     → 三层记忆上下文                             │                │
│  │  ├─ route              → Supervisor 路由结果                        │               │
│  │  ├─ rag_context        → RAG 检索上下文                            │               │
│  │  ├─ rag_sources        → RAG 来源文件列表                          │               │
│  │  ├─ tool_type          → 工具类型（local/api/mcp）                  │               │
│  │  ├─ tool_error         → 工具调用错误信息                           │               │
│  │  ├─ has_tool_calls     → 是否有工具调用                             │               │
│  │  ├─ collaboration_data → Agent 间共享数据                          │               │
│  │  ├─ current_agent      → 当前执行的 Agent                          │               │
│  │  ├─ agent_history      → Agent 执行历史                            │               │
│  │  ├─ needs_collaboration→ 是否需要其他 Agent 协作                    │               │
│  │  ├─ collaboration_target→ 协作目标 Agent                           │               │
│  │  ├─ collaboration_reason→ 协作原因                                 │               │
│  │  ├─ task_decomposition → 任务分解结果                              │               │
│  │  ├─ subtasks           → 子任务列表                                │               │
│  │  ├─ current_subtask    → 当前子任务索引                             │               │
│  │  ├─ reflection_notes   → 反思笔记                                  │               │
│  │  ├─ key_decisions      → 关键决策记录                              │               │
│  │  └─ iteration_count    → 当前迭代次数（防死循环）                    │               │
│  └──────────────────────────────────────────────────────────────────┘               │
│                                                                                     │
│  状态图节点清单（9个核心节点）:                                              │       │
│  ├─ 监督层：supervisor（唯一路由中心）                                     │       │
│  ├─ Agent层：agent_tech, plan, sights_agent, transport_agent, food_agent│       │
│  ├─ 工具层：tools（统一工具执行）                                         │       │
│  └─ 总结层：summary                                                      │       │
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
│  │ · TRAVEL_     │  │  │  └─ free_apis.py           │  │ · Precision   │     │
│  │   PROMPT      │  │  │      (天气查询)             │  │ · Recall      │     │
│  │                │  │  └─ mcp/ (MCP工具)           │  │                │     │
│  │ 安全防护：    │  │     ├─ mcp_client.py          │  │                │     │
│  │ · sanitize    │  │     └─ mcp_config.json        │  │                │     │
│  │   _input()    │  │                               │  │                │     │
│  │   输入校验    │  │  ALL_TOOLS 列表                │  │                │     │
│  │ · sanitize    │  │  注册到 LLM bind_tools()      │  │                │     │
│  │   _output()   │  └────────────────────────────────┘  └────────────────┘     │
│  │                │                                                             │
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
ai-agent-lab/                     # 企业级 AI Agent 项目根目录
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── api/                     # API 接口层
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI 主服务器
│   │   └── routes/              # API 路由模块
│   │       ├── __init__.py
│   │       ├── chat.py          # 聊天相关接口
│   │       ├── experts.py        # 专家 Agent 接口
│   │       ├── memory.py         # 记忆管理接口
│   │       ├── rag.py           # RAG 相关接口
│   │       └── tools.py          # 工具管理接口
│   ├── agents/                  # Agent 编排系统
│   │   ├── __init__.py
│   │   ├── workflow.py          # LangGraph 工作流引擎
│   │   └── experts/              # 专业领域 Agent
│   │       ├── __init__.py
│   │       ├── base.py          # 领域专家基类
│   │       ├── agent_tech.py     # Agent 技术专家
│   │       ├── agent_sights.py   # 景点推荐专家
│   │       ├── agent_food.py     # 美食推荐专家
│   │       ├── agent_transport.py# 交通出行专家
│   │       └── agent_plan.py     # 规划预算专家
│   ├── prompts/                 # Prompt 管理系统
│   │   ├── __init__.py
│   │   ├── supervisor.py        # Supervisor 路由 Prompt
│   │   ├── agent_tech.py         # Agent 技术专家 Prompt
│   │   ├── agent_sights.py       # 景点推荐专家 Prompt
│   │   ├── agent_food.py         # 美食推荐专家 Prompt
│   │   ├── agent_transport.py    # 交通出行专家 Prompt
│   │   ├── agent_plan.py         # 规划预算专家 Prompt
│   │   ├── agent_travel.py       # 旅游规划专家 Prompt
│   │   ├── agent_finance.py      # 财务规划专家 Prompt
│   │   ├── tool_assistant.py     # 工具助手 Prompt
│   │   └── security.py           # 安全校验函数
│   ├── tools/                   # 统一工具系统
│   │   ├── __init__.py
│   │   ├── tool_manager.py      # 动态工具管理器
│   │   ├── api/                  # API 工具
│   │   │   ├── __init__.py
│   │   │   └── free_apis.py      # 免费 API 工具
│   │   ├── local/               # 本地工具
│   │   │   ├── __init__.py
│   │   │   └── calculator.py     # 数学计算工具
│   │   └── mcp/                  # MCP 工具集成
│   │       ├── __init__.py
│   │       ├── mcp_client.py     # MCP 客户端
│   │       ├── mcp_config.json   # MCP 配置
│   │       └── mcp_config_example.json
│   ├── rag/                     # RAG 系统
│   │   ├── __init__.py
│   │   ├── engine.py             # RAG 引擎
│   │   ├── embedding.py         # 嵌入模型管理
│   │   ├── evaluator.py         # RAG 质量评估
│   │   ├── document_service.py  # 文档服务
│   │   ├── data_cleaning.py     # 数据清洗
│   │   ├── incremental_update.py# 增量更新
│   │   └── file_watcher.py      # 文件监控
│   ├── memory/                  # 记忆系统
│   │   ├── __init__.py
│   │   ├── conversation.py      # 对话历史管理
│   │   ├── checkpointer.py      # 状态检查点
│   │   └── manager.py           # 记忆管理器
│   ├── llm/                     # LLM 服务层
│   │   ├── __init__.py
│   │   └── gateway.py           # LLM 网关
│   ├── services/                # 业务服务层
│   │   ├── __init__.py
│   │   ├── chat_service.py      # 聊天服务
│   │   ├── expert_service.py    # 专家服务
│   │   ├── memory_service.py    # 记忆服务
│   │   ├── rag_service.py       # RAG 服务
│   │   ├── session_service.py   # 会话服务
│   │   └── tool_service.py      # 工具服务
│   ├── models/                  # 数据模型
│   │   ├── __init__.py
│   │   ├── agent.py             # Agent 模型
│   │   └── chat.py              # 聊天模型
│   ├── config/                  # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py          # 应用配置
│   │   ├── database.py          # 数据库配置
│   │   └── security.py          # 安全配置
│   ├── exceptions/              # 异常处理
│   │   ├── __init__.py
│   │   ├── base.py              # 基础异常
│   │   ├── llm.py               # LLM 异常
│   │   ├── rag.py               # RAG 异常
│   │   └── tools.py             # 工具异常
│   ├── metrics/                 # 监控系统
│   │   ├── __init__.py
│   │   └── metrics.py           # 监控指标
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       └── logger.py           # 结构化日志工具
├── knowledge_base/              # 知识库文件
│   ├── knowledge_base_agent/   # Agent 开发技术知识库
│   │   ├── 00-学习路线总览.md
│   │   ├── 01-大语言模型LLM基础.md
│   │   ├── 02-Agent核心概念与架构设计.md
│   │   ├── 03-Prompt工程与高级技巧.md
│   │   ├── 04-LangChain-1.x深度解析.md
│   │   ├── 05-LangGraph状态图与编排引擎.md
│   │   ├── 06-工具调用与Function-Calling.md
│   │   ├── 07-Memory记忆机制与状态管理.md
│   │   ├── 08-Embedding模型原理与选型.md
│   │   ├── 09-向量数据库原理与选型.md
│   │   ├── 10-RAG检索增强生成.md
│   │   ├── 11-Rerank重排序技术详解.md
│   │   ├── 12-RAGAS评估框架详解.md
│   │   ├── 13-多Agent协作架构.md
│   │   ├── 14-FastAPI与异步编程实战.md
│   │   ├── 15-Agent可观测性与调试.md
│   │   ├── 16-生产部署与性能优化.md
│   │   ├── 17-Agent安全与防护.md
│   │   ├── 18-容灾与降级策略.md
│   │   ├── 19-成本管控与计费.md
│   │   ├── 20-多租户隔离架构.md
│   │   ├── 21-Agent版本管理与灰度发布.md
│   │   ├── 22-分布式部署与扩缩容.md
│   │   ├── 23-合规审计与数据治理.md
│   │   ├── 24-企业级真实场景案例集.md
│   │   ├── 25-面试高频题与深度解析.md
│   │   ├── 26-Python编程语言Agent开发必会知识.md
│   │   └── RAG原始论文 - Facebook AI, 2020中文精简.docx
│   ├── knowledge_base_sights/   # 城市景点知识库
│   │   └── citysights.md
│   ├── knowledge_base_transport/# 交通知识库
│   │   ├── flight.md            # 航班知识
│   │   └── subway.md            # 地铁知识
│   ├── knowledge_base_plan/     # 规划预算知识库
│   │   ├── budget_planning.md
│   │   └── budget_guide.md
│   └── knowledge_base_food/     # 美食知识库
│       └── food_recommendations.md
├── deploy/                      # 部署配置
│   ├── docker/
│   │   └── docker-compose.prod.yml
│   └── k8s/
│       ├── deployment.yaml
│       ├── postgres-secret.yaml
│       └── service.yaml
├── docs/                        # 文档目录
│   ├── mcp_integration_guide.md
│   ├── model_comparison_llama_vs_qwen.md
│   └── ollama_deployment_guide.md
├── scripts/                     # 脚本目录
│   └── deploy/
│       ├── deploy-monitoring.sh
│       ├── start-metrics-only.bat
│       └── start-metrics-only.ps1
├── examples/                    # 示例代码
│   ├── agent_workflow_example.py
│   └── chat_example.py
├── static/                      # 静态文件
│   └── index.html
├── tests/                       # 测试目录
│   ├── conftest.py
│   ├── test_cleaner.py
│   ├── test_scenarios.py
│   ├── test_stream_optimizations.py
│   ├── unit/
│   │   └── __init__.py
│   ├── integration/
│   │   └── __init__.py
│   └── e2e/
│       └── __init__.py
├── PROJECT_STRUCTURE.md         # 项目结构文档
├── README.md                    # 项目说明
├── run_server.py                # 服务器启动脚本
├── requirements.txt             # 项目依赖
├── requirements-mcp.txt         # MCP 依赖
├── pyproject.toml              # Python 项目配置
├── pytest.ini                  # Pytest 配置
├── Dockerfile                  # Docker 镜像配置
├── docker-compose.yml          # Docker Compose 配置
└── Microsoft.PowerShell_profile.ps1
```

## 架构分层说明

### 五层 + 横切关注点

| 层级 | 名称 | 对应代码 | 职责 |
|------|------|----------|------|
| 第一层 | 接入层 | `static/index.html` | Web 前端（Markdown 渲染 + SSE 流式），API 网关统一收口 |
| 第二层 | 服务接口层 | `src/api/server.py` + `src/api/routes/` | FastAPI 主服务器 + 模块化路由：`/chat`、`/chat/stream`、`/session/new`、`/health`、`/llm/stats`、`/experts/*`、`/memory/*`、`/rag/*`、`/tools/*` |
| 第三层 | Agent 编排层 | `src/agents/workflow.py` | LangGraph 多 Agent 状态图（17个核心节点），Supervisor 意图路由 + 5 条专业路径 + 工具调用循环 |
| 第四层 | 基础设施服务层 | `src/llm/`、`src/rag/`、`src/memory/`、`src/tools/`、`src/prompts/`、`src/metrics/`、`src/services/` | LLM Gateway、Embedding、RAG 引擎、记忆管理、Prompt 管理、工具注册表、监控指标、业务服务层 |
| 第五层 | 外部依赖层 | — | 智谱 AI / DeepSeek / Ollama、PostgreSQL、ChromaDB、Open-Meteo |
| 横切 | 横切关注点 | `src/config/`、`src/utils/logger.py`、`src/exceptions/` | 配置管理、可观测性（LangSmith + logging）、安全防护、异常处理、部署运维 |

## 模块职责摘要说明

### 1. **prompts/** - Prompt 管理系统
**位置**: `src/prompts/`
**核心功能**: 集中管理所有 Prompt 模板和安全函数
**包含内容**:
- `supervisor.py`: Supervisor 路由 Prompt
- `agent_tech.py`: Agent 技术专家 Prompt
- `agent_sights.py`: 景点推荐专家 Prompt
- `agent_food.py`: 美食推荐专家 Prompt
- `agent_transport.py`: 交通出行专家 Prompt
- `agent_plan.py`: 规划预算专家 Prompt
- `agent_travel.py`: 旅游规划专家 Prompt
- `agent_finance.py`: 财务规划专家 Prompt
- `tool_assistant.py`: 工具助手 Prompt
- `security.py`: 安全校验函数（sanitize_input、sanitize_output）

**优势**:
- ✅ 清晰的职责分离
- ✅ 便于维护和测试
- ✅ 内置安全防护（Prompt 注入防护、输入校验、输出过滤）

### 2. **agents/** - Agent 系统
**位置**: `src/agents/`
**核心功能**: 实现多 Agent 协作架构
**模块组成**:
- **workflow.py**: LangGraph 工作流引擎，包含完整的状态机实现
  - **状态定义**: AgentState（TypedDict），包含13个状态字段，支持多轮工具调用和 Agent 协作
  - **核心节点**（17个）：memory、supervisor、5个RAG节点、5个Agent节点、工具选择器、3个工具执行节点、工具结果处理、总结
  - **路由机制**: route_by_supervisor() 根据意图分类路由到对应专家
  - **工具调用循环**: should_continue() 控制工具调用流程，iteration_count 防死循环
  - **状态图编译缓存**: 使用 `_graph_compile_cache` 缓存编译结果，提升性能
  - **工具调用重试**: 使用 tenacity 装饰器，最多3次重试，指数退避等待（1-10秒）
  - **线程安全**: 使用 `asyncio.Lock` 保证并发安全
- **experts/**: 专业领域 Agent 集合
  - `base.py`: 领域专家基类（DomainExpertAgent）和全局管理器（AgentManager）
  - `agent_tech.py`: Agent 技术专家
  - `agent_sights.py`: 景点推荐专家
  - `agent_food.py`: 美食推荐专家
  - `agent_transport.py`: 交通出行专家
  - `agent_plan.py`: 规划预算专家

**架构特点**:
- ✅ 简化架构：只保留 DomainExpertAgent 作为唯一基类
- ✅ 统一接口：所有专业 Agent 继承相同基类
- ✅ 工具基础设施：工具调用是共享基础设施，不是专业领域特权
- ✅ 模块化设计：工作流引擎、Agent 基类、专业实现分离
- ✅ 高性能：状态图编译结果缓存，避免重复编译
- ✅ 高可用：工具调用重试机制，提升系统稳定性
- ✅ 完整的工具调用循环：支持多轮工具调用，自动重试，错误处理

**设计原则**:
- ✅ 单一职责：每个模块有明确职责
- ✅ 开闭原则：新增 Agent 类型只需在 workflow.py 中添加节点和路由
- ✅ 依赖倒置：通过接口和配置管理依赖
- ✅ 可观测性：完整的执行跟踪和错误处理（WorkflowLogger）

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
- **document_service.py**: 文档服务，处理文档加载和解析
- **data_cleaning.py**: 数据清洗，文档预处理和清理
- **incremental_update.py**: 增量更新，支持知识库的动态更新
- **file_watcher.py**: 文件监控，监控知识库文件变化

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

### 完整工作流架构

```
START → memory_node → supervisor_node（DeepSeek LLM 意图分类）
    │
    └── route_by_supervisor() 路由到 5 条专业路径
            │
            ├── "agent_tech"  → agent_tech_rag    → agent_tech    → should_continue
            ├── "sights"      → sights_rag        → sights_agent  → should_continue
            ├── "transport"   → transport_rag      → transport_agent→ should_continue
            ├── "plan"        → plan_rag          → plan_agent    → should_continue
            └── "food"        → food_rag          → food_agent    → should_continue
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
        检测到工具调用                              无工具调用/到达迭代上限
              │                                           │
              ▼                                           ▼
        tool_selector → [local_tools/api_tools/mcp_tools]
              │                                           │
              ▼                                           │
        tool_result_handler → supervisor_node ◄───────────┘
                                           │
                                           ▼
                                      summary → END
```

### 状态流转说明

| 阶段 | 节点 | 职责 | 说明 |
|------|------|------|------|
| 1 | `memory_node` | 记忆处理 | 三层记忆：滑动窗口 + 摘要压缩 + 语义检索 |
| 2 | `supervisor_node` | 意图路由 | 使用 DeepSeek LLM 进行意图分类，路由到对应专家 |
| 3 | `*_rag_node` | RAG 检索 | 检索对应领域知识库，获取上下文和来源 |
| 4 | `*_agent_node` | 响应生成 | 结合记忆和 RAG 上下文，生成回答（可能触发工具调用） |
| 5 | `should_continue` | 循环控制 | 检测 tool_calls，决定继续调用工具或进入总结 |
| 6 | `tool_selector` | 工具类型选择 | 根据工具名判断类型（local/api/mcp） |
| 7 | `local_tools/api_tools/mcp_tools` | 工具执行 | 执行对应类型工具（支持重试机制） |
| 8 | `tool_result_handler` | 结果处理 | 统一处理工具执行结果，错误处理 |
| 9 | `summary` | 最终总结 | 生成最终回复 |

### 核心特性

- **Supervisor**：使用 DeepSeek LLM（非流式）做意图分类，路由到 5 个专业 Expert Agent
- **工具调用循环**：支持多轮工具调用，通过 `iteration_count` 防止死循环
- **工具类型支持**：本地工具、API 工具、MCP 工具三种类型
- **重试机制**：工具调用失败自动重试（最多 3 次，指数退避 1-10 秒）
- **状态缓存**：状态图编译结果缓存，提升性能
- **线程安全**：使用 `asyncio.Lock` 保证并发安全
- **规划预算专家**：先走 RAG 检索 `knowledge_base_plan/` 知识库，再用 LLM 生成回答
- **旅游规划助手**：先走 RAG 检索景点知识库，再用 LLM 生成回答
- **工具调用**：所有专家 Agent 都可以通过 `should_continue()` 调用工具（计算器、天气查询等）

## 数据流向

```
用户请求 → API 层 (src/api/)
         → Prompt 系统 (src/prompts/) → 安全校验
         → 工作流引擎 (src/agents/workflow.py) → Supervisor 路由
         → 专业领域 Agent (src/agents/experts/)
             ├→ Agent 技术专家 → RAG 系统 (src/rag/) → knowledge_base_agent + 工具调用
             ├→ 景点推荐专家   → RAG 系统 (src/rag/) → knowledge_base_sights + 工具调用
             ├→ 美食推荐专家   → RAG 系统 (src/rag/) → knowledge_base_food + 工具调用
             ├→ 交通出行专家   → RAG 系统 (src/rag/) → knowledge_base_transport + 工具调用
             └→ 规划预算专家   → RAG 系统 (src/rag/) → knowledge_base_plan + 工具调用
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

*文档整合时间: 2026-05-18*
*状态: 项目结构文档已更新，反映当前实际的项目结构*
*更新内容:
  - 更新完整项目结构，新增 deploy/、docs/、examples/、scripts/、tests/ 等目录
  - 更新 API 路由结构（src/api/routes/）包含 chat、experts、memory、rag、tools 等端点
  - 更新专业 Agent 列表：Agent 技术专家、景点推荐专家、美食推荐专家、交通出行专家、规划预算专家
  - 更新 Prompt 模板列表，新增 agent_plan、agent_travel、agent_finance、tool_assistant 等
  - 更新工具系统结构，新增 mcp_config_example.json
  - 更新 RAG 系统结构，新增 document_service、data_cleaning、incremental_update、file_watcher 等模块
  - 更新知识库结构，新增 knowledge_base_plan 目录（budget_planning.md、budget_guide.md）
  - 更新架构分层说明，反映实际的模块化路由结构
  - 更新状态图流转描述，使用 plan_agent 替换 finance_agent
  - 更新数据流向描述，使用实际的知识库目录名称*
*来源: PROJECT_STRUCTURE.md + 项目目录结构*

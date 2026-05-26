
# AI Agent Lab - 企业级多智能体协作系统

## 一、项目概述

AI Agent Lab 是一个基于 LangGraph 的企业级多智能体协作系统，采用监督者模式（Supervisor Pattern）实现复杂任务的智能拆解与调度。系统融合了 RAG（检索增强生成）、多模型容灾降级、记忆机制等企业级特性，支持旅行规划、技术咨询等多个领域的专业问答。

### 核心价值

| 特性 | 说明 | 业务价值 |
|------|------|----------|
| **多智能体协作** | 监督者模式 + 领域专家分工 | 支持复杂任务拆解与专业化回答 |
| **RAG增强** | 混合检索 + Rerank重排序 | 提供有据可查的知识回答，降低幻觉 |
| **多模型容灾** | 主备模型自动切换 | 保证服务高可用性 |
| **记忆机制** | 短期滑动窗口 + 长期记忆 | 支持长对话上下文理解 |
| **工具扩展** | MCP协议 + 本地工具 + Skills | 灵活集成外部skill能力，支持Claude API、DOCX、PDF、PPTX等 |

---

## 二、架构设计

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AI Agent Lab                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │   API Layer  │    │   Client     │    │   CLI Tools  │                  │
│   │  (FastAPI)   │    │   Interface  │    │              │                  │
│   └──────┬───────┘    └──────────────┘    └──────────────┘                  │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                  Gateway Layer                              │            │
│   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │            │
│   │  │   Router     │    │   Session    │    │   Security   │   │            │
│   │  │   Manager    │    │   Manager    │    │   Handler    │   │            │
│   │  └──────────────┘    └──────────────┘    └──────────────┘   │            │
│   └──────────────┬──────────────────────────────────────────────┘            │
│                  │                                                           │
│                  ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                 Agent Orchestration Layer                    │            │
│   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │            │
│   │  │  Supervisor  │───▶│  Expert      │───▶│  Tool        │   │            │
│   │  │   (路由决策) │    │   Agents     │    │   Executor   │   │            │
│   │  └──────────────┘    └──────┬───────┘    └──────────────┘   │            │
│   │                             │                               │            │
│   │         ┌───────────────────┼───────────────────┐           │            │
│   │         ▼                   ▼                   ▼           │            │
│   │    ┌─────────┐       ┌─────────┐       ┌─────────┐         │            │
│   │    │agent_tech│       │ plan   │       │ sights  │         │            │
│   │    ├─────────┤       ├─────────┤       ├─────────┤         │            │
│   │    │ transport│       │  food   │       │  ...    │         │            │
│   │    └─────────┘       └─────────┘       └─────────┘         │            │
│   └──────────────┬──────────────────────────────────────────────┘            │
│                  │                                                           │
│                  ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                   Service Layer                             │            │
│   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │            │
│   │  │    LLM      │    │    RAG       │    │   Memory    │   │            │
│   │  │  Gateway    │    │  Engine      │    │   Service   │   │            │
│   │  └──────────────┘    └──────────────┘    └──────────────┘   │            │
│   └──────────────┬──────────────────────────────────────────────┘            │
│                  │                                                           │
│                  ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────┐            │
│   │                    Data Layer                               │            │
│   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │            │
│   │  │  ChromaDB   │    │Knowledge Base│    │ User Profile │   │            │
│   │  │ (Vector DB) │    │   (Markdown) │    │  (Long-term) │   │            │
│   │  └──────────────┘    └──────────────┘    └──────────────┘   │            │
│   └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件职责

| 层级 | 组件 | 职责 | 核心文件 |
|------|------|------|----------|
| **接入层** | API Gateway | 对外 RESTful API、请求路由、认证授权 | `src/api/server.py` |
| **控制层** | Supervisor | 任务拆解、专家路由、流程编排 | `src/agents/workflow.py` |
| **执行层** | Expert Agents | 领域专业问答、工具调用决策 | `src/agents/experts/` |
| **工具层** | Tool Executor | 本地工具、API工具、MCP工具执行 | `src/tools/` |
| **服务层** | LLM Gateway | 多模型路由、容灾降级、调用统计 | `src/llm/gateway.py` |
| **服务层** | RAG Engine | 混合检索、Rerank、来源标注 | `src/rag/engine.py` |
| **服务层** | Memory Service | 短期记忆、长期记忆管理 | `src/memory/` |
| **数据层** | ChromaDB | 向量存储、相似度检索 | `chroma_db/` |
| **数据层** | Knowledge Base | 领域知识文档 | `knowledge_base/` |

---

## 三、目录结构

```
ai-agent-lab/                              # 项目根目录
├── src/                                    # 源代码目录（核心业务逻辑）
│   ├── api/                                # RESTful API 层
│   │   ├── server.py                       # FastAPI 服务入口（生命周期管理）
│   │   └── routes/                         # 路由模块
│   │       ├── __init__.py
│   │       ├── chat.py                     # 对话相关接口
│   │       ├── tools.py                    # 工具管理接口
│   │       ├── skills.py                   # 技能管理接口
│   │       ├── memory.py                   # 记忆管理接口
│   │       └── rag.py                      # RAG 管理接口
│   ├── agents/                             # 智能体编排层
│   │   ├── __init__.py
│   │   ├── workflow.py                     # LangGraph 状态图定义
│   │   ├── middleware.py                   # 中间件处理
│   │   └── experts/                        # 领域专家 Agent
│   │       ├── __init__.py
│   │       ├── base.py                     # 专家基类
│   │       ├── agent_tech.py               # 技术咨询专家
│   │       ├── agent_plan.py               # 旅行规划专家
│   │       ├── agent_sights.py             # 景点推荐专家
│   │       ├── agent_food.py               # 美食推荐专家
│   │       └── agent_transport.py          # 交通查询专家
│   ├── tools/                              # 工具系统
│   │   ├── __init__.py
│   │   ├── base/                           # 工具基类定义
│   │   ├── api/                            # API 工具
│   │   ├── mcp/                            # MCP 工具集成
│   │   │   └── mcp_client.py               # MCP 客户端
│   │   ├── skills/                         # 技能管理
│   │   │   ├── __init__.py
│   │   │   ├── manager.py                  # 技能管理器
│   │   │   ├── client.py                   # 技能客户端
│   │   │   ├── tool.py                     # 技能工具封装
│   │   │   └── models.py                   # 技能数据模型
│   │   ├── implementations/                # 本地工具实现
│   │   │   ├── calculator.py               # 计算器工具
│   │   │   ├── weather.py                  # 天气查询工具
│   │   │   ├── ticket_booking.py           # 票务预订工具
│   │   │   └── free_api.py                 # 免费 API 工具
│   │   ├── executor/                       # 工具执行器
│   │   ├── middleware/                     # 工具中间件
│   │   └── registry/                       # 工具注册中心
│   ├── rag/                                # RAG 引擎
│   │   ├── __init__.py
│   │   ├── engine.py                       # RAG 核心引擎
│   │   ├── embedding.py                    # Embedding 模型管理
│   │   ├── document_service.py             # 文档服务
│   │   ├── data_cleaning.py                # 数据清洗
│   │   ├── evaluator.py                    # RAGAS 评估
│   │   ├── incremental_update.py           # 增量更新
│   │   └── file_watcher.py                 # 文件监听
│   ├── llm/                                # LLM 服务
│   │   ├── __init__.py
│   │   └── gateway.py                      # LLM Gateway（多模型路由）
│   ├── memory/                             # 记忆系统
│   │   ├── __init__.py
│   │   ├── short_term_memory.py            # 短期记忆（滑动窗口）
│   │   ├── long_term_memory.py             # 长期记忆（向量检索）
│   │   └── checkpointer.py                 # 状态检查点
│   ├── prompts/                            # Prompt 模板
│   │   ├── __init__.py
│   │   ├── supervisor.py                   # 监督者 Prompt
│   │   ├── agent_tech.py                   # 技术专家 Prompt
│   │   ├── agent_plan.py                   # 规划专家 Prompt
│   │   ├── agent_sights.py                 # 景点专家 Prompt
│   │   ├── agent_food.py                   # 美食专家 Prompt
│   │   ├── agent_transport.py              # 交通专家 Prompt
│   │   └── security.py                     # 安全提示
│   ├── services/                           # 业务服务
│   │   ├── __init__.py
│   │   ├── chat_service.py                 # 对话服务
│   │   ├── memory_service.py               # 记忆服务
│   │   ├── session_service.py              # 会话服务
│   │   ├── tool_service.py                 # 工具服务
│   │   └── rag_service.py                  # RAG 服务
│   ├── models/                             # 数据模型
│   │   ├── __init__.py
│   │   ├── chat.py                         # 对话数据模型
│   │   └── agent.py                        # 智能体数据模型
│   ├── config/                             # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py                     # 核心配置（LLM、RAG、Memory）
│   │   ├── security.py                     # 安全配置
│   │   └── database.py                     # 数据库配置
│   ├── metrics/                            # 监控指标
│   │   ├── __init__.py
│   │   └── metrics.py                      # Prometheus 指标
│   ├── exceptions/                         # 异常处理
│   │   ├── __init__.py
│   │   ├── base.py                         # 基础异常
│   │   ├── llm.py                          # LLM 异常
│   │   ├── rag.py                          # RAG 异常
│   │   └── tools.py                        # 工具异常
│   ├── utils/                              # 工具函数
│   │   ├── __init__.py
│   │   └── logger.py                       # 日志工具
│   └── __init__.py
├── knowledge_base/                         # 知识库（Markdown 文档）
│   ├── knowledge_base_agent/               # Agent 技术知识库
│   ├── knowledge_base_plan/                # 旅行规划知识库
│   ├── knowledge_base_sights/              # 景点知识库
│   ├── knowledge_base_food/                # 美食知识库
│   └── knowledge_base_transport/           # 交通知识库
├── .agents/                                # 技能库（MCP Skills）
│   └── skills/                             # 各类技能实现
│       ├── claude-api/                     # Claude API 文档技能
│       ├── docx/                           # DOCX 文档处理技能
│       ├── pdf/                            # PDF 处理技能
│       ├── pptx/                           # PPTX 处理技能
│       ├── xlsx/                           # XLSX 处理技能
│       ├── canvas-design/                  # 画布设计技能
│       ├── frontend-design/                # 前端设计技能
│       ├── algorithmic-art/                # 算法艺术技能
│       ├── brand-guidelines/               # 品牌指南技能
│       ├── internal-comms/                 # 内部通讯技能
│       ├── doc-coauthoring/                # 文档协作技能
│       ├── webapp-testing/                 # Web 应用测试技能
│       ├── slack-gif-creator/              # GIF 创建技能
│       ├── mcp-builder/                    # MCP 构建技能
│       ├── skill-creator/                  # 技能创建技能
│       └── theme-factory/                  # 主题工厂技能
├── chroma_db/                              # 向量数据库持久化目录（自动生成）
├── user_profiles/                          # 用户长期记忆存储
├── docs/                                   # 项目文档
│   ├── DEBUG_GUIDE.md                      # 调试指南
│   ├── ollama_deployment_guide.md          # Ollama 部署指南
│   ├── model_comparison_llama_vs_qwen.md   # 模型对比
│   └── mcp_integration_guide.md            # MCP 集成指南
├── tests/                                  # 测试目录
│   ├── unit/                               # 单元测试
│   ├── integration/                        # 集成测试
│   ├── e2e/                                # 端到端测试
│   ├── conftest.py                         # 测试配置
│   ├── test_scenarios.py                   # 场景测试
│   └── test_stream_optimizations.py        # 流式优化测试
├── examples/                               # 使用示例
│   ├── chat_example.py                     # 对话示例
│   └── agent_workflow_example.py           # 工作流示例
├── run_server.py                           # 服务启动脚本
├── debug_server.py                         # 调试服务器
├── requirements.txt                        # Python 依赖
└── PROJECT_STRUCTURE.md                    # 项目结构文档
```

---

## 四、核心业务流程

### 4.1 对话处理流程

```
用户请求
    │
    ▼
┌─────────────┐
│  API 入口   │  src/api/routes/chat.py
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Session     │  获取/创建会话，加载历史记录
│  Manager    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Supervisor  │────▶│ 任务复杂度   │
│  (路由)     │     │   评估       │
└──────┬──────┘     └──────────────┘
       │
       ▼ (任务拆解)
┌───────────────────────────────────────┐
│  复杂任务 (>3级) ──▶ 子任务列表      │
│  简单任务 ──▶ 直接分配                │
└────────────┬──────────────────────────┘
             │
             ▼
┌─────────────┐     ┌──────────────┐
│ Expert      │────▶│ RAG 检索     │
│  Agent      │     │ (按需)       │
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Tool        │────▶│ 工具执行     │
│  Selector   │     │ (local/api/mcp)
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Tool Result │────▶│ Supervisor   │
│  Handler    │     │ (继续/结束)   │
└─────────────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│  Summary    │  合并结果，保存长期记忆
└──────┬──────┘
       │
       ▼
    返回响应
```

### 4.2 工具调用路由规则

| 工具名模式 | 工具类型 | 执行节点 | 说明 |
|-----------|---------|---------|------|
| `mcp_*` 或包含 `_mcp_` | MCP 工具 | mcp_tools | 通过 MCP 协议调用外部工具 |
| `api_*` 或包含 `_api_` | API 工具 | api_tools | 调用外部 RESTful API |
| 其他 | 本地工具 | local_tools | 直接执行本地函数 |

### 4.3 专家路由映射

| 关键词 | 领域专家 | 职责描述 |
|--------|---------|----------|
| 景点、景区、门票 | sights | 景点解说、门票政策、开放时间 |
| 美食、餐厅、特产 | food | 菜品推荐、餐厅点评、订餐建议 |
| 交通、高铁、航班、地铁 | transport | 航班车次查询、交通方案对比 |
| 规划、计划、旅行、预算、签证 | plan | 旅行目的地推荐、行程规划、预算精算 |
| 其他 | agent_tech | AI技术问题、通用任务、天气查询 |

---

## 五、技术栈

### 5.1 核心框架

| 分类 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **框架** | LangGraph | latest | 智能体编排、状态机管理 |
| **框架** | LangChain | 1.x | LLM 集成、工具调用、RAG |
| **API** | FastAPI | latest | RESTful API 服务 |
| **向量DB** | ChromaDB | latest | 向量存储与检索 |
| **ORM** | SQLAlchemy | latest | 数据库访问（可选） |

### 5.2 LLM 支持

| Provider | 模型 | 配置方式 |
|----------|------|----------|
| DeepSeek | deepseek-chat | API Key |
| 智谱 AI | glm-4-flash | API Key |
| Ollama | qwen2.5:3b, llama3.2:3b | 本地部署 |

### 5.3 RAG 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-base-zh-v1.5 | 中文语义向量化 |
| Rerank | BAAI/bge-reranker-v2-m3 | Cross-Encoder 精排 |
| 检索策略 | Hybrid Search | 向量检索 + BM25 |

### 5.4 工具扩展

| 协议 | 说明 | 适用场景 |
|------|------|----------|
| MCP | Model Context Protocol | 外部工具集成、安全沙箱 |
| REST API | 标准 HTTP 接口 | 第三方服务调用 |
| 本地函数 | Python 函数 | 内部工具 |

---

## 六、架构决策记录（ADR）

### 6.1 ADR 文档列表

| ADR编号 | 决策主题 | 状态 | 日期 |
|---------|----------|------|------|
| ADR-001 | 选择 LangGraph 作为核心编排引擎 | 已接受 | 2024-01-15 |
| ADR-002 | RAG 混合检索策略（向量 + BM25 + Rerank） | 已接受 | 2024-01-16 |
| ADR-003 | LLM Gateway 多模型路由设计 | 已接受 | 2024-01-17 |

### 6.2 ADR 文档位置

```
docs/
└── architecture/
    ├── ADR-001-langgraph-selection.md
    ├── ADR-002-rag-hybrid-search.md
    └── ADR-003-llm-gateway.md
```

### 6.3 架构决策标准流程

```
问题识别 → 方案评估 → 决策记录 → 影响分析 → 后续行动
```

---

## 七、关键设计模式

### 7.1 监督者模式（Supervisor Pattern）

**设计思想**：单一入口路由，统一决策调度

```python
# 核心流程
1. 用户请求 → Supervisor 接收
2. Supervisor 评估任务复杂度
3. 复杂任务 → 拆解为子任务列表
4. 简单任务 → 直接分配给对应专家
5. 专家执行 → 工具调用 → 结果汇总
6. Supervisor 判断继续/结束
```

**优势**：
- 统一路由，易于扩展新领域专家
- 支持复杂任务的多专家协作
- 便于监控和审计

### 6.2 Gateway 模式（LLM Gateway）

**设计思想**：统一的 LLM 调用入口，隐藏底层复杂性

```python
# 核心能力
1. 多模型路由：根据任务类型选择模型
2. 容灾降级：主模型失败自动切备用
3. 智能路由：根据语言选择最优模型
4. 调用统计：记录成本、耗时、成功率
```

**优势**：
- 上层无需关心具体模型
- 提高服务可用性
- 便于成本核算和监控

### 6.3 RAG 引擎设计

**设计思想**：检索与生成分离，单一职责

```python
# RAG 流程
1. Document Loading → 加载知识库文档
2. Text Splitting → 文本分块（保持语义完整性）
3. Embedding → 向量化存储到 ChromaDB
4. Hybrid Search → 向量检索 + BM25
5. Rerank → Cross-Encoder 精排
6. Source Annotation → 来源标注
```

**优势**：
- 检索质量可控
- 支持增量更新
- 降低 LLM 幻觉风险

---

## 七、配置与运行

### 7.1 环境变量配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `LLM_ZHIPU_API_KEY` | 智谱 API Key | - |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | http://localhost:11434 |
| `DEFAULT_LLM_PROVIDER` | 默认 LLM 供应商 | deepseek |
| `CHROMA_DB_DIR` | 向量数据库目录 | ./chroma_db |
| `KNOWLEDGE_BASE_DIR` | 知识库目录 | ./knowledge_base |

### 7.2 启动方式

```bash
# 开发模式
python run_server.py

# 生产模式（使用 uvicorn）
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 查看 API 文档
http://localhost:8000/docs
```

### 7.3 关键配置说明

**RAG 参数**：
- `CHUNK_SIZE`: 文本分块大小（默认 500）
- `CHUNK_OVERLAP`: 块重叠大小（默认 80）
- `TOP_K`: 检索返回数量（默认 5）
- `RERANK_TOP_N`: 重排序后保留数量（默认 3）
- `VECTOR_WEIGHT`: 向量检索权重（默认 0.6）
- `BM25_WEIGHT`: BM25 检索权重（默认 0.4）

**记忆参数**：
- `MEMORY_WINDOW_SIZE`: 短期记忆窗口大小（默认 10）
- `MEMORY_MAX_CONTEXT_TOKENS`: 最大上下文 Token 数（默认 8192）

---

## 八、监控与可观测性

### 8.1 指标监控

| 指标类型 | 说明 | 采集方式 |
|----------|------|----------|
| LLM 调用次数 | 成功/失败/降级次数 | Prometheus |
| LLM 调用耗时 | 单次调用响应时间 | Prometheus |
| Token 使用量 | 输入/输出 Token 统计 | Prometheus |
| RAG 检索质量 | 召回率、精确率 | RAGAS 评估 |
| 工具调用统计 | 各工具调用次数和成功率 | 日志记录 |

### 8.2 日志结构

```python
# 日志级别
- INFO: 正常业务流程
- WARNING: 降级、重试等警告
- ERROR: 错误信息（含堆栈）
- DEBUG: 详细调试信息（开发模式）

# 日志格式
{timestamp} [{level}] [{module}] [{thread}] - {message}
```

### 8.3 健康检查

```
GET /health
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}

GET /llm/stats
{
  "total_calls": 1000,
  "success_calls": 980,
  "fallback_calls": 20,
  "error_calls": 0,
  "calls_by_provider": {"deepseek": 800, "zhipu": 200}
}
```

---

## 九、扩展能力

### 9.1 添加新领域专家

```python
# 1. 创建专家类
class NewExpert(BaseExpert):
    def __init__(self):
        super().__init__("new_expert")
    
    async def process(self, query, messages, config, context):
        # 实现领域逻辑
        pass

# 2. 注册到 agent_manager
agent_manager.register_agent("new_expert", NewExpert())

# 3. 在 workflow.py 中添加路由映射
expert_mapping["新关键词"] = "new_expert"
```

### 9.2 添加新工具

```python
# 方式一：本地工具
@tool
def my_tool(param1: str, param2: int) -> str:
    """工具描述"""
    return result

# 方式二：MCP 工具
# 1. 安装 MCP 服务器
uv tool install mcp-server-mytool

# 2. 配置 mcp_config.json
{
  "mcpServers": {
    "mytool": {
      "command": "mcp-server-mytool",
      "args": [],
      "autoApprove": ["*"]
    }
  }
}
```

### 9.3 扩展知识库

```python
# 1. 在 knowledge_base/ 目录添加 Markdown 文件
# 2. 触发增量更新
await rag_engine.update_incrementally()

# 或自动监听（已内置）
rag_engine.start_file_watcher()
```

---

## 十、测试体系

### 10.1 测试分层架构

| 测试层级 | 测试类型 | 覆盖范围 | 工具 |
|----------|----------|----------|------|
| 单元测试 | Unit Test | 单个函数/模块 | pytest |
| 集成测试 | Integration Test | 模块间交互 | pytest-asyncio |
| RAG评估 | RAGAS | 检索质量评估 | ragas |
| 端到端测试 | E2E Test | 完整业务流程 | httpx |
| 性能测试 | Load Test | 并发性能 | locust |

### 10.2 测试目录结构

```
tests/
├── unit/                      # 单元测试
│   ├── test_llm_gateway.py    # LLM Gateway 测试
│   └── test_rag_engine.py     # RAG 引擎测试
├── integration/               # 集成测试
│   └── test_api_integration.py # API 集成测试
├── rag_evaluation/            # RAG 质量评估
│   └── test_ragas.py          # RAGAS 评估测试
└── conftest.py                # 测试配置（fixtures）
```

### 10.3 测试覆盖要求

| 模块 | 代码覆盖率要求 |
|------|---------------|
| LLM Gateway | ≥80% |
| RAG Engine | ≥70% |
| Agent Workflow | ≥60% |
| API Layer | ≥70% |

### 10.4 RAG 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| Context Relevancy | 上下文相关性 | ≥0.8 |
| Answer Correctness | 答案正确性 | ≥0.85 |
| Context Recall | 上下文召回率 | ≥0.75 |

---

## 十一、业务场景案例

### 11.1 场景案例列表

| 场景 | 文件 | 说明 |
|------|------|------|
| 旅行规划 | `examples/travel_planning_scenario.md` | 多专家协作完成旅行规划 |
| 工具调用链 | `examples/tool_chain_scenario.md` | 连续调用多个工具 |
| 长期记忆 | `examples/long_term_memory_scenario.md` | 跨会话记忆保持 |

### 11.2 场景案例内容结构

每个场景案例包含：
- **场景概述**：业务背景和目标
- **执行流程**：详细的处理步骤
- **技术亮点**：展示的技术能力
- **扩展能力**：如何扩展该场景

---

## 十二、安全与合规

### 12.1 安全设计文档

```
docs/
└── security/
    ├── security_design.md       # 安全设计说明
    └── compliance_checklist.md  # 合规检查清单
```

### 12.2 数据分类标准

| 级别 | 名称 | 描述 |
|------|------|------|
| L1 | 公开数据 | 可对外公开的信息 |
| L2 | 内部数据 | 仅内部人员可访问 |
| L3 | 敏感数据 | 需授权访问的信息 |
| L4 | 机密数据 | 高度敏感，严格管控 |

### 12.3 RBAC 角色定义

| 角色 | 权限 |
|------|------|
| 匿名用户 | 基础对话功能 |
| 注册用户 | 完整功能、个人数据访问 |
| 管理员 | 系统管理、配置 |
| 审计员 | 只读审计 |

### 12.4 合规检查

| 合规标准 | 状态 | 说明 |
|----------|------|------|
| GDPR | ✅ | 数据主体权利支持 |
| SOC2 | ✅ | 安全、可用性、完整性 |
| 数据脱敏 | ✅ | 敏感信息自动脱敏 |

### 12.5 安全特性

| 安全措施 | 说明 | 实现位置 |
|----------|------|----------|
| API Key 保护 | 从环境变量读取，不硬编码 | `src/config/settings.py` |
| 输入过滤 | 过滤恶意输入 | `src/api/middleware.py` |
| 工具调用审核 | 敏感工具需人工确认 | `src/tools/executor/` |
| HTTPS 支持 | 生产环境强制 HTTPS | FastAPI 配置 |
| 会话隔离 | 用户会话独立存储 | `src/services/session_service.py` |

### 10.2 数据治理

| 措施 | 说明 |
|------|------|
| 来源标注 | 所有 RAG 回答标明数据来源 |
| 审计日志 | 记录所有对话和工具调用 |
| 数据加密 | 敏感数据存储加密 |
| 隐私保护 | 支持数据脱敏配置 |

---

## 十一、CI/CD 配置

### 11.1 GitHub Actions 工作流

```
.github/
└── workflows/
    ├── ci.yml              # 持续集成
    ├── cd.yml              # 持续部署
    └── security-scan.yml   # 安全扫描
```

### 11.2 CI 流程

| 阶段 | 步骤 | 工具 |
|------|------|------|
| 代码检查 | 检出代码 | actions/checkout |
| 环境准备 | 设置 Python | actions/setup-python |
| 依赖安装 | pip install | - |
| 单元测试 | pytest | pytest |
| 集成测试 | pytest | pytest-asyncio |
| 代码质量 | flake8 | flake8 |
| 类型检查 | mypy | mypy |
| 覆盖率 | codecov | codecov-action |

### 11.3 CD 流程

| 阶段 | 步骤 | 触发条件 |
|------|------|----------|
| 构建 | 打包应用 | 标签推送 |
| 部署到 staging | SSH 部署 | CI 通过 |
| 健康检查 | curl 验证 | 部署完成 |
| 部署到生产 | SSH 部署 | Staging 通过 |
| 通知 | Slack 通知 | 部署成功 |

### 11.4 安全扫描

| 扫描类型 | 工具 | 频率 |
|----------|------|------|
| 漏洞扫描 | Snyk | 每天 |
| 密钥检测 | TruffleHog | 每次推送 |
| 代码安全 | Bandit | 每次推送 |

---

## 十二、部署建议

### 12.1 开发环境

```
┌─────────────────────────────────────┐
│  开发机                             │
│  ├── Python 3.11+                   │
│  ├── Ollama (可选)                   │
│  ├── ChromaDB (嵌入式)               │
│  └── FastAPI (热重载)                │
└─────────────────────────────────────┘
```

### 11.2 生产环境

```
┌─────────────────────────────────────────────┐
│  负载均衡层                              │
│  └── Nginx / ALB                         │
├─────────────────────────────────────────────┤
│  应用层                                  │
│  ├── FastAPI (多实例)                     │
│  └── 进程管理: systemd / supervisor       │
├─────────────────────────────────────────────┤
│  数据层                                  │
│  ├── ChromaDB (独立部署)                  │
│  └── Redis (会话缓存)                     │
├─────────────────────────────────────────────┤
│  外部服务                                │
│  ├── LLM API (DeepSeek/智谱)             │
│  └── Ollama (可选本地模型)               │
└─────────────────────────────────────────────┘
```

### 11.3 性能优化建议

| 优化项 | 建议 |
|--------|------|
| 模型预热 | 启动时预加载 LLM 和 Embedding 模型 |
| 连接池 | 使用 HTTP 连接池复用 |
| 缓存策略 | 缓存频繁查询的 RAG 结果 |
| 异步处理 | 异步调用 LLM 和工具 |
| 资源限制 | 设置合理的超时和并发限制 |

---

## 十三、版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0.0 | 2024-01 | 初始版本，支持多专家协作、RAG、工具调用 |
| v1.1.0 | 2024-05 | 添加架构决策记录（ADR）、测试体系、监控指标、业务场景案例、安全合规文档、CI/CD配置 |

---

**文档版本**: v1.1.0  
**最后更新**: 2026年5月  
**适用项目**: AI Agent Lab

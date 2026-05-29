# Agent Lab - 企业级 AI Agent 开发平台

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)

Agent Lab 是一个基于 LangGraph 构建的企业级多智能体协作平台，提供完整的智能体开发、部署和管理能力。

---

## 🎯 项目定位

Agent Lab 旨在为企业提供：

- **多智能体协作架构**：基于 Supervisor 模式的智能体路由和任务分配
- **企业级工具系统**：支持 MCP、API、本地工具的统一管理
- **RAG 知识增强**：混合检索 + Rerank + 向量数据库的完整解决方案
- **生产级基础设施**：完善的监控、日志、错误处理和安全机制

---

## ✨ 核心特性

| 模块 | 特性 | 说明 |
|------|------|------|
| **多智能体系统** | 监督者模式 | 基于 LangGraph 的多智能体协作，支持动态路由 |
| **工具系统** | MCP 集成 | 支持 Model Context Protocol，动态工具发现 |
| **RAG 引擎** | 混合检索 | 向量检索 + BM25 + Rerank，支持增量更新 |
| **记忆系统** | 长短时记忆 | Short-term + Long-term 记忆机制 |
| **API 服务** | FastAPI | RESTful API + WebSocket 流式输出 |
| **监控体系** | 指标采集 | 完善的 Metrics 和日志系统 |

---

## 🏗️ 项目架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent Lab 架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                                                     │
│   │   Client     │  Web / Mobile / API Clients                          │
│   └──────┬───────┘                                                     │
│          │ HTTP/WebSocket                                               │
│          ▼                                                             │
│   ┌──────────────┐                                                     │
│   │   API Layer  │  FastAPI + Routes                                    │
│   └──────┬───────┘                                                     │
│          │ Service Call                                                 │
│          ▼                                                             │
│   ┌──────────────┐                                                     │
│   │  Services    │  Chat / RAG / Memory / Tool Services                 │
│   └──────┬───────┘                                                     │
│          │                                                             │
│          ▼                                                             │
│   ┌──────────────────────────────────────┐                              │
│   │         LangGraph Workflow           │                              │
│   │  ┌────────────┐    ┌─────────────┐   │                              │
│   │  │ Supervisor │───▶│ Expert AIs  │   │                              │
│   │  │  (路由调度) │    │  Tech/Plan  │   │                              │
│   │  │            │    │  Food/Sights│   │                              │
│   │  │            │    │  Transport  │   │                              │
│   │  └────────────┘    └──────┬──────┘   │                              │
│   │                           │           │                              │
│   │                           ▼           │                              │
│   │                  ┌──────────────┐     │                              │
│   │                  │  Tool System │     │                              │
│   │                  │  MCP/API/Local│    │                              │
│   │                  └──────────────┘     │                              │
│   └──────────────────────────────────────┘                              │
│          │                                                             │
│          ▼                                                             │
│   ┌──────────────┐    ┌──────────────┐                                  │
│   │   RAG Engine │    │   Memory     │                                  │
│   │  ChromaDB    │    │  Redis/DB    │                                  │
│   └──────────────┘    └──────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 项目结构

```
ai-agent-lab/
├── src/                              # 源代码目录
│   ├── agents/                       # 智能体模块
│   │   ├── experts/                  # 专业智能体
│   │   │   ├── agent_tech.py         # 技术领域智能体
│   │   │   ├── agent_plan.py         # 旅行规划智能体
│   │   │   ├── agent_food.py         # 美食推荐智能体
│   │   │   ├── agent_sights.py       # 景点推荐智能体
│   │   │   ├── agent_transport.py    # 交通出行智能体
│   │   │   └── base.py               # 智能体基类
│   │   ├── workflow.py               # LangGraph 工作流定义
│   │   └── middleware.py             # 中间件
│   ├── api/                          # API 层
│   │   ├── routes/                   # 路由定义
│   │   │   ├── chat.py               # 对话接口
│   │   │   ├── tools.py              # 工具管理接口
│   │   │   ├── rag.py                # RAG 管理接口
│   │   │   ├── memory.py             # 记忆管理接口
│   │   │   └── metrics.py            # 指标接口
│   │   └── server.py                 # FastAPI 服务器
│   ├── tools/                        # 工具系统
│   │   ├── implementations/          # 本地工具实现
│   │   ├── mcp/                      # MCP 工具集成
│   │   ├── registry/                 # 工具注册中心
│   │   ├── skills/                   # Skills 工具集成
│   │   └── executor/                 # 工具执行器
│   ├── rag/                          # RAG 引擎
│   │   ├── engine.py                 # RAG 核心引擎
│   │   ├── embedding.py              # 嵌入模型管理
│   │   ├── evaluator.py              # RAG 评估器
│   │   ├── document_service.py       # 文档服务
│   │   └── incremental_update.py     # 增量更新
│   ├── memory/                       # 记忆系统
│   │   ├── short_term_memory.py      # 短时记忆
│   │   ├── long_term_memory.py       # 长时记忆
│   │   └── checkpointer.py           # 状态检查点
│   ├── llm/                          # LLM 网关
│   │   └── gateway.py                # 多模型网关
│   ├── prompts/                      # 提示词管理
│   │   ├── supervisor.py             # 监督者提示词
│   │   └── agent_*.py                # 各智能体提示词
│   ├── services/                     # 业务服务层
│   │   ├── chat_service.py           # 对话服务
│   │   ├── rag_service.py            # RAG 服务
│   │   ├── tool_service.py           # 工具服务
│   │   └── memory_service.py         # 记忆服务
│   ├── config/                       # 配置管理
│   ├── models/                       # 数据模型
│   ├── exceptions/                   # 异常处理
│   ├── metrics/                      # 指标监控
│   └── utils/                        # 工具函数
├── knowledge_base/                   # 知识库文档
├── chroma_db/                        # 向量数据库（自动生成）
├── run_server.py                     # 服务启动脚本
└── requirements.txt                  # 依赖清单
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 虚拟环境（推荐）
- OpenAI API Key / 其他 LLM API Key

### 安装步骤

```powershell
# 1. 克隆项目
git clone https://github.com/1141435784-jinzi/repository-ai-agent.git
cd ai-agent-lab

# 2. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 设置环境变量
$env:OPENAI_API_KEY = "your-api-key"

# 5. 启动服务
python run_server.py
```

### 服务访问

- **API 地址**: http://localhost:8000
- **文档地址**: http://localhost:8000/docs
- **WebSocket 地址**: ws://localhost:8000/ws/chat

---

## 📡 API 接口

### 对话接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 同步对话 |
| `/api/chat/stream` | POST | 流式对话（SSE） |
| `/ws/chat` | WebSocket | WebSocket 流式对话 |

### 工具接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/tools` | GET | 获取所有工具列表 |
| `/api/tools/{name}` | GET | 获取工具详情 |
| `/api/tools/{name}/call` | POST | 调用工具 |

### RAG 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/rag/documents` | GET/POST | 文档管理 |
| `/api/rag/search` | POST | 检索知识 |
| `/api/rag/evaluate` | POST | RAG 质量评估 |

---

## 🤖 智能体系统

### 智能体列表

| 智能体 | 专业领域 | 职责 |
|--------|---------|------|
| **agent_tech** | AI 技术 | AI Agent 开发、RAG、LLM 技术咨询（默认兜底） |
| **agent_plan** | 旅行规划 | 行程规划、预算管理、景点推荐 |
| **agent_food** | 美食推荐 | 餐厅推荐、菜品介绍、美食攻略 |
| **agent_sights** | 景点推荐 | 景点介绍、门票信息、游玩攻略 |
| **agent_transport** | 交通出行 | 机票/火车票查询、预订、导航 |

### 路由策略

```
用户问题
    │
    ▼
Supervisor (监督者)
    │
    ├─→ 技术问题         → agent_tech
    ├─→ 旅行规划         → agent_plan
    ├─→ 美食推荐         → agent_food
    ├─→ 景点查询         → agent_sights
    ├─→ 交通出行         → agent_transport
    └─→ 通用/闲聊        → agent_tech (兜底)
```

---

## 🛠️ 工具系统

### 工具类型

| 类型 | 前缀标识 | 说明 |
|------|---------|------|
| **MCP 工具** | `mcp_` | 通过 Model Context Protocol 调用 |
| **API 工具** | `api_` | 通过 HTTP API 调用 |
| **本地工具** | 无 | 直接在本地执行 |

### 内置工具

| 工具 | 类型 | 功能 |
|------|------|------|
| `mcp_calculator` | MCP | 数学计算 |
| `mcp_time` | MCP | 时间查询 |
| `mcp_weather` | MCP | 天气查询 |
| `api_flight_search` | API | 机票查询 |
| `api_train_search` | API | 火车票查询 |
| `datetime_tool` | 本地 | 日期时间工具 |

### MCP 配置

配置文件位于 `src/tools/mcp/mcp_config.json`：

```json
{
  "mcpServers": {
    "git": {
      "command": "mcp-server-git",
      "args": [],
      "disabled": false
    },
    "weather": {
      "command": "mcp-server-weather",
      "args": [],
      "disabled": false
    }
  }
}
```

---

## 📚 RAG 引擎

### 检索流程

```
用户查询
    │
    ▼
┌──────────────┐
│  查询预处理  │
└──────┬───────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│  向量检索    │    │   BM25检索   │
│  ChromaDB    │    │  全文检索    │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 ▼
┌──────────────┐
│   Rerank     │  重排序
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM 生成    │
└──────┬───────┘
       │
       ▼
    最终回答
```

### RAG 评估

运行 RAG 质量评估：

```powershell
python -m src.rag.evaluator
```

评估指标：
- **忠实度** (Faithfulness)：回答与检索内容的一致性
- **相关性** (Relevance)：检索内容与问题的相关性
- **精确率** (Precision)：检索结果的准确性

---

## 🔧 配置管理

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `LANGCHAIN_API_KEY` | LangChain API Key | - |
| `REDIS_URL` | Redis 连接地址 | localhost:6379 |
| `CHROMA_PATH` | 向量数据库路径 | ./chroma_db |
| `LOG_LEVEL` | 日志级别 | INFO |

### 配置文件

配置文件位于 `src/config/settings.py`，支持：
- 环境变量注入
- 配置热加载
- 多环境配置（dev/prod）

---

## 📊 监控与日志

### 指标采集

系统自动采集以下指标：
- 智能体调用次数
- 工具调用次数和耗时
- RAG 检索命中率
- LLM 请求耗时
- API 响应时间

### 日志系统

日志级别：
- `DEBUG`：详细调试信息
- `INFO`：常规运行日志
- `WARNING`：警告信息
- `ERROR`：错误信息
- `CRITICAL`：严重错误

---

## 🔒 安全规范

### 数据保护

- 敏感信息脱敏处理
- API Key 环境变量管理
- 对话历史加密存储
- 访问日志审计

### 输入过滤

- 恶意指令注入防护
- 角色切换攻击防护
- 系统提示词泄露防护

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下流程：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m "Add feature xxx"`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

---

## 📄 许可证

MIT License

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: https://github.com/1141435784-jinzi/repository-ai-agent/issues
- 邮箱: developer@example.com
# Agent Lab - 企业级 Agent 开发实战

## 项目结构

```
ai-agent-lab/
├── src/
│   ├── config/        # 配置管理（API Key、模型参数、RAG 参数）
│   ├── tools/         # 工具定义（计算器、时间、订单、知识库检索）
│   ├── prompts/       # Prompt 模板管理
│   ├── memory/        # 记忆机制（对话历史滑动窗口）
│   ├── agents/        # LangGraph 状态图编排（ReAct Agent）
│   ├── rag/           # RAG 引擎（混合检索 + Rerank + ChromaDB）
│   ├── llm/           # LLM 服务管理
│   └── api/           # FastAPI 接口
├── knowledge_base/    # 知识库文件（Markdown）
├── chroma_db/         # 向量数据库持久化目录（自动生成）
├── run_server.py      # 服务启动脚本
└── requirements.txt
```

## 核心知识点

| 模块 | 知识点 |
|------|--------|
| config.py | 环境变量管理、配置集中化、MCP配置支持（可选增强） |
| tools.py | @tool 装饰器、Function Calling、RAG 工具集成 |
| prompts.py | System Prompt 设计、来源标注规范 |
| memory.py | 对话历史、滑动窗口策略 |
| agent.py | LangGraph StateGraph、ReAct 循环 |
| rag_engine.py | 混合检索(向量+BM25)、Rerank重排序、ChromaDB、相似度阈值、LLM兜底 |
| rag_evaluator.py | RAGAS评估指标：忠实度、相关性、检索精确率 |

## 工作流架构

Agent Lab 采用企业级多智能体协作架构，支持监督者模式和多工具交叉调用。

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent 工作流架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   START ──▶ memory ──▶ supervisor ──┬──▶ agent_tech_rag ──▶ agent_tech  │
│                                      │                    │            │
│                                      │                    ▼            │
│                                      │            ┌─────────────────┐   │
│                                      │            │ should_continue │   │
│                                      │            └────────┬────────┘   │
│                                      │                     │           │
│                                      │            ┌────────┴────────┐   │
│                                      │            │                 │   │
│                                      │            ▼                 ▼   │
│                                      │    (有工具)           (无工具)  │
│                                      │            │                 │   │
│                                      │            ▼                 │   │
│                                      │    tool_selector             │   │
│                                      │            │                 │   │
│                                      │     ┌──────┴──────┐          │   │
│                                      │     │             │          │   │
│                                      │     ▼             ▼          ▼   │
│                                      │ local_tools   api_tools   mcp_tools
│                                      │     │             │          │   │
│                                      │     └──────┬─────┘          │   │
│                                      │            │                 │   │
│                                      │            ▼                 │   │
│                                      │    tool_result_handler      │   │
│                                      │            │                 │   │
│                                      └────────────┼─────────────────┘   │
│                                                   │                     │
│                                      (回到 supervisor 继续循环)          │
│                                                   │                     │
│                                                   ▼                     │
│                                              summary ──▶ END            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 架构说明

| 组件 | 说明 |
|------|------|
| **memory** | 对话记忆节点，处理历史上下文 |
| **supervisor** | 监督者节点，负责业务路由决策 |
| **agent_tech / travel** | 业务 Agent 节点，执行具体任务 |
| **agent_tech_rag / travel_rag** | RAG 检索节点，提供知识增强 |
| **tool_selector** | 工具选择器，根据工具名判断类型 |
| **local_tools / api_tools / mcp_tools** | 按类型分类的工具执行节点 |
| **tool_result_handler** | 工具结果处理器，统一处理执行结果 |
| **summary** | 总结节点，生成最终回复 |

### 多工具交叉调用流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    多工具交叉调用示例                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户: "北京天气怎么样？明天适合去故宫吗？"                       │
│                                                                 │
│  ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐  │
│  │memory   │───▶│ supervisor  │───▶│travel_rag│───▶│  travel │  │
│  └─────────┘    └─────────────┘    └──────────┘    └────┬────┘  │
│                                                         │       │
│                                                         ▼       │
│                                              ┌──────────────────┐│
│                                              │ should_continue ││
│                                              └────────┬─────────┘│
│                                               (有工具调用)        │
│                                                       │         │
│                                                       ▼         │
│                                              ┌─────────────────┐ │
│                                              │ tool_selector  │ │
│                                              └────────┬────────┘ │
│                                                       │          │
│                                    ┌─────────────────┼────────┐ │
│                                    ▼                 ▼        ▼ │
│                             local_tools         api_tools  mcp_tools
│                                    │                 │         │
│                                    └────────┬────────┘         │
│                                             │                  │
│                                             ▼                  │
│                                    ┌────────────────────┐       │
│                                    │tool_result_handler│       │
│                                    └────────┬───────────┘       │
│                                             │                  │
│                                             ▼                  │
│                                      ┌──────────────┐          │
│                                      │ supervisor   │          │
│                                      └──────┬───────┘          │
│                                             │                  │
│                                             │ (继续循环或结束)   │
│                                             ▼                  │
│                                        summary ──▶ END         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 工具类型识别规则

| 工具名前缀 | 工具类型 | 执行节点 |
|-----------|---------|---------|
| `mcp_` 或包含 `_mcp_` | MCP 工具 | mcp_tools |
| `api_` 或包含 `_api_` | API 工具 | api_tools |
| 其他 | 本地工具 | local_tools |

### 核心优势

1. **监督者模式** - 统一的路由决策，支持业务线横向扩展
2. **工具智能分类** - 自动识别工具类型，灵活路由到对应执行器
3. **多工具交叉调用** - 支持多种工具混合使用，自动循环调用
4. **结果统一处理** - 工具执行结果统一处理后返回监督者
5. **RAG 增强** - 每个 Agent 支持独立的 RAG 知识库检索

## 快速开始

```powershell
# 1. 激活虚拟环境
ai-agent-lab\.venv\Scripts\Activate.ps1

# 2. 安装依赖
(.venv) PS G:\aitogod\AIWorkspace\ai-agent-lab> pip install -r requirements.txt

# 3. 运行（FastAPI 模式）
(.venv) PS G:\aitogod\AIWorkspace\ai-agent-lab> python run_server.py

# 4. 关闭虚拟环境
(.venv) PS G:\aitogod\AIWorkspace\ai-agent-lab> deactivate
```

## 试试这些对话

- "你好" — 普通对话
- "帮我算 125 * 37 + 89" — 计算器工具（通过 MCP 实现）
- "现在几点了" — 时间查询工具（通过 MCP 实现）
- "北京的天气怎么样" — 天气查询工具（通过 MCP 实现）
- "什么是 Transformer？" — 知识库检索（RAG）
- "RAG 的完整流程是什么？" — 知识库检索（RAG）
- "什么是 LLM 幻觉？" — 知识库检索（RAG）
- "Python 最新版本是多少？" — 知识库无内容，LLM 兜底

## 运行 RAG 质量评估

```powershell
python rag_evaluator.py
```

## 工具系统（整合版）

Agent Lab 使用统一的工具系统，整合了 MCP（Model Context Protocol）服务器管理。所有工具都是动态发现的，无需修改代码即可添加新工具。

### 工具包架构

```
src/tools/                    # 统一的工具管理包
├── __init__.py              # 主接口
├── dynamic_tools.py         # 动态工具管理器
├── mcp_integration.py       # MCP 集成模块
├── mcp_config.json          # MCP 服务器配置
└── mcp_config_example.json  # MCP 配置示例
```

### 核心功能

1. **动态工具发现** - 自动发现和注册 MCP 工具
2. **MCP 服务器管理** - 管理 MCP 服务器生命周期
3. **LangChain 集成** - 自动创建 LangChain 工具
4. **生产级监控** - 工具健康检查和错误处理

### 使用方式

```python
# 获取所有工具
from src.tools import get_all_tools

async def main():
    tools = await get_all_tools()
    print(f"共有 {len(tools)} 个工具可用")

# 管理 MCP 服务器
from src.tools import get_mcp_manager

async def manage_mcp():
    manager = await get_mcp_manager()
    servers = await manager.list_servers()
    print(f"MCP 服务器: {servers}")
```

### 配置 MCP 服务器

在 `src/tools/mcp_config.json` 中配置 MCP 服务器：

```json
{
  "mcpServers": {
    "git": {
      "command": "mcp-server-git",
      "args": [],
      "env": {"GIT_REPO_PATH": "."},
      "disabled": false,
      "autoApprove": ["git_status", "git_log", "git_diff"]
    },
    "weather": {
      "command": "mcp-server-weather",
      "args": [],
      "env": {},
      "disabled": false,
      "autoApprove": ["get_current_weather", "get_forecast"]
    }
  }
}
```

### 安装 MCP 服务器

```bash
# 使用 uv 安装 MCP 服务器
uv tool install mcp-server-git
uv tool install mcp-server-weather
uv tool install mcp-server-postgresql
```

### 相关文档

- [MCP 迁移指南](./MCP_MIGRATION_GUIDE.md) - 从基础工具迁移到 MCP 的详细指南
- [MCP 配置示例](./mcp_config_example.json) - MCP 服务器配置示例

### 架构优势

1. **统一管理** - 所有工具功能整合到 tools 包
2. **动态发现** - 工具自动被发现和注册，无需修改代码
3. **企业级特性** - 完善的错误处理、监控和健康检查
4. **标准化接口** - 所有工具使用统一的 MCP 协议
5. **易于扩展** - 只需安装新的 MCP 服务器即可添加新工具

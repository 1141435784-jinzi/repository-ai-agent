# Agent Lab - 企业级 Agent 开发实战

## 项目结构

```
agent-lab/
├── config.py          # 配置管理（API Key、模型参数、RAG 参数）
├── tools.py           # 工具定义（计算器、时间、订单、知识库检索）
├── prompts.py         # Prompt 模板管理
├── memory.py          # 记忆机制（对话历史滑动窗口）
├── agent.py           # LangGraph 状态图编排（ReAct Agent）
├── rag_engine.py      # RAG 引擎（混合检索 + Rerank + ChromaDB）
├── rag_evaluator.py   # RAG 质量评估（RAGAS 指标）
├── main.py            # 交互式入口
├── knowledge_base/    # 知识库文件（Markdown）
├── chroma_db/         # 向量数据库持久化目录（自动生成）
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

## 快速开始

```powershell
# 1. 激活虚拟环境
agent-lab\.venv\Scripts\Activate.ps1

# 2. 安装依赖
(.venv) PS D:\PythonWorkSpace\agent-lab> pip install -r requirements.txt

# 3. 运行(终端模式)
(.venv) PS D:\PythonWorkSpace\agent-lab> python main.py
    运行（FastAPI 模式）
(.venv) PS uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 4. 关闭虚拟环境
(.venv) PS D:\PythonWorkSpace\agent-lab> deactivate
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

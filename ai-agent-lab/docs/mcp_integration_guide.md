# MCP 集成指南

## 概述

本文档详细介绍了如何在 Agent Lab 项目中集成和使用 MCP（Model Context Protocol）工具。MCP 提供了标准化的工具调用协议，让 Agent 能够访问各种外部服务和数据源。

## 阶段1：基础MCP工具集成 ✅

### 已集成的MCP服务器

#### 1. 文件系统 MCP (`filesystem`)
- **功能**：文件读写、目录列表
- **集成点**：RAG引擎的知识库加载
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["fs-mcp"],
    "env": {"MCP_SERVER_FILESYSTEM_ROOT": "./"}
  }
  ```

#### 2. Git MCP (`git`)
- **功能**：Git状态、提交历史、差异查看
- **集成点**：新增`git_status`工具
- **配置**：
  ```json
  {
    "command": "uvx", 
    "args": ["mcp-server-git"],
    "env": {"GIT_REPO_PATH": "."}
  }
  ```

#### 3. 配置管理 MCP (`config`)
- **功能**：动态配置获取、多环境支持
- **集成点**：`MCPConfigManager`类
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["mcp-configuration-manager"],
    "env": {"CONFIG_STORE_PATH": "./config_store"}
  }
  ```

## 阶段2：业务功能MCP集成 ✅

#### 4. PostgreSQL MCP (`postgresql`)
- **功能**：数据库查询、表结构查看
- **集成点**：新增`query_database`工具
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["mcp-server-postgresql"],
    "env": {
      "POSTGRES_HOST": "localhost",
      "POSTGRES_PORT": "5432",
      "POSTGRES_DB": "agent_lab",
      "POSTGRES_USER": "postgres",
      "POSTGRES_PASSWORD": ""
    }
  }
  ```

#### 5. DuckDuckGo MCP (`duckduckgo`)
- **功能**：隐私友好的Web搜索
- **集成点**：新增`web_search`工具
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["mcp-server-duckduckgo"],
    "env": {}
  }
  ```

#### 6. 文档处理 MCP (`document`)
- **功能**：PDF/Word/Excel文档解析
- **集成点**：新增`process_document`工具
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["mcp-server-document-processing"],
    "env": {"DOCUMENT_PROCESSING_CACHE_DIR": "./.cache/documents"}
  }
  ```

#### 7. 天气查询 MCP (`weather`)
- **功能**：当前天气和预报
- **集成点**：增强现有的`query_weather`工具
- **配置**：
  ```json
  {
    "command": "uvx",
    "args": ["mcp-server-weather"],
    "env": {}
  }
  ```

## 安装步骤

### 1. 安装 uv（Python包管理器）
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装 MCP 服务器
```bash
# 使用 uvx 安装所有 MCP 服务器
uvx install fs-mcp
uvx install mcp-server-git
uvx install mcp-configuration-manager
uvx install mcp-server-postgresql
uvx install mcp-server-duckduckgo
uvx install mcp-server-document-processing
uvx install mcp-server-weather
```

### 3. 配置环境变量
在 `.env` 文件中添加：
```bash
# MCP 配置
MCP_ENABLED=true
MCP_CONFIG_SERVER=config
MCP_CONFIG_NAMESPACE=agent-lab

# PostgreSQL 配置（如果使用）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_lab
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

## 使用指南

### 1. 启动服务
```bash
# 使用虚拟环境 Python
& "agent-lab/.venv/Scripts/python.exe" scripts/run_server.py
```

### 2. 检查 MCP 状态
```bash
# 通过 API 检查
curl http://localhost:8000/mcp/status
```

### 3. 测试 MCP 工具
```bash
# 测试 Git 状态
curl -X POST http://localhost:8000/mcp/tool/call \
  -H "Content-Type: application/json" \
  -d '{"server": "git", "tool": "git_status"}'

# 测试数据库查询
curl -X POST http://localhost:8000/mcp/tool/call \
  -H "Content-Type: application/json" \
  -d '{"server": "postgresql", "tool": "list_tables"}'

# 测试 Web 搜索
curl -X POST http://localhost:8000/mcp/tool/call \
  -H "Content-Type: application/json" \
  -d '{"server": "duckduckgo", "tool": "search", "params": {"query": "Python MCP"}}'
```

### 4. 通过 Agent 使用 MCP 工具
```python
# Agent 会自动使用集成的 MCP 工具
# 例如，当用户问："Git仓库现在是什么状态？"
# Agent 会自动调用 git_status 工具

# 当用户问："帮我查一下北京的天气"
# Agent 会自动调用 query_weather 工具（MCP增强版）

# 当用户问："搜索一下最新的Python新闻"
# Agent 会自动调用 web_search 工具
```

## API 接口

### 1. MCP 状态查询
```
GET /mcp/status
```
返回所有 MCP 服务器的状态和可用工具列表。

### 2. MCP 工具调用
```
POST /mcp/tool/call
{
  "server": "git",
  "tool": "git_status",
  "params": {}
}
```
直接调用指定的 MCP 工具。

### 3. LLM 统计
```
GET /llm/stats
```
查看 LLM Gateway 的调用统计。

## 代码结构

### 新增文件
```
agent-lab/
├── src/mcp/
│   ├── __init__.py          # MCP模块导出
│   ├── client.py            # MCP客户端管理器
│   ├── config.json          # MCP服务器配置
│   ├── config_example.json  # 配置示例
│   └── GUIDE.md            # 使用指南
├── tests/
│   └── test_mcp_integration.py  # 集成测试
└── docs/
    └── mcp_integration_guide.md  # 本文档
```

### 修改文件
1. **src/config/__init__.py** - 完善 `MCPConfigManager._get_from_mcp()` 方法
2. **src/rag/engine.py** - 添加 MCP 文件系统支持
3. **src/tools/__init__.py** - 新增 4 个 MCP 增强工具
4. **src/api/server.py** - 添加 MCP 状态和工具调用接口

## 新增工具说明

### 1. `git_status()`
- **功能**：查看 Git 仓库状态
- **MCP 集成**：通过 `mcp-server-git` 获取真实状态
- **使用场景**：用户询问项目版本、变更历史

### 2. `query_database(query: str)`
- **功能**：执行数据库查询
- **MCP 集成**：通过 `mcp-server-postgresql` 执行 SQL
- **使用场景**：查询对话历史、用户数据

### 3. `web_search(query: str)`
- **功能**：Web 搜索实时信息
- **MCP 集成**：通过 `mcp-server-duckduckgo` 搜索
- **使用场景**：知识库中没有的实时信息

### 4. `process_document(file_path: str)`
- **功能**：处理文档文件
- **MCP 集成**：通过 `mcp-server-document-processing` 解析
- **使用场景**：分析 PDF、Word、Excel 文档

## 测试

### 运行集成测试
```bash
# 使用虚拟环境 Python
& "agent-lab/.venv/Scripts/python.exe" tests/test_mcp_integration.py
```

### 测试输出示例
```
开始运行MCP集成测试...
============================================================
运行测试: test_mcp_manager_initialization
============================================================
测试MCP客户端管理器初始化...
已加载 7 个MCP客户端
  - filesystem: 文件系统操作工具 (可用: True)
  - git: Git版本控制工具 (可用: True)
  - config: 配置管理工具 (可用: True)
  - postgresql: PostgreSQL数据库查询工具 (可用: True)
  - duckduckgo: 隐私��好的Web搜索工具 (可用: True)
  - document: 文档处理工具 (可用: True)
  - weather: 天气查询工具 (可用: True)
✓ 测试通过: test_mcp_manager_initialization
```

## 故障排除

### 常见问题

#### 1. MCP 服务器启动失败
**症状**：`/mcp/status` 返回客户端不可用
**解决**：
```bash
# 检查 uvx 是否安装
uvx --version

# 重新安装 MCP 服务器
uvx install mcp-server-git --force
```

#### 2. PostgreSQL 连接失败
**症状**：`query_database` 工具返回错误
**解决**：
1. 检查 PostgreSQL 服务是否运行
2. 验证 `.env` 中的数据库配置
3. 检查防火墙设置

#### 3. Web 搜索无结果
**症状**：`web_search` 返回空结果
**解决**：
1. 检查网络连接
2. 尝试不同的搜索关键词
3. 检查 DuckDuckGo API 限制

#### 4. 文档处理失败
**症状**：`process_document` 无法解析文件
**解决**：
1. 确认文件路径正确
2. 检查文件格式是否支持
3. 查看文件权限

### 日志查看
```python
# 查看 MCP 相关日志
import logging
logging.getLogger("src.mcp").setLevel(logging.DEBUG)
```

## 性能优化

### 1. 连接池
MCP 客户端管理器实现了连接池，复用 MCP 服务器连接。

### 2. 懒加载
MCP 客户端按需初始化，减少启动时间。

### 3. 缓存
配置获取使用 LRU 缓存，减少重复调用。

### 4. 超时控制
所有 MCP 调用都有超时设置，防止阻塞。

## 安全考虑

### 1. 输入验证
所有工具参数都经过验证，防止注入攻击。

### 2. 权限控制
MCP 工具调用通过 `autoApprove` 配置控制权限。

### 3. 敏感信息
数据库密码等敏感信息通过环境变量管理。

### 4. 审计日志
所有 MCP 工具调用都记录日志，便于审计。

## 扩展开发

### 添加新的 MCP 服务器
1. 在 `src/mcp/config.json` 中添加服务器配置
2. 在 `src/mcp/client.py` 的 `MCP_SERVERS` 中添加服务器信息
3. 在 `src/tools/__init__.py` 中添加对应的工具函数
4. 更新测试和文档

### 自定义 MCP 服务器
参考 [MCP 官方文档](https://modelcontextprotocol.io) 创建自定义服务器。

## 总结

通过阶段1和阶段2的集成，Agent Lab 项目现在具备了完整的 MCP 工具生态系统：

### ✅ 已实现功能
1. **7 个 MCP 服务器**：覆盖文件、Git、配置、数据库、搜索、文档、天气
2. **4 个新增工具**：Git状态、数据库查询、Web搜索、文档处理
3. **统一管理**：MCP客户端管理器统一管理所有连接
4. **完整API**：状态查询、工具调用、监控接口
5. **全面测试**：集成测试覆盖所有功能

### 🚀 业务价值
1. **知识库动态管理**：通过 MCP 文件系统支持远程知识库
2. **实时信息获取**：通过 Web 搜索补充知识库不足
3. **数据查询能力**：直接查询数据库获取历史数据
4. **文档处理能力**：支持多种格式的文档分析
5. **配置动态更新**：支持生产环境配置热更新

### 📈 下一步建议
1. **监控告警**：添加 MCP 工具调用监控和告警
2. **性能优化**：根据使用情况优化缓存策略
3. **用户体验**：添加工具使用统计和推荐
4. **扩展集成**：根据业务需求添加更多 MCP 服务器

现在你的 Agent Lab 项目已经具备了企业级的 MCP 工具集成能力，可以支持更复杂的业务场景和更高的可扩展性。
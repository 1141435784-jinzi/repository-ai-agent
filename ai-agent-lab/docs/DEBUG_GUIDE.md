# FastAPI 断点调试配置指南 (Windows)

## 📋 目录
- [问题说明](#问题说明)
- [调试配置](#调试配置)
- [使用方法](#使用方法)
- [断点调试示例](#断点调试示例)
- [常见问题](#常见问题)

---

## 问题说明

在 Windows 环境下使用 Python 虚拟环境运行 FastAPI 时遇到的问题：

1. **事件循环问题**：Windows 上 uvicorn 默认使用 `ProactorEventLoop`，但 `psycopg3` 异步连接池不兼容
2. **reload 限制**：不加 `reload` 参数会报错，但调试时可能需要稳定的环境

**解决方案**：显式设置 `SelectorEventLoop`。

---

## 调试配置

已为你创建以下文件：

### 1. `.vscode/launch.json` - VS Code 调试配置
包含 4 种调试配置：

| 配置名称 | 用途 |
|---------|------|
| `Python: FastAPI (Debug)` | 使用 reload 模式调试，适合开发 |
| `Python: FastAPI (No Reload - Debug)` | 推荐用于稳定断点调试，使用 debug_server.py |
| `Python: Current File` | 调试当前打开的 Python 文件 |
| `Python: Attach to Process` | 附加到运行中的进程 |

### 2. `debug_server.py` - 调试专用启动脚本
- 自动检测 Windows 平台并设置 `SelectorEventLoop`
- 关闭 reload 模式，更适合断点调试
- 提供友好的启动信息

---

## 使用方法

### 方法一：推荐 - 使用 debug_server.py (无 reload)

1. **确保虚拟环境已激活**
   ```powershell
   # 在项目根目录
   .\venv\Scripts\Activate.ps1
   ```

2. **在 VS Code 中设置断点**
   - 打开要调试的文件（如 `src/services/chat_service.py`）
   - 在代码行号左侧点击，设置断点（红点）

3. **启动调试**
   - 按 `F5` 或点击左侧调试图标
   - 选择配置：`Python: FastAPI (No Reload - Debug)`
   - 等待服务器启动完成

4. **触发断点**
   - 使用浏览器或 API 工具访问接口
   - 当代码执行到断点位置时会暂停

### 方法二：使用 reload 模式

1. 同样设置断点
2. 选择配置：`Python: FastAPI (Debug)`
3. 按 F5 启动
4. 注意：reload 模式下可能会影响断点稳定性

### 方法三：直接运行 debug_server.py

```powershell
# 激活虚拟环境后
python debug_server.py
```

---

## 断点调试示例

### 示例 1：调试聊天接口

1. 在 `src/services/chat_service.py` 第 45 行设置断点
2. 启动调试配置 `Python: FastAPI (No Reload - Debug)`
3. 浏览器打开 `http://localhost:8000` 发送消息
4. 当代码执行到断点时：
   - 查看变量值（鼠标悬停）
   - 单步执行（F10）
   - 进入函数（F11）
   - 继续运行（F5）

### 示例 2：调试特定函数

1. 打开要调试的函数所在文件
2. 在函数内设置断点
3. 使用 API 工具调用对应接口触发

### 调试控制台命令

在调试暂停时，可以在调试控制台执行：
- 查看变量：`variable_name`
- 执行代码：`print(some_value)`
- 调用函数：`some_function()`

---

## 调试快捷键

| 快捷键 | 功能 |
|-------|------|
| `F5` | 开始/继续调试 |
| `F9` | 切换断点 |
| `F10` | 单步跳过（不进入函数） |
| `F11` | 单步进入（进入函数） |
| `Shift+F11` | 单步跳出 |
| `Shift+F5` | 停止调试 |
| `Ctrl+Shift+F5` | 重启调试 |

---

## 常见问题

### Q1: 调试时找不到模块？
**A**: 确保 `PYTHONPATH` 包含项目根目录。launch.json 中已配置 `"PYTHONPATH": "${workspaceFolder}"`

### Q2: 断点不生效？
**A**: 
- 确保使用正确的调试配置
- 检查断点是否在可执行的代码行（不在注释或空行）
- 尝试使用 `Python: FastAPI (No Reload - Debug)` 配置

### Q3: 还是有事件循环错误？
**A**: 
- 使用 `debug_server.py` 而不是 `run_server.py`
- 检查是否正确激活了虚拟环境

### Q4: 如何调试异步代码？
**A**: 
- VS Code 对 async/await 有良好支持
- 设置断点时正常设置即可
- 单步执行时会正确处理异步调用

### Q5: 想查看数据库查询？
**A**: 
- 在数据库相关代码处设置断点
- 可以在调试控制台执行数据库查询
- 或者使用 `src/utils/logger.py` 查看日志

---

## 进阶调试技巧

### 1. 条件断点
- 右键断点 → "编辑断点"
- 设置条件，如 `thread_id == "some_value"`
- 只有满足条件时才会暂停

### 2. 日志断点
- 右键断点 → "编辑断点"
- 选择"日志消息"
- 不暂停程序，只输出日志

### 3. 数据断点
- 在调试控制台中设置
- 监视变量变化

### 4. 多线程调试
- 使用 VS Code 调试工具栏查看所有线程
- 可以切换到不同线程查看

---

## 相关文件

- [`.vscode/launch.json`](../.vscode/launch.json) - VS Code 调试配置
- [`debug_server.py`](../debug_server.py) - 调试启动脚本
- [`run_server.py`](../run_server.py) - 正常启动脚本
- [`src/api/server.py`](../src/api/server.py) - FastAPI 应用

---

*最后更新: 2026-05-18*

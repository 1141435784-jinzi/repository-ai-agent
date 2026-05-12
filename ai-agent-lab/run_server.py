"""
=== FastAPI 服务启动脚本 ===

【知识点】为什么需要 reload=True？
Windows 上 uvicorn 单进程模式默认使用 ProactorEventLoop，
psycopg3 的异步连接池不兼容 ProactorEventLoop。
但 uvicorn 在 --reload 或 --workers 模式下会自动切换为 SelectorEventLoop，
从而兼容 psycopg3 异步模式。

参考：https://www.uvicorn.org/concepts/event-loop/
"On Windows, when running with --reload or multiple workers,
 it uses SelectorEventLoop instead."

使用方式：
    python run_server.py
"""

import sys
import os

# 添加项目根目录到 Python 路径，确保可以导入 src 模块
# run_server.py 现在在 agent-lab/ 根目录中，所以需要添加当前目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
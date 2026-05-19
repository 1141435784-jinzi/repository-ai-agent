"""
FastAPI 调试启动脚本

【使用方式】：
1. 在 VS Code 中使用 "Python: FastAPI (Debug with Reload)" 配置启动
2. 或者直接运行：python debug_server.py

【注意】：
- --reload 模式下 uvicorn 会自动切换为 SelectorEventLoop（兼容 Windows + psycopg3）
- 调试时如需稳定断点，修改代码后请手动重启调试会话
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import uvicorn

if __name__ == "__main__":
    print("🚀 启动 FastAPI 调试服务器...")
    print(f"📍 项目目录: {project_root}")
    print(f"🔗 访问地址: http://localhost:8000")
    print(f"📚 文档地址: http://localhost:8000/docs")
    print(f"🔄 模式: 开发模式 (reload)")
    print("-" * 60)

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

@echo off
REM Agent Lab 指标收集模式启动脚本
REM 不需要 Docker！

echo ========================================
echo   Agent Lab 指标收集模式
echo   不需要 Docker，立即可用！
echo ========================================
echo.

REM 检查 Python
where python >nul 2>nul
if errorlevel 1 (
    echo ❌ 未找到 Python
    echo 请安装 Python 3.12 并配置 PATH
    pause
    exit /b 1
)

REM 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo ✅ 使用虚拟环境: .venv
    call ".venv\Scripts\activate.bat"
    set VENV_ENABLED=1
) else if exist "venv\Scripts\activate.bat" (
    echo ✅ 使用虚拟环境: venv
    call "venv\Scripts\activate.bat"
    set VENV_ENABLED=1
) else (
    echo ⚠️  未使用虚拟环境
    set VENV_ENABLED=0
)

REM 安装依赖
echo.
echo 📥 安装监控依赖...
pip install prometheus-client

REM 启动服务
echo.
echo 🚀 启动 Agent API 服务...
echo 📊 访问地址:
echo   健康检查: http://localhost:8000/health
echo   指标端点: http://localhost:8000/metrics  ← 核心！
echo   LLM统计: http://localhost:8000/llm/stats
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 运行服务
python -m uvicorn api:app --host 0.0.0.0 --port 8000

REM 如果服务停止，暂停显示
if errorlevel 1 (
    echo.
    echo ❌ 服务启动失败
    pause
)
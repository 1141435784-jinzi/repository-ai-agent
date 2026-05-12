# Agent Lab 指标收集模式启动脚本
# 不需要 Docker！

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Agent Lab 指标收集模式" -ForegroundColor Yellow
Write-Host "  不需要 Docker，立即可用！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python 未找到或未配置 PATH" -ForegroundColor Red
        Write-Host "请安装 Python 3.12 并配置 PATH" -ForegroundColor Yellow
        pause
        exit 1
    }
} catch {
    Write-Host "❌ Python 检查失败: $_" -ForegroundColor Red
    pause
    exit 1
}

# 检查虚拟环境
$venvPath = $null
if (Test-Path ".venv\Scripts\activate.ps1") {
    $venvPath = ".venv\Scripts\activate.ps1"
    Write-Host "✅ 使用虚拟环境: .venv" -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\activate.ps1") {
    $venvPath = "venv\Scripts\activate.ps1"
    Write-Host "✅ 使用虚拟环境: venv" -ForegroundColor Green
} else {
    Write-Host "⚠️  未使用虚拟环境" -ForegroundColor Yellow
}

# 安装依赖
Write-Host ""
Write-Host "📥 安装监控依赖..." -ForegroundColor Cyan
try {
    if ($venvPath) {
        # 激活虚拟环境
        & $venvPath
    }
    
    pip install prometheus-client
    Write-Host "✅ 依赖安装完成" -ForegroundColor Green
} catch {
    Write-Host "❌ 依赖安装失败: $_" -ForegroundColor Red
    Write-Host "尝试继续..." -ForegroundColor Yellow
}

# 启动服务
Write-Host ""
Write-Host "🚀 启动 Agent API 服务..." -ForegroundColor Cyan
Write-Host "📊 访问地址:" -ForegroundColor Yellow
Write-Host "  健康检查: http://localhost:8000/health" -ForegroundColor Gray
Write-Host "  指标端点: http://localhost:8000/metrics ← 核心！" -ForegroundColor Green
Write-Host "  LLM统计: http://localhost:8000/llm/stats" -ForegroundColor Gray
Write-Host ""
Write-Host "按 Ctrl+C 停止服务" -ForegroundColor Yellow
Write-Host ""

try {
    # 运行服务
    python -m uvicorn api:app --host 0.0.0.0 --port 8000
} catch {
    Write-Host ""
    Write-Host "❌ 服务启动失败: $_" -ForegroundColor Red
    pause
}
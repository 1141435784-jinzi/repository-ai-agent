#!/bin/bash

# Agent Lab 指标收集系统部署脚本
# 核心指标收集功能，基于 Google SRE 四大黄金指标

set -e

echo "🚀 Agent Lab 指标收集系统部署"
echo "=" * 60
echo ""

echo "📊 功能说明:"
echo "   • 基于 Google SRE 四大黄金指标设计"
echo "   • 集成 Prometheus 客户端库"
echo "   • 暴露标准 /metrics 端点"
echo "   • 不需要 Docker，立即可用"
echo ""

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
if ! command -v python &> /dev/null; then
    echo "❌ Python 未找到"
    echo "请确保 Python 3.12 已安装并配置 PATH"
    exit 1
fi

python_version=$(python --version 2>&1)
echo "✅ $python_version"

# 检查虚拟环境
echo ""
echo "🔍 检查虚拟环境..."
if [ -f ".venv/Scripts/activate" ]; then
    echo "✅ 检测到虚拟环境: .venv/"
    VENV_ACTIVATE=".venv/Scripts/activate"
elif [ -f "venv/Scripts/activate" ]; then
    echo "✅ 检测到虚拟环境: venv/"
    VENV_ACTIVATE="venv/Scripts/activate"
else
    echo "⚠️  未检测到虚拟环境"
    echo "建议使用虚拟环境，但可以继续"
    VENV_ACTIVATE=""
fi

# 安装依赖
echo ""
echo "📥 安装监控依赖..."
if [ -n "$VENV_ACTIVATE" ]; then
    # 使用虚拟环境
    source "$VENV_ACTIVATE"
    pip install prometheus-client
    echo "✅ 依赖安装完成 (使用虚拟环境)"
else
    # 使用系统 Python
    pip install prometheus-client
    echo "✅ 依赖安装完成 (使用系统 Python)"
fi

# 显示已实现的指标
echo ""
echo "🔧 配置完成！"
echo "-" * 40
echo ""
echo "📈 已实现的监控指标:"

echo "1. 延迟指标 (Latency)"
echo "   • llm_call_duration_seconds - LLM 调用耗时"
echo "   • agent_execution_duration_seconds - Agent 执行耗时"
echo "   • api_request_duration_seconds - API 请求耗时"
echo "   • rag_retrieval_duration_seconds - RAG 检索耗时"
echo ""

echo "2. 流量指标 (Traffic)"
echo "   • llm_calls_total - LLM 总调用次数"
echo "   • agent_calls_total - Agent 总调用次数"
echo "   • api_requests_total - API 总请求次数"
echo "   • user_sessions_total - 用户会话总数"
echo ""

echo "3. 错误指标 (Errors)"
echo "   • llm_errors_total - LLM 错误总数"
echo "   • agent_errors_total - Agent 错误总数"
echo "   • api_errors_total - API 错误总数"
echo "   • llm_fallbacks_total - 降级调用总数"
echo ""

echo "4. 饱和度指标 (Saturation)"
echo "   • user_sessions_active - 活跃用户会话数"
echo "   • llm_tokens_total - LLM Token 总使用量"
echo "   • llm_cost_total_yuan - LLM 总成本（元）"
echo ""

echo "5. 业务指标"
echo "   • smart_routing_total - 智能路由调用总数"
echo "   • rag_relevance_score - RAG 检索相关性分数"
echo ""

# 启动说明
echo "🚀 启动服务:"
echo ""
if [ -n "$VENV_ACTIVATE" ]; then
    echo "   使用虚拟环境启动:"
    echo "   source \"$VENV_ACTIVATE\""
    echo "   python -m uvicorn api:app --host 0.0.0.0 --port 8000"
else
    echo "   直接启动:"
    echo "   python -m uvicorn api:app --host 0.0.0.0 --port 8000"
fi
echo ""
echo "   或使用批处理文件:"
echo "   start-metrics-only.bat"
echo ""

# 访问地址
echo "📊 访问地址:"
echo "   • 健康检查:     http://localhost:8000/health"
echo "   • 指标端点:     http://localhost:8000/metrics"
echo "   • LLM 统计:     http://localhost:8000/llm/stats"
echo "   • 创建会话:     http://localhost:8000/session/new (POST)"
echo "   • 聊天接口:     http://localhost:8000/chat (POST)"
echo ""

# 验证方法
echo "🧪 验证安装:"
echo "   python test_metrics.py"
echo ""
echo "   或手动验证:"
echo "   curl http://localhost:8000/metrics"
echo ""

# 使用示例
echo "💡 使用示例:"
echo "   1. 启动服务后，访问 http://localhost:8000/metrics"
echo "   2. 查看 Prometheus 格式的指标数据"
echo "   3. 可以集成到现有的 Prometheus 监控系统"
echo "   4. 或使用 curl 定期抓取指标进行分析"
echo ""

echo "=" * 60
echo "✅ 指标收集系统部署完成！"
echo ""
echo "📚 相关文件:"
echo "   • prometheus_metrics.py - 指标定义模块"
echo "   • llm_service.py - LLM 指标集成"
echo "   • api.py - API 指标集成"
echo "   • test_metrics.py - 测试脚本"
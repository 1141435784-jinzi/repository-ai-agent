# Ollama 本地模型部署指南

## 概述

本文档提供在企业级项目中部署和管理 Ollama 本地大语言模型的完整指南。Ollama 是一个轻量级的本地模型部署工具，支持多种开源模型，适合开发环境和中小规模生产部署。

## 核心优势

1. **数据安全**：敏感数据不出境，符合企业合规要求
2. **成本控制**：无API调用费用，适合高频使用场景
3. **离线可用**：网络中断时仍可提供服务
4. **灵活定制**：支持模型微调和参数调整
5. **生态集成**：完美集成 LangChain/LangGraph 生态

## 安装部署

### Windows 安装

```bash
# 方法1：使用安装程序
# 从 https://ollama.com/download 下载 Windows 安装程序

# 方法2：使用 PowerShell
curl -fsSL https://ollama.com/install.sh | sh

# 验证安装
ollama --version
```

### 启动服务

```bash
# 启动 Ollama 服务（默认端口 11434）
ollama serve

# 后台运行（Windows）
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
```

## 模型管理

### 常用模型推荐

| 模型 | 大小 | 特点 | 适用场景 |
|------|------|------|----------|
| `qwen2.5:3b` | ~2GB | 中文优化，性能优秀 | 中文对话、文档处理 |
| `llama3.2:3b` | ~2GB | Meta官方，英文优秀 | 代码生成、英文对话 |
| `gemma2:2b` | ~1.5GB | Google出品，轻量高效 | 快速响应、资源受限环境 |
| `mistral:7b` | ~4GB | 7B参数，能力强 | 复杂推理、高质量输出 |

### 模型操作命令

```bash
# 拉取模型
ollama pull qwen2.5:3b

# 查看已下载模型
ollama list

# 运行模型
ollama run qwen2.5:3b

# 删除模型
ollama rm qwen2.5:3b
```

## 项目集成配置

### 环境变量配置

在 `.env` 文件中配置：

```env
# Ollama 基础配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

# 性能优化配置
OLLAMA_NUM_GPU=1           # GPU数量（0=CPU，1=单GPU）
OLLAMA_NUM_CTX=4096       # 上下文长度
OLLAMA_TEMPERATURE=0.7    # 输出随机性
OLLAMA_TIMEOUT=120        # 超时时间（秒）
OLLAMA_MAX_TOKENS=2048    # 最大生成token数
```

### 代码集成示例

```python
from llm_service import get_llm, invoke_with_fallback
from langchain.schema import HumanMessage, SystemMessage

# 获取 Ollama LLM 实例
llm = get_llm(provider="ollama")

# 调用本地模型
messages = [
    SystemMessage(content="你是一个有帮助的AI助手。"),
    HumanMessage(content="请介绍一下你自己。")
]

response = invoke_with_fallback(
    messages=messages,
    provider="ollama",  # 指定使用本地模型
    temperature=0.7
)
```

## 生产环境优化

### GPU 配置优化

```bash
# 查看可用GPU
nvidia-smi

# 配置Ollama使用GPU
# 在 .env 中设置
OLLAMA_NUM_GPU=1  # 使用1个GPU
```

### 内存管理

```python
# 根据可用内存选择模型
# 8GB内存：3B模型
# 16GB内存：7B模型  
# 32GB内存：13B模型

# 监控内存使用
import psutil
memory_info = psutil.virtual_memory()
print(f"可用内存: {memory_info.available / 1024**3:.1f} GB")
```

### 性能调优参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `OLLAMA_NUM_CTX` | 4096 | 平衡性能和上下文长度 |
| `OLLAMA_TIMEOUT` | 120 | 避免复杂任务超时 |
| `OLLAMA_MAX_TOKENS` | 2048 | 控制生成长度 |
| `temperature` | 0.7 | 日常对话的合理随机性 |

## 监控与运维

### 健康检查

```python
from llm_service import check_ollama_health

# 定期健康检查
if check_ollama_health():
    print("Ollama 服务正常")
else:
    print("Ollama 服务异常，触发告警")
```

### 模型监控

```python
from llm_service import get_ollama_models, get_ollama_model_info

# 获取模型列表
models = get_ollama_models()
print(f"可用模型: {len(models)}个")

# 检查模型健康状态
model_info = get_ollama_model_info("qwen2.5:3b")
if model_info:
    print(f"模型参数: {model_info.get('parameter_size')}")
```

### 性能指标收集

```python
from llm_service import get_call_stats

# 获取调用统计
stats = get_call_stats()
print(f"总调用次数: {stats.total_calls}")
print(f"成功调用: {stats.success_calls}")
print(f"降级调用: {stats.fallback_calls}")
print(f"各Provider调用分布: {stats.calls_by_provider}")
```

## 故障排查

### 常见问题

1. **连接失败**
   ```
   错误：无法连接到 Ollama 服务
   解决：确保 ollama serve 正在运行
   ```

2. **模型不存在**
   ```
   错误：配置的模型不存在
   解决：运行 ollama pull <model_name> 下载模型
   ```

3. **GPU内存不足**
   ```
   错误：CUDA out of memory
   解决：减小模型大小或增加GPU内存
   ```

4. **响应超时**
   ```
   错误：请求超时
   解决：增加 OLLAMA_TIMEOUT 值
   ```

### 诊断命令

```bash
# 检查Ollama服务状态
curl http://localhost:11434/api/tags

# 查看日志
ollama logs

# 检查端口占用
netstat -ano | findstr :11434
```

## 安全考虑

1. **访问控制**
   - 限制Ollama服务仅本地访问
   - 使用防火墙规则限制外部访问

2. **模型安全**
   - 仅使用可信来源的模型
   - 定期更新模型版本

3. **数据保护**
   - 敏感数据本地处理
   - 对话记录加密存储

## 扩展功能

### 多模型切换

```python
# 根据任务类型选择模型
def select_model_by_task(task_type: str) -> str:
    if task_type == "code_generation":
        return "llama3.2:3b"
    elif task_type == "chinese_dialogue":
        return "qwen2.5:3b"
    else:
        return "gemma2:2b"  # 默认轻量模型
```

### 模型预热

```python
# 服务启动时预热模型
def warm_up_ollama():
    llm = get_llm(provider="ollama")
    # 发送简单请求预热模型
    messages = [HumanMessage(content="Hello")]
    try:
        llm.invoke(messages)
        print("模型预热完成")
    except:
        print("模型预热失败")
```

## 最佳实践

1. **开发环境**：使用轻量模型（3B参数）
2. **测试环境**：与实际生产环境配置一致
3. **生产环境**：
   - 使用性能监控
   - 设置自动扩缩容
   - 定期备份模型数据

4. **版本管理**：
   - 记录模型版本
   - 测试新版本兼容性
   - 保留回滚能力

## 后续优化方向

1. **容器化部署**：使用 Docker 封装 Ollama
2. **负载均衡**：多实例部署
3. **自动扩缩容**：根据负载动态调整
4. **模型压缩**：使用量化技术减小模型大小

---

**最后更新**：2026年5月4日  
**适用版本**：Ollama 0.5.0+，Python 3.12+
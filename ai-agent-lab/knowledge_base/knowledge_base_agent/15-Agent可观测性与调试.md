# 09 - Agent 可观测性与调试

> 生产环境不能盲飞。你得看到 Agent 在想什么、做什么、花了多少钱、哪里出了问题。

---

## 一、为什么可观测性是生产级 Agent 的必备能力？

### 1.1 现实类比：Agent 可观测性就像飞机的黑匣子

飞机出事故后，调查人员靠黑匣子还原事故经过。Agent 出问题后，你靠什么还原？

没有可观测性：
- 用户说"Agent 给了错误答案"→ 你不知道 Agent 调用了哪些工具、看到了什么数据、怎么推理的
- 老板问"Agent 每天花多少钱"→ 你不知道
- 某个 Agent 突然变慢了 → 你不知道是模型慢了还是工具慢了

有了可观测性：
- 每一步都有 Trace，像回放录像一样还原 Agent 的完整执行过程
- Token 消耗、延迟、成功率一目了然
- 哪个工具最慢、哪个 Prompt 效果差，数据说话

### 1.2 可观测性的三大支柱

| 支柱 | 含义 | Agent 场景 |
|------|------|-----------|
| Tracing（追踪） | 记录完整的执行链路 | Agent 的每一步：LLM 调用、工具调用、状态变化 |
| Metrics（指标） | 量化的性能数据 | Token 消耗、延迟、成功率、成本 |
| Logging（日志） | 详细的运行日志 | 错误信息、异常堆栈、调试信息 |

---

## 二、LangSmith：Agent 的专业观测平台

### 2.1 LangSmith 是什么？

LangSmith 是 LangChain 官方的可观测性平台，专为 LLM 应用设计。它不是通用的 APM（如 Datadog），而是专门理解 LLM 调用链的工具。

### 2.2 集成 LangSmith

```python
import os

# 只需设置环境变量，LangChain/LangGraph 自动上报 Trace
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "my-agent-project"
os.environ["LANGSMITH_TRACING"] = "true"

# 之后所有的 LangChain/LangGraph 调用都会自动被追踪
# 不需要修改任何业务代码
```

### 2.3 自定义 Trace 元数据

```python
from langsmith import traceable

@traceable(
    name="process_customer_request",
    tags=["customer-service", "production"],
    metadata={"version": "2.1", "team": "cs-team"}
)
def process_request(customer_id: str, request: str) -> str:
    """处理客户请求 —— 自动被 LangSmith 追踪"""
    result = agent.invoke({
        "messages": [HumanMessage(content=request)],
        "customer_id": customer_id,
    })
    return result

# 在 Trace 中添加反馈
from langsmith import Client

client = Client()
client.create_feedback(
    run_id=run_id,
    key="user_satisfaction",
    score=0.8,
    comment="用户表示满意"
)
```

---

## 三、自建可观测性方案

### 3.1 基于 Callback 的追踪

```python
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
import json
import logging

logger = logging.getLogger("agent_trace")

class AgentTraceCallback(BaseCallbackHandler):
    """自定义追踪回调 —— 记录 Agent 的完整执行链路"""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.start_time = datetime.now()
        self.steps: list[dict] = []
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.steps.append({
            "type": "llm_start",
            "time": datetime.now().isoformat(),
            "model": serialized.get("name", "unknown"),
        })

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        self.steps.append({
            "type": "llm_end",
            "time": datetime.now().isoformat(),
            "tokens": tokens,
            "output_preview": str(response.generations[0][0].text)[:200],
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.steps.append({
            "type": "tool_start",
            "time": datetime.now().isoformat(),
            "tool": serialized.get("name", "unknown"),
            "input": input_str[:500],
        })

    def on_tool_end(self, output, **kwargs):
        self.steps.append({
            "type": "tool_end",
            "time": datetime.now().isoformat(),
            "output": str(output)[:500],
        })

    def on_tool_error(self, error, **kwargs):
        self.steps.append({
            "type": "tool_error",
            "time": datetime.now().isoformat(),
            "error": str(error),
        })
        logger.error(f"[{self.trace_id}] Tool error: {error}")

    def get_summary(self) -> dict:
        """获取执行摘要"""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "trace_id": self.trace_id,
            "duration_seconds": duration,
            "total_tokens": self.total_tokens,
            "total_steps": len(self.steps),
            "errors": [s for s in self.steps if s["type"] == "tool_error"],
        }

# 使用
trace = AgentTraceCallback(trace_id="req-20260417-001")
result = agent.invoke(input_data, config={"callbacks": [trace]})
print(json.dumps(trace.get_summary(), indent=2, ensure_ascii=False))
```

### 3.2 关键指标监控

```python
from dataclasses import dataclass, field
from datetime import datetime
import time

@dataclass
class AgentMetrics:
    """Agent 运行指标收集器"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.request_count, 1)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    def record_request(self, success: bool, latency: float, tokens: int):
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.latencies.append(latency)
        self.total_tokens += tokens

    def report(self) -> dict:
        return {
            "requests": self.request_count,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_latency": f"{self.avg_latency:.2f}s",
            "p95_latency": f"{self.p95_latency:.2f}s",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
        }

# 使用
metrics = AgentMetrics()

async def handle_request(request):
    start = time.time()
    try:
        result = await agent.ainvoke(request)
        metrics.record_request(success=True, latency=time.time() - start, tokens=result.get("tokens", 0))
        return result
    except Exception as e:
        metrics.record_request(success=False, latency=time.time() - start, tokens=0)
        raise
```

---

## 四、Agent 评估（Evaluation）

### 4.1 为什么需要评估？

**企业场景**：你改了一版 Prompt，怎么知道是变好了还是变差了？靠感觉？不行。需要量化评估。

### 4.2 评估维度

| 维度 | 含义 | 评估方法 |
|------|------|----------|
| 正确性 | 回答是否正确 | 与标准答案对比 / LLM 评估 |
| 忠实度 | 是否基于检索内容（RAG） | 检查回答是否有检索内容支撑 |
| 相关性 | 回答是否切题 | LLM 评估 |
| 有害性 | 是否包含有害内容 | 安全分类器 |
| 工具使用 | 是否正确使用了工具 | 检查工具调用序列 |
| 效率 | Token 消耗和延迟 | 直接测量 |

### 4.3 使用 LangSmith 评估

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 创建评估数据集
dataset = client.create_dataset("customer-service-eval")

# 添加测试用例
client.create_examples(
    inputs=[
        {"question": "我的订单 ORD-12345 到哪了？"},
        {"question": "我要退货，买了三天的手机屏幕碎了"},
        {"question": "你们的营业时间是什么时候？"},
    ],
    outputs=[
        {"answer": "应该查询物流信息并返回具体状态"},
        {"answer": "应该引导用户走退货流程，并表示同情"},
        {"answer": "应该从知识库中检索营业时间信息"},
    ],
    dataset_id=dataset.id,
)

# 定义评估函数
def correctness_evaluator(run, example):
    """评估回答的正确性"""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")

    # 用 LLM 评估
    eval_result = llm.invoke(
        f"评估以下回答是否满足预期。\n预期：{reference}\n实际：{prediction}\n评分（0-1）："
    )
    score = float(eval_result.content.strip())
    return {"key": "correctness", "score": score}

# 运行评估
results = evaluate(
    lambda inputs: agent.invoke(inputs),
    data=dataset,
    evaluators=[correctness_evaluator],
)
```

---

## 五、调试技巧

### 5.1 LangGraph 状态检查

```python
# 打印每一步的状态变化
for event in app.stream(input_data, config, stream_mode="debug"):
    print(f"类型: {event['type']}")
    if event['type'] == 'task':
        print(f"节点: {event['payload'].get('name', 'N/A')}")
    elif event['type'] == 'task_result':
        print(f"结果: {json.dumps(event['payload'].get('result', {}), ensure_ascii=False)[:200]}")
    print("---")
```

### 5.2 常见问题排查清单

| 问题 | 可能原因 | 排查方法 |
|------|----------|----------|
| Agent 不调用工具 | 工具描述不清晰 / Prompt 没引导 | 检查工具 docstring，优化 Prompt |
| Agent 调错工具 | 工具之间描述重叠 | 让工具描述更具区分度 |
| Agent 死循环 | 条件路由逻辑有误 | 检查条件边函数，加最大迭代限制 |
| 回答有幻觉 | RAG 检索不到 / Prompt 没约束 | 检查检索结果，加"不知道就说不知道" |
| 延迟太高 | 工具调用太多 / 模型太慢 | 查看 Trace，定位瓶颈 |
| Token 消耗过高 | 消息列表太长 / 循环次数多 | 加消息裁剪，限制迭代次数 |

---

## 六、本章面试要点

1. **Agent 可观测性包含哪些方面？**
   → Tracing（执行链路追踪）、Metrics（性能指标）、Logging（运行日志）

2. **LangSmith 和通用 APM（如 Datadog）有什么区别？**
   → LangSmith 专为 LLM 应用设计，理解 LLM 调用链、工具调用、Token 消耗等 LLM 特有概念

3. **如何评估 Agent 的效果？**
   → 构建评估数据集 + 定义评估指标（正确性、忠实度、相关性等）+ 自动化评估 + A/B 测试

4. **Agent 死循环怎么排查？**
   → 查看 Trace 中的节点执行序列，检查条件路由逻辑，加最大迭代限制

5. **生产环境中如何控制 Agent 的成本？**
   → Token 消耗监控 + 成本控制中间件 + 消息裁剪 + 迭代次数限制 + 模型降级策略

# Agent 版本管理与灰度发布

> 适用场景：生产 Agent 系统的 Prompt/模型/工具变更需要安全上线，避免一次更新影响所有用户
> 最后更新：2026 年 4 月

---

## 一、为什么 Agent 需要版本管理

Agent 系统的"代码"不只是 Python 代码，还包括：

- **Prompt**：System Prompt 的措辞变化直接影响 Agent 行为
- **模型**：从 GPT-4o 切换到 GPT-4o-mini 可能改变回答质量
- **工具**：新增/修改工具影响 Agent 的能力边界
- **RAG 配置**：chunk_size、检索权重、Rerank 模型的变化影响检索质量
- **知识库**：文档更新可能引入错误信息

这些变更中任何一个都可能导致 Agent 行为退化，但又不像传统代码那样容易通过单元测试覆盖。

**现实例子**：某团队修改了 System Prompt 中的一句话，导致 Agent 在特定场景下不再调用工具，直接编造答案。因为没有灰度发布，所有用户同时受影响，直到用户投诉才发现。

---

## 二、Prompt 版本控制

### 2.1 Prompt 作为配置管理

```python
# prompts/v1.0.py
SYSTEM_PROMPT_V1_0 = """你是一个企业知识助手。
回答问题时请基于检索到的上下文，不要编造信息。"""

# prompts/v1.1.py
SYSTEM_PROMPT_V1_1 = """你是一个企业知识助手。
回答问题时请基于检索到的上下文，不要编造信息。
如果上下文中没有相关信息，请明确告知用户"我没有找到相关信息"。"""

# prompts/v2.0.py
SYSTEM_PROMPT_V2_0 = """你是一个企业知识助手。
## 回答规则：
1. 必须基于检索到的上下文回答
2. 没有相关信息时明确告知
3. 引用来源文档名称
4. 对不确定的内容标注"待确认"
"""
```

### 2.2 Prompt 版本注册表

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptVersion:
    version: str
    content: str
    description: str
    created_at: datetime
    author: str
    is_active: bool = False

class PromptRegistry:
    """Prompt 版本注册表"""

    def __init__(self):
        self.versions: dict[str, PromptVersion] = {}
        self.active_version: str = ""

    def register(self, version: PromptVersion):
        self.versions[version.version] = version

    def activate(self, version: str):
        """激活指定版本"""
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")
        # 停用旧版本
        if self.active_version:
            self.versions[self.active_version].is_active = False
        # 激活新版本
        self.versions[version].is_active = True
        self.active_version = version

    def get_active_prompt(self) -> str:
        return self.versions[self.active_version].content

    def rollback(self, version: str):
        """回滚到指定版本"""
        self.activate(version)
        logger.warning(f"Prompt 已回滚到版本 {version}")
```

---

## 三、A/B 测试

### 3.1 流量分配

```python
import hashlib

class ABTestRouter:
    """A/B 测试流量路由"""

    def __init__(self, experiments: dict):
        """
        experiments = {
            "prompt_v2": {
                "control": {"prompt": PROMPT_V1, "weight": 0.8},
                "treatment": {"prompt": PROMPT_V2, "weight": 0.2},
            }
        }
        """
        self.experiments = experiments

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """根据用户 ID 确定性地分配实验组"""
        # 使用 hash 确保同一用户始终在同一组
        hash_input = f"{experiment_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        ratio = (hash_value % 1000) / 1000.0

        experiment = self.experiments[experiment_name]
        control_weight = experiment["control"]["weight"]

        if ratio < control_weight:
            return "control"
        else:
            return "treatment"

    def get_config(self, experiment_name: str, user_id: str) -> dict:
        """获取用户对应的实验配置"""
        variant = self.get_variant(experiment_name, user_id)
        return self.experiments[experiment_name][variant]
```

### 3.2 在 Agent 中使用 A/B 测试

```python
ab_router = ABTestRouter(experiments={
    "new_prompt": {
        "control": {"prompt": PROMPT_V1, "model": "gpt-4o", "weight": 0.8},
        "treatment": {"prompt": PROMPT_V2, "model": "gpt-4o", "weight": 0.2},
    }
})

@app.post("/chat")
async def chat(request: ChatRequest):
    # 获取实验配置
    config = ab_router.get_config("new_prompt", request.user_id)
    variant = ab_router.get_variant("new_prompt", request.user_id)

    # 使用对应配置构建 Agent
    agent = create_agent(
        model=init_chat_model(f"openai:{config['model']}"),
        tools=tools,
        prompt=config["prompt"],
    )

    result = await agent.ainvoke(...)

    # 记录实验数据
    log_experiment(
        experiment="new_prompt",
        variant=variant,
        user_id=request.user_id,
        question=request.message,
        answer=result["messages"][-1].content,
    )

    return result
```

### 3.3 评估 A/B 测试结果

```python
def evaluate_experiment(experiment_name: str) -> dict:
    """评估 A/B 测试结果"""
    control_data = get_experiment_data(experiment_name, "control")
    treatment_data = get_experiment_data(experiment_name, "treatment")

    return {
        "control": {
            "sample_size": len(control_data),
            "avg_faithfulness": avg(d["faithfulness"] for d in control_data),
            "avg_relevancy": avg(d["relevancy"] for d in control_data),
            "avg_user_rating": avg(d["user_rating"] for d in control_data),
            "avg_latency_ms": avg(d["latency_ms"] for d in control_data),
            "avg_cost_per_request": avg(d["cost"] for d in control_data),
        },
        "treatment": {
            "sample_size": len(treatment_data),
            "avg_faithfulness": avg(d["faithfulness"] for d in treatment_data),
            "avg_relevancy": avg(d["relevancy"] for d in treatment_data),
            "avg_user_rating": avg(d["user_rating"] for d in treatment_data),
            "avg_latency_ms": avg(d["latency_ms"] for d in treatment_data),
            "avg_cost_per_request": avg(d["cost"] for d in treatment_data),
        },
    }
```

---

## 四、金丝雀发布

金丝雀发布是灰度发布的一种：先让少量用户使用新版本，观察一段时间没问题后再全量推送。

### 4.1 发布流程

```
阶段 1：金丝雀（5% 流量）
    ↓ 观察 1-2 小时，检查错误率、延迟、用户反馈
阶段 2：小规模灰度（20% 流量）
    ↓ 观察 1 天，对比 RAGAS 指标
阶段 3：大规模灰度（50% 流量）
    ↓ 观察 1-3 天
阶段 4：全量发布（100% 流量）

任何阶段发现问题 → 立即回滚到上一个稳定版本
```

### 4.2 实现

```python
class CanaryDeployer:
    """金丝雀发布管理"""

    def __init__(self):
        self.stages = [0.05, 0.20, 0.50, 1.0]
        self.current_stage = 0
        self.new_version_config = None
        self.old_version_config = None

    def start_canary(self, new_config: dict):
        """开始金丝雀发布"""
        self.new_version_config = new_config
        self.old_version_config = get_current_config()
        self.current_stage = 0
        logger.info(f"金丝雀发布开始，流量比例: {self.stages[0]*100}%")

    def promote(self):
        """推进到下一阶段"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            logger.info(f"金丝雀推进到阶段 {self.current_stage}，"
                       f"流量比例: {self.stages[self.current_stage]*100}%")

    def rollback(self):
        """回滚"""
        self.new_version_config = None
        self.current_stage = 0
        logger.warning("金丝雀发布已回滚")

    def get_config_for_user(self, user_id: str) -> dict:
        """根据用户决定使用新版还是旧版"""
        if self.new_version_config is None:
            return self.old_version_config

        # 确定性分配
        ratio = hash(user_id) % 100 / 100.0
        if ratio < self.stages[self.current_stage]:
            return self.new_version_config
        return self.old_version_config
```

---

## 五、回滚机制

### 5.1 快速回滚

```python
class VersionManager:
    """版本管理器，支持快速回滚"""

    def __init__(self):
        self.version_history: list[dict] = []
        self.current_version: dict = None

    def deploy(self, config: dict):
        """部署新版本"""
        if self.current_version:
            self.version_history.append(self.current_version)
        self.current_version = {
            **config,
            "deployed_at": datetime.now().isoformat(),
        }

    def rollback(self, steps: int = 1):
        """回滚 N 个版本"""
        if len(self.version_history) < steps:
            raise ValueError("没有足够的历史版本可回滚")

        for _ in range(steps):
            self.current_version = self.version_history.pop()

        logger.warning(f"已回滚 {steps} 个版本")
        return self.current_version
```

### 5.2 自动回滚触发条件

```python
class AutoRollbackMonitor:
    """自动回滚监控"""

    def __init__(self, version_manager: VersionManager):
        self.vm = version_manager
        self.error_threshold = 0.05      # 错误率超过 5%
        self.latency_threshold = 10000   # 延迟超过 10 秒
        self.rating_threshold = 3.0      # 用户评分低于 3 分

    def check_and_rollback(self, metrics: dict) -> bool:
        """检查指标，必要时自动回滚"""
        reasons = []

        if metrics["error_rate"] > self.error_threshold:
            reasons.append(f"错误率 {metrics['error_rate']:.1%} 超过阈值")

        if metrics["avg_latency_ms"] > self.latency_threshold:
            reasons.append(f"平均延迟 {metrics['avg_latency_ms']}ms 超过阈值")

        if metrics["avg_user_rating"] < self.rating_threshold:
            reasons.append(f"用户评分 {metrics['avg_user_rating']:.1f} 低于阈值")

        if reasons:
            logger.critical(f"触发自动回滚: {'; '.join(reasons)}")
            self.vm.rollback()
            return True

        return False
```

---

## 六、变更管理最佳实践

| 变更类型 | 风险等级 | 建议发布方式 |
|---|---|---|
| Prompt 措辞微调 | 中 | A/B 测试 → 金丝雀 |
| 模型切换 | 高 | 金丝雀 + RAGAS 评估 |
| 新增工具 | 中 | 金丝雀 + 功能开关 |
| 知识库更新 | 低-中 | 更新后跑 RAGAS 回归测试 |
| RAG 参数调整 | 中 | A/B 测试 + RAGAS 对比 |
| 架构重构 | 高 | 影子模式（双跑对比）→ 金丝雀 |

---

## 七、检查清单

- [ ] Prompt 有版本控制（不是直接改线上的字符串）
- [ ] 支持 A/B 测试（同一用户始终在同一组）
- [ ] 支持金丝雀发布（分阶段放量）
- [ ] 有快速回滚机制（一键回到上一个稳定版本）
- [ ] 有自动回滚触发条件（错误率、延迟、评分）
- [ ] 每次变更都有 RAGAS 评估对比
- [ ] 变更记录可追溯（谁在什么时候改了什么）

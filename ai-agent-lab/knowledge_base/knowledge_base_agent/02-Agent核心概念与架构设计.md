# 01 - Agent 核心概念与架构设计

> 这是整套文档的地基。搞不清 Agent 的本质，后面的一切都是空中楼阁。

---

## 一、Agent 到底是什么？

### 1.1 一句话定义

Agent = LLM + 感知 + 决策 + 行动 + 记忆

Agent 四大核心模块

```
                    ┌─────────────────────────────────┐
                    │         AI Agent                 │
                    │                                  │
                    │  ┌───────────┐  ┌───────────┐   │
                    │  │  Planning  │  │  Memory   │   │
                    │  │  (规划)    │  │  (记忆)    │   │
                    │  └─────┬─────┘  └─────┬─────┘   │
                    │        │              │          │
                    │  ┌─────▼──────────────▼─────┐   │
                    │  │        LLM (大脑)         │   │
                    │  │   理解、推理、决策、生成    │   │
                    │  └─────────────┬─────────────┘   │
                    │               │                  │
                    │  ┌────────────▼────────────┐     │
                    │  │      Tools (工具)        │     │
                    │  │  搜索/数据库/API/计算...  │     │
                    │  └─────────────────────────┘     │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │  外部环境     │
                            │ (用户/系统)   │
                            └──────────────┘
```

它不是一个简单的"问答机器人"，而是一个**能自主规划、调用工具、根据反馈调整策略的智能体**。

### 1.2 现实类比：Agent 就像一个新入职的高级员工

想象你是一家电商公司的 CTO，你招了一个新的高级工程师（Agent）：
- **大脑（LLM）**：他有丰富的知识储备，能理解你的需求
- **工具（Tools）**：他能用公司的内部系统——查订单、改库存、发邮件
- **记忆（Memory）**：他记得之前的对话和处理过的工单
- **决策（Reasoning）**：面对复杂问题，他会先想一想，拆解步骤，再动手
- **自主性（Autonomy）**：你不需要告诉他每一步怎么做，给个目标他就能自己搞定

没有 Agent 之前：用户说"帮我查一下订单 12345 的物流状态，如果已签收就自动发满意度调查邮件"——你需要写一堆 if-else 的硬编码流程。

有了 Agent 之后：Agent 自己理解意图 → 调用订单查询工具 → 判断状态 → 决定是否调用邮件工具 → 完成任务。**流程是 Agent 自己规划的，不是你硬编码的。**

### 1.3 Agent vs ChatBot vs Pipeline

| 维度 | ChatBot | Pipeline | Agent |
|------|---------|----------|-------|
| 决策方式 | 固定规则/模板匹配 | 预定义流程 | LLM 自主决策 |
| 工具调用 | 无或有限 | 固定顺序调用 | 动态选择调用 |
| 流程灵活性 | 低 | 中（分支有限） | 高（运行时决策） |
| 错误处理 | 硬编码 | 硬编码 | 自我反思+重试 |
| 适用场景 | FAQ、简单客服 | ETL、固定流程 | 复杂、开放性任务 |

---

## 二、Agent 的核心架构模式

### 2.1 ReAct 模式（Reasoning + Acting）

这是当前最主流的 Agent 架构模式，也是 LangChain/LangGraph 的核心思想。

**核心循环：**

```
思考(Thought) → 行动(Action) → 观察(Observation) → 思考 → 行动 → ... → 最终回答

         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    ┌──────────┐    ┌──────────┐    ┌──────────┐│   ┌──────────┐
    │  感知     │ →  │  思考     │ →  │  行动    ││ → │  观察     │
    │ Perceive │    │  Think   │    │  Act     ││   │ Observe  │
    │          │    │          │    │          ││   │          │
    │ 接收用户  │    │ LLM推理  │    │ 调用工具  ││   │ 获取结果  │
    │ 输入/环境 │    │ 决定下一步│    │ 或回答   ││   │ 反馈给LLM │
    └──────────┘    └──────────┘    └──────────┘│   └────┬─────┘
                                                │        │
                                                └────────┘
                                            (循环直到任务完成)
```

**真实企业场景：金融风控审批**

一个贷款审批 Agent 收到申请后：
1. **思考**：这个申请需要查征信、查收入、查负债率
2. **行动**：调用征信查询工具
3. **观察**：征信评分 720，良好
4. **思考**：征信没问题，接下来查收入证明
5. **行动**：调用收入验证工具
6. **观察**：月收入 3 万，负债率 25%
7. **思考**：各项指标都达标，可以批准
8. **最终回答**：审批通过，建议额度 50 万

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class LoanReviewState(TypedDict):
    applicant_id: str
    credit_score: int | None
    income: float | None
    debt_ratio: float | None
    decision: str | None
    messages: Annotated[list, add]

def check_credit(state: LoanReviewState) -> dict:
    """调用征信查询工具"""
    # 实际项目中这里调用外部 API
    score = query_credit_api(state["applicant_id"])
    return {"credit_score": score, "messages": [f"征信评分: {score}"]}

def check_income(state: LoanReviewState) -> dict:
    """调用收入验证工具"""
    income, debt_ratio = query_income_api(state["applicant_id"])
    return {
        "income": income,
        "debt_ratio": debt_ratio,
        "messages": [f"月收入: {income}, 负债率: {debt_ratio}"]
    }

def make_decision(state: LoanReviewState) -> dict:
    """LLM 综合判断"""
    if (state["credit_score"] and state["credit_score"] >= 650
        and state["debt_ratio"] and state["debt_ratio"] < 0.5):
        return {"decision": "approved", "messages": ["审批通过"]}
    return {"decision": "rejected", "messages": ["审批拒绝"]}

# 构建审批流程图
graph = StateGraph(LoanReviewState)
graph.add_node("check_credit", check_credit)
graph.add_node("check_income", check_income)
graph.add_node("make_decision", make_decision)

graph.add_edge(START, "check_credit")
graph.add_edge("check_credit", "check_income")
graph.add_edge("check_income", "make_decision")
graph.add_edge("make_decision", END)

app = graph.compile()
```

### 2.2 Plan-and-Execute 模式

**核心思想**：先制定完整计划，再逐步执行。适合复杂、多步骤任务。

**现实类比**：就像项目经理接到一个大需求，先写 PRD 拆解任务，再分配给团队成员逐个执行。

**真实企业场景：ERP 系统数据迁移**

用户说："把旧 ERP 系统的客户数据迁移到新系统，要做数据清洗和格式转换"

Agent 的工作方式：
1. **规划阶段**：
   - 步骤1：连接旧系统数据库，导出客户表
   - 步骤2：数据清洗（去重、补全、格式标准化）
   - 步骤3：字段映射（旧系统字段 → 新系统字段）
   - 步骤4：数据验证
   - 步骤5：写入新系统
2. **执行阶段**：按计划逐步执行，每步完成后检查结果
3. **重规划**：如果某步失败（比如字段映射有冲突），重新调整计划

```python
class MigrationState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: Annotated[list, add]
    completed: bool

def planner(state: MigrationState) -> dict:
    """LLM 生成执行计划"""
    # LLM 根据任务描述生成步骤列表
    plan = llm_generate_plan(state["task"])
    return {"plan": plan, "current_step": 0}

def executor(state: MigrationState) -> dict:
    """执行当前步骤"""
    step = state["plan"][state["current_step"]]
    result = execute_step(step)
    return {
        "current_step": state["current_step"] + 1,
        "results": [result]
    }

def should_continue(state: MigrationState) -> str:
    """判断是否继续执行"""
    if state["current_step"] >= len(state["plan"]):
        return "done"
    return "continue"
```

### 2.3 Self-Reflection 模式（自我反思）

**核心思想**：Agent 执行后，自己检查结果质量，不满意就重做。

**现实类比**：就像一个资深工程师写完代码后会自己 Code Review，发现问题就改，而不是直接提交。

**真实企业场景：自动化代码生成与审查**

```python
class CodeGenState(TypedDict):
    requirement: str
    generated_code: str | None
    review_feedback: str | None
    iteration: int
    is_approved: bool

def generate_code(state: CodeGenState) -> dict:
    """LLM 生成代码"""
    code = llm_generate(state["requirement"], state.get("review_feedback"))
    return {"generated_code": code, "iteration": state["iteration"] + 1}

def review_code(state: CodeGenState) -> dict:
    """LLM 自我审查代码"""
    feedback = llm_review(state["generated_code"])
    is_good = "APPROVED" in feedback
    return {"review_feedback": feedback, "is_approved": is_good}

def should_retry(state: CodeGenState) -> str:
    if state["is_approved"] or state["iteration"] >= 3:
        return "finish"
    return "retry"
```

---

## 三、Agent 架构的三个层次

### 3.1 单 Agent 架构

一个 Agent 搞定所有事情。适合任务边界清晰、复杂度中等的场景。

**适用场景**：智能客服（单一领域）、代码助手、数据查询助手

### 3.2 多 Agent 协作架构

多个专业 Agent 各司其职，通过协调机制协作。

**适用场景**：复杂业务流程（如端到端的订单处理）、跨领域任务

**现实类比**：就像一个公司的不同部门——销售部接单、仓储部发货、财务部结算、客服部跟进。每个部门（Agent）有自己的专业能力，通过流程（Graph）协作。

### 3.3 层级 Agent 架构

有 Supervisor（主管）Agent 负责任务分配和协调，Worker Agent 负责执行。

**适用场景**：大型企业级系统，需要动态任务分配和负载均衡

```
         ┌─────────────┐
         │  Supervisor  │  ← 理解任务、分配工作、汇总结果
         │    Agent     │
         └──────┬───────┘
        ┌───────┼───────┐
        ▼       ▼       ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │ 查询   │ │ 分析   │ │ 报告   │  ← 各自专精一个领域
   │ Agent  │ │ Agent  │ │ Agent  │
   └────────┘ └────────┘ └────────┘
```

---

## 四、架构选型决策树

面试中经常被问到"什么场景用什么架构"，这里给一个决策框架：

```
任务是否需要多步骤推理？
├── 否 → 简单 LLM 调用即可，不需要 Agent
└── 是 → 是否需要调用外部工具？
    ├── 否 → Chain（LangChain LCEL）足够
    └── 是 → 是否需要动态决策调用哪个工具？
        ├── 否 → 固定 Pipeline
        └── 是 → Agent
            ├── 单一领域 → 单 Agent（ReAct）
            ├── 多领域协作 → 多 Agent（Supervisor/Swarm）
            └── 需要持久化/长时间运行 → LangGraph StateGraph
```

---

## 五、关键概念速查表

| 概念 | 解释 | 企业场景举例 |
|------|------|-------------|
| Tool Calling | Agent 调用外部工具获取信息或执行操作 | 查数据库、调 API、发邮件 |
| State | Agent 在执行过程中维护的上下文信息 | 当前审批进度、已收集的信息 |
| Checkpoint | 状态的持久化快照，支持恢复和回溯 | 长流程中断后从断点恢复 |
| Human-in-the-Loop | 在关键节点暂停，等待人工确认 | 大额转账需要主管审批 |
| Streaming | 实时流式输出 Agent 的思考和行动过程 | 用户看到 Agent 正在处理的实时状态 |
| Middleware | LangChain 1.x 新概念，拦截和修改 Agent 循环 | 统一日志记录、权限校验 |
| **Agentic Workflow** | 以 Agent 为中心的自动化工作流，替代传统 DAG | 订单处理全自动化，Agent 自主决策流程分支 |
| **Multi-Agent Orchestration** | 多个专业 Agent 通过协调机制协作完成复杂任务 | 跨部门项目审批，销售 Agent、财务 Agent、法务 Agent 各司其职 |
| **Thought Tree of Thoughts** | Agent 生成多个思考路径，探索最优解 | 复杂问题分析，Agent 从不同角度推理后选择最佳方案 |
| **Reflection Loop** | Agent 执行后自我反思并优化后续行动 | 代码生成后，Agent 自我审查代码质量并优化 |
| **Reasoning Tokens** | LLM 专门用于推理的 Token，提升思考深度 | 复杂数学计算、逻辑推理场景 |
| **Tool Use Planning** | Agent 提前规划工具调用顺序和参数 | 数据查询任务，Agent 先查 schema 再构造 SQL |
| **Agent Memory System** | 分层记忆架构，包括短期记忆、长期记忆、语义记忆 | 记住用户偏好、历史对话上下文、业务规则 |
| **Guardrails** | 安全防护机制，确保 Agent 行为符合规范 | 防止 Prompt 注入、限制敏感操作、合规检查 |
| **Observability for Agents** | Agent 执行的全链路可观测性 | 追踪 Agent 的思考、工具调用、决策过程，用于调试和审计 |
| **Agent as a Service (AaaS)** | 将 Agent 能力封装为服务供其他系统调用 | 企业内部 Agent 服务平台，各业务线按需调用 |

---

## 六、本章面试要点

### 基础面试题

1. **什么是 AI Agent？和传统 ChatBot 的本质区别是什么？**
   → Agent 具备自主决策和工具调用能力，流程不是硬编码的

2. **ReAct 模式的核心循环是什么？为什么它有效？**
   → Thought-Action-Observation 循环，模拟人类"想-做-看-再想"的认知过程

3. **什么时候该用 Agent，什么时候不该用？**
   → 用决策树判断：需要动态决策 + 工具调用 → Agent；固定流程 → Pipeline

4. **单 Agent vs 多 Agent，如何选型？**
   → 单一领域用单 Agent；跨领域协作用多 Agent；大规模系统用层级架构

5. **LangGraph 相比 LangChain 的 AgentExecutor 有什么优势？**
   → 状态持久化、条件路由、人工介入、可视化调试、生产级可靠性

### 进阶面试题

6. **Agentic Workflow 和传统 DAG 工作流有什么区别？**
   → 传统 DAG 是静态预定义的，Agentic Workflow 在运行时由 LLM 动态决策流程分支，更灵活但也更复杂

7. **Multi-Agent 协作有哪些常见架构模式？**
   → 主管-工人模式（Supervisor-Worker）、对等协作模式（Peer-to-Peer）、层级协作模式（Hierarchical）、Swarm 模式（群智能）

8. **什么是 Tree of Thoughts（ToT）推理？相比直接回答有什么优势？**
   → ToT 通过生成多个思考路径、评估每个路径的可行性、探索最优解，适合复杂问题分析，能提高答案准确性和可靠性

9. **Reflection Loop（反思循环）在实际项目中如何应用？**
   → 代码生成后自我审查、回答质量评估、错误自动修正、策略优化迭代

10. **Agent 的记忆系统应该如何设计分层？**
    → 短期记忆（滑动窗口最近对话）、工作记忆（当前任务上下文）、长期记忆（语义检索历史）、知识库记忆（企业私有知识）

### 企业级实战面试题

11. **如何设计 Agent 的 Guardrails 安全防护机制？**
    → 输入过滤（防 Prompt 注入）、工具权限控制（最小权限原则）、输出验证（合规检查）、行为监控（异常检测）、人工兜底（关键节点审批）

12. **Agent 的可观测性方案包含哪些内容？**
    → 全链路追踪（思考、工具调用、决策）、结构化日志、性能指标（延迟、成功率、成本）、可视化调试界面、审计日志

13. **在生产环境中，如何确保 Agent 的可靠性？**
    → 重试机制、熔断降级、多模型备份、状态持久化（Checkpoint）、错误恢复策略、监控告警

14. **Agent 成本优化有哪些方法？**
    → 模型路由（根据任务复杂度选择合适模型）、Prompt 优化（减少 Token 消耗）、结果缓存（重复查询直接返回）、批量处理、轻量级 Agent（简单任务用小模型）

15. **如何评估 Agent 的性能？**
    → 任务成功率、答案质量（准确度、相关性）、用户满意度、成本效益、响应延迟、可扩展性

### 架构设计面试题

16. **请设计一个企业级客服 Agent 系统的架构。**
    → 接入层（多端适配）、API 层（接口、限流、认证）、编排层（LangGraph 状态图）、服务层（LLM Gateway、RAG 引擎、记忆管理）、数据层（向量库、关系库、缓存）、运维层（监控、日志、安全）

17. **如何实现 Agent 的灰度发布和 A/B 测试？**
    → Prompt 版本控制、配置热更新、流量切分（按用户比例/特征）、指标对比（成功率、满意度、成本）、一键回滚

18. **Multi-Agent 协作时如何处理冲突和协调？**
    → 明确职责边界、统一状态协议、仲裁机制（Supervisor 决策）、优先级策略、投票机制、超时处理

19. **如何将 Agent 能力封装为服务供其他系统调用（AaaS）？**
    → RESTful/GraphQL API、SDK、事件驱动（消息队列）、身份认证、权限控制、计量计费、SLA 保障

20. **Agent 系统如何与企业现有系统（ERP、CRM、OA）集成？**
    → API 对接、数据库连接、RPA 集成、事件订阅、数据同步、身份打通、权限映射

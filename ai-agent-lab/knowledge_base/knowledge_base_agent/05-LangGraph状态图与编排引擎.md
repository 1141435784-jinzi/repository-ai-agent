# 03 - LangGraph 状态图与编排引擎

> LangGraph 是企业级 Agent 的编排核心。如果说 LangChain 是 Agent 的"积木块"，LangGraph 就是把积木搭成大厦的"建筑图纸"。

---

## 一、为什么需要 LangGraph？

### 1.1 LangChain 的 LCEL 不够用吗？

LCEL 适合**线性流程**：A → B → C。但企业级 Agent 需要：
- **条件分支**：根据结果走不同路径
- **循环**：ReAct 的 think-act-observe 循环
- **状态持久化**：长时间运行的任务需要保存进度
- **人工介入**：关键节点暂停等待审批
- **错误恢复**：从失败点恢复，而不是从头开始

**现实类比**：LCEL 像一条流水线——物料从头走到尾。LangGraph 像一个完整的工厂——有流水线、有质检站、有返工通道、有仓库（状态存储）、有主管审批环节。

### 1.2 LangGraph 的核心思想

LangGraph 把 Agent 的执行流程建模为一个**有向图（Directed Graph）**：
- **节点（Node）**：执行具体操作的函数（调用 LLM、执行工具、处理数据）
- **边（Edge）**：节点之间的连接，定义执行顺序
- **条件边（Conditional Edge）**：根据状态动态决定下一步走哪个节点
- **状态（State）**：在整个图中流转的共享数据

```
        ┌──────────┐
        │  START   │
        └────┬─────┘
             ▼
        ┌──────────┐
        │  Agent   │ ← LLM 思考：需要调用工具吗？
        │  (LLM)   │
        └────┬─────┘
             │
        ┌────┴────┐  条件边
        ▼         ▼
   ┌────────┐  ┌──────┐
   │ Tools  │  │ END  │  ← 不需要工具，直接回答
   └────┬───┘  └──────┘
        │
        └──────→ 回到 Agent（循环）
```

---

## 二、StateGraph 核心 API 详解

### 2.1 定义状态（State）

状态是 LangGraph 的灵魂。它是一个 TypedDict，在所有节点之间共享。

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import MessagesState

# 方式一：自定义状态
class OrderProcessState(TypedDict):
    """订单处理流程的状态"""
    order_id: str                              # 订单号
    customer_id: str                           # 客户 ID
    order_status: str | None                   # 订单状态
    payment_verified: bool                     # 支付是否验证
    inventory_checked: bool                    # 库存是否检查
    shipping_arranged: bool                    # 物流是否安排
    messages: Annotated[list, add]             # 消息列表（追加模式）
    error: str | None                          # 错误信息

# 方式二：使用内置的 MessagesState（适合对话场景）
# MessagesState 自带 messages 字段，自动处理消息追加
```

**关键概念：Reducer（归约器）**

`Annotated[list, add]` 中的 `add` 就是 Reducer。它定义了当多个节点都更新同一个字段时，如何合并。

```python
# 没有 Reducer：后写入的覆盖先写入的
messages: list  # 节点 A 写 ["hello"]，节点 B 写 ["world"] → 结果是 ["world"]

# 有 Reducer（add）：追加而不是覆盖
messages: Annotated[list, add]  # 节点 A 写 ["hello"]，节点 B 写 ["world"] → 结果是 ["hello", "world"]
```

**现实类比**：Reducer 就像会议记录。没有 Reducer，每个人发言后只保留最后一个人说的话；有了 Reducer（add），所有人的发言都会被记录下来。

### 2.2 定义节点（Node）

节点就是普通的 Python 函数，接收 State，返回要更新的字段。

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

llm = init_chat_model("openai:gpt-4o")

def verify_payment(state: OrderProcessState) -> dict:
    """验证支付状态"""
    order_id = state["order_id"]
    # 调用支付系统 API
    is_paid = payment_service.check(order_id)
    return {
        "payment_verified": is_paid,
        "messages": [f"支付验证{'通过' if is_paid else '未通过'}"]
    }

def check_inventory(state: OrderProcessState) -> dict:
    """检查库存"""
    order_id = state["order_id"]
    in_stock = inventory_service.check(order_id)
    return {
        "inventory_checked": in_stock,
        "messages": [f"库存{'充足' if in_stock else '不足'}"]
    }

def arrange_shipping(state: OrderProcessState) -> dict:
    """安排物流"""
    tracking_no = shipping_service.create(state["order_id"])
    return {
        "shipping_arranged": True,
        "messages": [f"物流已安排，运单号: {tracking_no}"]
    }

def handle_error(state: OrderProcessState) -> dict:
    """处理异常"""
    return {
        "messages": [f"订单处理异常: {state.get('error', '未知错误')}"],
        "order_status": "error"
    }
```

### 2.3 构建图（Graph）

```python
from langgraph.graph import StateGraph, START, END

# 创建图
graph = StateGraph(OrderProcessState)

# 添加节点
graph.add_node("verify_payment", verify_payment)
graph.add_node("check_inventory", check_inventory)
graph.add_node("arrange_shipping", arrange_shipping)
graph.add_node("handle_error", handle_error)

# 添加边
graph.add_edge(START, "verify_payment")

# 条件边：支付验证后，根据结果走不同路径
def after_payment(state: OrderProcessState) -> str:
    if state["payment_verified"]:
        return "check_inventory"
    return "handle_error"

graph.add_conditional_edges("verify_payment", after_payment)

# 条件边：库存检查后
def after_inventory(state: OrderProcessState) -> str:
    if state["inventory_checked"]:
        return "arrange_shipping"
    return "handle_error"

graph.add_conditional_edges("check_inventory", after_inventory)

# 终止边
graph.add_edge("arrange_shipping", END)
graph.add_edge("handle_error", END)

# 编译
app = graph.compile()

# 运行
result = app.invoke({
    "order_id": "ORD-12345",
    "customer_id": "CUST-001",
    "payment_verified": False,
    "inventory_checked": False,
    "shipping_arranged": False,
    "messages": [],
    "error": None,
})
```

### 2.4 可视化图结构

```python
# 生成 Mermaid 图（可在 Markdown 中渲染）
print(app.get_graph().draw_mermaid())

# 生成 PNG 图片
app.get_graph().draw_mermaid_png(output_file_path="order_flow.png")
```

---

## 三、条件路由：Agent 的"大脑决策"

### 3.1 基本条件路由

```python
def route_by_intent(state: CustomerServiceState) -> str:
    """根据用户意图路由到不同处理节点"""
    intent = state["detected_intent"]
    match intent:
        case "refund":
            return "refund_handler"
        case "exchange":
            return "exchange_handler"
        case "complaint":
            return "complaint_handler"
        case "inquiry":
            return "inquiry_handler"
        case _:
            return "general_handler"

graph.add_conditional_edges(
    "intent_detector",
    route_by_intent,
    # 显式声明所有可能的目标节点（可选但推荐）
    {
        "refund_handler": "refund_handler",
        "exchange_handler": "exchange_handler",
        "complaint_handler": "complaint_handler",
        "inquiry_handler": "inquiry_handler",
        "general_handler": "general_handler",
    }
)
```

### 3.2 LLM 驱动的动态路由

```python
from langchain_core.messages import AIMessage

def should_continue(state: AgentState) -> str:
    """根据 LLM 的输出决定是否继续调用工具"""
    last_message = state["messages"][-1]

    # 如果 LLM 返回了工具调用请求
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 否则直接结束
    return "end"

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tool_executor", "end": END}
)
```

---

## 四、Checkpoint：状态持久化与恢复

### 4.1 为什么需要 Checkpoint？

**现实场景**：一个保险理赔 Agent 正在处理一个复杂案件，已经完成了资料审核和定损评估，正准备做最终审批。这时候：
- 服务器重启了 → 没有 Checkpoint，一切从头开始
- 有 Checkpoint → 从"最终审批"这一步恢复，之前的工作不会丢失

**Checkpoint 解决的核心问题**：
1. **故障恢复**：从断点恢复，不丢失进度
2. **Human-in-the-Loop**：暂停等待人工审批，审批后继续
3. **时间旅行**：回溯到任意历史状态，用于调试和审计
4. **多会话管理**：每个用户/会话有独立的状态

### 4.2 使用 Checkpoint

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# 开发环境：内存存储
checkpointer = MemorySaver()

# 测试环境：SQLite
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 生产环境：PostgreSQL
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_db"
)

# 编译时传入 checkpointer
app = graph.compile(checkpointer=checkpointer)

# 运行时指定 thread_id（会话标识）
config = {"configurable": {"thread_id": "user-session-001"}}

# 第一次调用
result1 = app.invoke(
    {"messages": [HumanMessage(content="帮我查一下理赔进度")]},
    config=config
)

# 第二次调用（同一个 thread_id，自动恢复上下文）
result2 = app.invoke(
    {"messages": [HumanMessage(content="那赔付金额是多少？")]},
    config=config
)
# Agent 记得之前的对话，知道你在问哪个理赔案件
```

### 4.3 时间旅行：回溯历史状态

```python
# 获取所有历史状态
history = list(app.get_state_history(config))

for state in history:
    print(f"步骤: {state.metadata.get('step', 'N/A')}")
    print(f"节点: {state.metadata.get('source', 'N/A')}")
    print(f"时间: {state.created_at}")
    print("---")

# 回溯到特定状态
target_state = history[3]  # 回到第 4 个状态
app.update_state(config, target_state.values)
```

---

## 五、Human-in-the-Loop：人工介入

### 5.1 为什么需要人工介入？

**企业合规要求**：
- 金融交易超过一定金额，必须人工审批
- 医疗诊断建议，必须医生确认
- 合同条款修改，必须法务审核

### 5.2 实现方式

```python
# 方式一：interrupt_before —— 在执行工具前暂停
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_transfer"]  # 在转账节点前暂停
)

# 运行到转账节点前会自动暂停
result = app.invoke(input_data, config)
# result 包含当前状态，前端展示给审批人

# 审批人确认后，继续执行
# 方式 A：直接继续（approve）
app.invoke(None, config)  # 传 None 表示继续

# 方式 B：修改状态后继续（比如调整转账金额）
app.update_state(config, {"transfer_amount": 50000})
app.invoke(None, config)

# 方式 C：拒绝（跳转到拒绝节点）
app.update_state(config, {"approved": False})
app.invoke(None, config)
```

### 5.3 完整示例：大额转账审批

```python
class TransferState(TypedDict):
    sender: str
    receiver: str
    amount: float
    approved: bool | None
    messages: Annotated[list, add]

def prepare_transfer(state: TransferState) -> dict:
    """准备转账信息"""
    return {
        "messages": [
            f"准备转账: {state['sender']} → {state['receiver']}, "
            f"金额: ¥{state['amount']:,.2f}"
        ]
    }

def execute_transfer(state: TransferState) -> dict:
    """执行转账（需要人工审批后才能到达这里）"""
    if not state.get("approved"):
        return {"messages": ["转账被拒绝"]}
    # 调用银行 API 执行转账
    return {"messages": ["转账成功"]}

graph = StateGraph(TransferState)
graph.add_node("prepare", prepare_transfer)
graph.add_node("execute", execute_transfer)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "execute")
graph.add_edge("execute", END)

# 在 execute 节点前暂停，等待人工审批
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"]
)
```

---

## 六、子图（Subgraph）：模块化编排

### 6.1 为什么需要子图？

**现实类比**：一个大型软件项目不会把所有代码写在一个文件里，而是拆分成模块。同样，复杂的 Agent 流程也应该拆分成子图，每个子图负责一个独立的业务模块。

### 6.2 子图示例

```python
# 子图：客户身份验证
def build_auth_subgraph():
    class AuthState(TypedDict):
        customer_id: str
        is_verified: bool
        auth_method: str | None

    def verify_identity(state: AuthState) -> dict:
        # 调用身份验证服务
        return {"is_verified": True, "auth_method": "SMS"}

    auth_graph = StateGraph(AuthState)
    auth_graph.add_node("verify", verify_identity)
    auth_graph.add_edge(START, "verify")
    auth_graph.add_edge("verify", END)
    return auth_graph.compile()

# 主图：使用子图
main_graph = StateGraph(MainState)
main_graph.add_node("authenticate", build_auth_subgraph())
main_graph.add_node("process_request", process_request)
main_graph.add_edge(START, "authenticate")
main_graph.add_edge("authenticate", "process_request")
main_graph.add_edge("process_request", END)
```

---

## 七、流式输出（Streaming）

### 7.1 企业场景中的流式需求

用户不想等 Agent 跑完所有步骤才看到结果。他们想看到：
- Agent 正在思考什么
- 当前执行到哪一步
- 实时的中间结果

### 7.2 流式 API

```python
# 流式输出所有事件
async for event in app.astream_events(input_data, config, version="v2"):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        # LLM 正在生成文本
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)

    elif kind == "on_tool_start":
        # 开始调用工具
        print(f"\n🔧 调用工具: {event['name']}")

    elif kind == "on_tool_end":
        # 工具调用完成
        print(f"✅ 工具结果: {event['data'].get('output', '')[:100]}")

# 按节点流式输出
async for chunk in app.astream(input_data, config, stream_mode="updates"):
    for node_name, node_output in chunk.items():
        print(f"[{node_name}] {node_output}")
```

---

## 八、LangGraph 2026 最新特性补充

### 8.1 LangGraph Cloud (企业级部署方案)

LangGraph Cloud 是 LangChain 官方提供的企业级 Agent 部署平台，解决了生产环境中的诸多问题：

```python
# LangGraph Cloud 部署配置
from langgraph_sdk import get_client

# 连接到 LangGraph Cloud
client = get_client(url="https://api.langchain.com")

# 部署 Agent
await client.assistants.create(
    graph=app,  # 你的 LangGraph 应用
    assistant_id="my-assistant",
    config={
        "model": "gpt-4o",
        "temperature": 0.7
    }
)

# 调用 Agent（支持多人并发、自动扩缩容）
thread = await client.threads.create()
await client.runs.create(thread["thread_id"], "my-assistant", input=...)
```

**核心价值**：
- 自动扩缩容，应对流量峰值
- 多租户隔离，安全可靠
- 内置监控和告警
- 版本管理和灰度发布
- 高可用架构设计

### 8.2 State Schema 演进与验证

```python
from pydantic import BaseModel, Field
from typing import Literal

# 使用 Pydantic 定义强类型状态
class OrderState(BaseModel):
    order_id: str = Field(..., description="订单号")
    status: Literal["pending", "paid", "shipped", "completed", "cancelled"] = "pending"
    items: list[str] = Field(default_factory=list, description="商品列表")
    total: float = Field(0.0, ge=0, description="总金额")

# 编译时自动验证状态
graph = StateGraph(OrderState)
# ... 定义节点和边 ...
app = graph.compile()
```

### 8.3 错误处理与重试策略

```python
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint import CheckpointAt

# 定义重试装饰器
def with_retry(max_retries=3, backoff_factor=2):
    def decorator(func):
        async def wrapper(state):
            for i in range(max_retries):
                try:
                    return await func(state)
                except Exception as e:
                    if i == max_retries - 1:
                        raise
                    await asyncio.sleep(backoff_factor ** i)
        return wrapper
    return decorator

@with_retry(max_retries=3)
def unreliable_tool_call(state):
    # 可能失败的工具调用
    result = external_api.call()
    return {"result": result}

# 错误处理节点
def handle_error(state):
    error = state.get("error")
    return {
        "messages": [f"处理失败: {error}，已记录日志"],
        "status": "error"
    }
```

### 8.4 并行节点执行

```python
from langgraph.graph import ParallelNode

# 并行执行多个节点
def check_parallel(state):
    # 三个任务并行执行
    tasks = [
        check_inventory(state),
        check_credit(state),
        check_shipping(state)
    ]
    # 等待所有任务完成
    results = asyncio.gather(*tasks)
    # 合并结果
    return {**results[0], **results[1], **results[2]}

graph.add_node("parallel_check", check_parallel)
```

### 8.5 LangGraph 与 LangSmith 深度集成

```python
# LangSmith 自动追踪
from langsmith import traceable

@traceable(name="agent_execution")
async def run_agent(input_data):
    return await app.ainvoke(input_data, config={"configurable": {"thread_id": "test"}})

# 自定义追踪标签
config = {
    "configurable": {
        "thread_id": "user-123",
        "user_id": "user-123",
        "tenant_id": "acme-corp"
    },
    "metadata": {
        "project": "customer-service",
        "version": "v2.1.0"
    }
}
```

---

## 九、本章面试要点

### 基础面试题

1. **LangGraph 的核心抽象是什么？**
   → StateGraph（状态图）= 节点（操作）+ 边（流转）+ 状态（共享数据）+ Checkpoint（持久化）

2. **Reducer 是什么？为什么需要它？**
   → 定义多个节点更新同一字段时的合并策略。没有 Reducer 会覆盖，有 Reducer（如 add）会追加

3. **Checkpoint 解决了什么问题？生产环境用什么存储？**
   → 故障恢复、Human-in-the-Loop、时间旅行、多会话管理。生产用 PostgreSQL

4. **如何实现 Human-in-the-Loop？**
   → interrupt_before/interrupt_after 暂停 + update_state 修改 + invoke(None) 继续

5. **子图的价值是什么？**
   → 模块化、可复用、独立测试、团队并行开发

6. **LangGraph 和 Airflow/Prefect 等工作流引擎有什么区别？**
   → LangGraph 专为 LLM Agent 设计，支持 LLM 驱动的动态路由、对话状态管理、流式输出；传统工作流引擎是静态 DAG，不支持运行时动态决策

### 进阶面试题

7. **LangGraph Cloud 解决了哪些企业级问题？**
   → 自动扩缩容、多租户隔离、监控告警、版本管理、高可用、灰度发布、安全认证

8. **如何使用 Pydantic 进行 State Schema 验证？有什么好处？**
   → 类型安全、数据验证、自动文档生成、IDE 支持、运行时校验，减少状态相关的 bug

9. **LangGraph 中如何设计可靠的错误处理和重试策略？**
   → 装饰器模式封装重试逻辑、指数退避、最大重试次数限制、错误状态节点、降级策略、告警机制

10. **如何实现 LangGraph 中的并行节点执行？**
    → asyncio.gather 并发执行多个异步任务、ParallelNode 工具、结果合并策略、超时控制

11. **astream 和 astream_events 有什么区别？分别适用于什么场景？**
    → astream 按节点输出更新，适合展示执行进度；astream_events 输出所有事件（LLM token、工具调用等），适合实时展示 Agent 思考过程和调试

12. **如何设计可扩展的 State 结构？**
    → 分层设计（核心字段 + 扩展字段）、使用 TypedDict/Pydantic、预留扩展字段、合理的 Reducer 策略、避免过深的嵌套

### 企业级实战面试题

13. **在生产环境中，如何监控和调试 LangGraph Agent？**
    → LangSmith 集成追踪、结构化日志、自定义 metrics、可视化图执行、状态历史回溯、性能分析（延迟、Token 使用）

14. **LangGraph 如何实现多租户隔离？**
    → thread_id 按租户隔离、Checkpoint 按租户分表/分库、配置按租户隔离、资源配额限制、权限控制

15. **如何实现 LangGraph Agent 的灰度发布和 A/B 测试？**
    → 版本化部署、流量切分（按用户/比例）、指标收集对比、快速回滚、Prompt 版本管理

16. **LangGraph 在高并发场景下如何优化性能？**
    → 异步 I/O、连接池、结果缓存、批量处理、轻量级状态、合理的 Checkpoint 频率、水平扩展

17. **如何保证 LangGraph 状态数据的一致性？**
    → 事务性 Checkpoint、乐观锁、状态版本号、幂等设计、补偿机制、回滚能力

### 架构设计面试题

18. **请设计一个基于 LangGraph 的企业级客服系统架构。**
    → 接入层（多端）→ API 层（FastAPI）→ 编排层（LangGraph 状态图：意图识别→RAG→工具调用→回答）→ 服务层（LLM Gateway、RAG 引擎、记忆管理）→ 数据层（PostgreSQL Checkpoint、Chrom 向量库、Redis 缓存）→ 运维层（监控、日志、安全）

19. **如何将 LangGraph 与企业现有微服务架构集成？**
    → API 网关统一接入、服务发现、消息队列解耦、事件驱动、统一认证授权、分布式追踪、数据一致性保障

20. **对于复杂的 Multi-Agent 系统，如何用 LangGraph 设计？**
    → 子图模块化（每个 Agent 一个子图）、Supervisor 协调子图、消息队列通信、共享状态管理、故障隔离、动态 Agent 注册发现

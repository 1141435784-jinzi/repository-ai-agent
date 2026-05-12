# 07 - Memory 记忆机制与状态管理

> 没有记忆的 Agent 就像一个金鱼——每次对话都是全新的开始。企业级 Agent 必须"记住"用户身份、业务上下文和历史偏好，才能提供连贯、个性化的服务。

---

## 一、为什么记忆如此重要？

### 1.1 现实场景：没有记忆的银行客服有多糟糕

想象你打电话给银行客服：

- **第一通电话**："我要查信用卡账单" → 客服帮你查了，发现有一笔异常扣款 ¥3,200
- **第二通电话**："上次那个异常扣款处理得怎么样了？" → 客服："您是哪位？什么扣款？请重新描述您的问题。"
- **第三通电话**："我要投诉！每次都要重复说一遍！" → 客服："请问您要投诉什么？"

你崩溃了。这就是**无状态 Agent** 的真实体验——每次对话都是独立的，不记得之前发生了什么。

**企业损失**：
- 客户满意度暴跌，NPS 评分下降
- 平均处理时长（AHT）翻倍，因为每次都要重新收集信息
- 客户流失率上升，因为体验太差

### 1.2 现实场景：没有记忆的 ERP 助手有多低效

某制造企业的 ERP 智能助手：

- **周一上午**：财务总监问"帮我看看上个月华东区的销售数据" → 助手查了，返回了详细报表
- **周一下午**：财务总监问"把刚才那个数据按产品线拆分一下" → 助手："您说的是哪个数据？请重新描述。"
- **周二上午**：财务总监问"上次那个华东区的数据，和去年同期对比一下" → 助手完全不记得

**没有记忆的代价**：
- 用户每次都要重复完整的查询条件
- 无法进行"渐进式分析"（先看总览，再下钻细节）
- 无法积累用户偏好（比如"这个用户总是关注华东区"）

### 1.3 记忆解决的核心问题

| 问题 | 没有记忆 | 有记忆 |
|------|----------|--------|
| 多轮对话 | 每轮都是独立的，无法理解"那个""上次" | 自动关联上下文，理解指代和省略 |
| 会话恢复 | 断线/重启后一切从零开始 | 从断点恢复，不丢失进度 |
| 个性化服务 | 对所有用户一视同仁 | 记住偏好、习惯，提供定制化响应 |
| 渐进式任务 | 无法在之前结果上继续 | 支持"接着上次的继续" |
| 成本控制 | 每次都要重新传完整上下文 | 智能裁剪，只传必要信息 |

---

## 二、Agent 记忆的三个层次

就像人类的记忆系统分为工作记忆、短期记忆和长期记忆，Agent 的记忆也有三个层次，各自解决不同的问题。

### 2.1 三层记忆对比

| 维度 | 工作记忆（Working Memory） | 短期记忆（Short-term Memory） | 长期记忆（Long-term Memory） |
|------|---------------------------|-------------------------------|------------------------------|
| **人类类比** | 你正在进行的对话内容 | 昨天开会讨论了什么 | 你知道张三喜欢喝咖啡 |
| **Agent 类比** | 当前对话的消息列表 | 上一次会话的摘要和状态 | 用户画像、偏好、历史知识 |
| **技术实现** | `MessagesState` + `add_messages` | Checkpoint + 会话摘要 | 向量数据库 + KV Store |
| **生命周期** | 单次会话（分钟~小时） | 跨会话（天~周） | 永久（月~年） |
| **存储位置** | 内存（State 对象） | PostgreSQL（Checkpoint） | ChromaDB / pgvector / Store |
| **访问速度** | 纳秒级（内存直接读取） | 毫秒级（数据库查询） | 10~100ms（向量检索） |
| **容量限制** | 受 LLM 上下文窗口限制 | 受数据库存储限制 | 几乎无限 |
| **核心价值** | 维持对话连贯性 | 实现会话恢复和延续 | 提供个性化和知识积累 |

### 2.2 三层记忆的协作关系

```
用户发起对话
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ① 加载长期记忆：从向量 DB 召回用户画像和相关知识       │
│  ② 加载短期记忆：从 Checkpoint 恢复上次会话状态        │
│  ③ 进入工作记忆：开始当前对话，消息追加到 messages      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ 长期记忆  │───→│ 短期记忆  │───→│ 工作记忆  │       │
│  │ (向量DB) │    │(Checkpoint)│   │(messages)│       │
│  └──────────┘    └──────────┘    └──────────┘       │
│                                                      │
│  对话结束后：                                         │
│  ④ 工作记忆 → Checkpoint 持久化（短期记忆）            │
│  ⑤ 提取关键信息 → 存入向量 DB（长期记忆）              │
└─────────────────────────────────────────────────────┘
```

**现实类比**：你是一个资深客户经理。
- **长期记忆**：你翻开 CRM 系统，看到"张三，VIP 客户，偏好保守型理财，上次买了 50 万国债"
- **短期记忆**：你看到昨天的通话记录，"张三咨询了大额存单利率，还没做决定"
- **工作记忆**：张三现在打电话进来了，你们正在讨论大额存单的具体条款

---

## 三、工作记忆：对话上下文管理

工作记忆是最基础也是最重要的记忆层。它管理的是**当前对话中的消息列表**——用户说了什么、Agent 回了什么、调用了哪些工具。

### 3.1 基本消息管理：MessagesState 与 add_messages

LangGraph 提供了 `MessagesState` 作为对话场景的标准状态定义，内置了 `messages` 字段和 `add_messages` reducer。

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = init_chat_model("openai:gpt-4o")

def chatbot(state: MessagesState) -> dict:
    """基础聊天节点：接收消息列表，返回 LLM 响应。

    MessagesState 自带 messages 字段，使用 add_messages reducer，
    新消息会自动追加到列表末尾，而不是覆盖。
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建最简单的对话图
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 编译时传入 checkpointer，启用状态持久化
app = graph.compile(checkpointer=MemorySaver())

# 同一个 thread_id 下的对话共享上下文
config = {"configurable": {"thread_id": "session-001"}}

# 第一轮对话
app.invoke({"messages": [HumanMessage(content="我叫张三，是华东区的销售经理")]}, config)

# 第二轮对话——Agent 记得你是谁
result = app.invoke({"messages": [HumanMessage(content="我叫什么？负责哪个区域？")]}, config)
print(result["messages"][-1].content)
# 输出：你叫张三，负责华东区。
```

**关键机制：add_messages reducer**

`add_messages` 不是简单的列表追加（`list.extend`），它有智能合并逻辑：

```python
from langgraph.graph import add_messages

# 1. 基本追加：新消息追加到末尾
messages = add_messages(
    [HumanMessage(content="你好", id="msg-1")],
    [AIMessage(content="你好！有什么可以帮你？", id="msg-2")]
)
# 结果：[HumanMessage("你好"), AIMessage("你好！有什么可以帮你？")]

# 2. 按 ID 更新：如果新消息的 id 和已有消息相同，则替换
messages = add_messages(
    [AIMessage(content="旧回答", id="msg-2")],
    [AIMessage(content="更新后的回答", id="msg-2")]
)
# 结果：[AIMessage("更新后的回答")]  ← 替换而非追加

# 3. 删除消息：使用 RemoveMessage
from langchain_core.messages import RemoveMessage
messages = add_messages(
    [HumanMessage(content="你好", id="msg-1"), AIMessage(content="你好！", id="msg-2")],
    [RemoveMessage(id="msg-1")]
)
# 结果：[AIMessage("你好！")]  ← msg-1 被删除
```

### 3.2 消息窗口管理：防止 Token 爆炸

**企业痛点**：客服系统中，一个复杂工单可能需要 30+ 轮对话。如果把所有消息都传给 LLM：
- Token 消耗飙升（GPT-4o 每百万 Token $2.5 输入 / $10 输出）
- 可能超过模型上下文窗口（即使 128K 也有上限）
- LLM 在超长上下文中容易"迷失"，回答质量下降

**方案一：滑动窗口——只保留最近 N 条消息**

```python
def chatbot_with_window(state: MessagesState) -> dict:
    """滑动窗口策略：只保留最近的消息。

    优点：实现简单，Token 消耗可控
    缺点：早期的重要信息会丢失
    适用：闲聊、简单问答等不依赖早期上下文的场景
    """
    messages = state["messages"]

    # 始终保留系统消息 + 最近 20 条对话消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    windowed = system_msgs + non_system_msgs[-20:]

    response = llm.invoke(windowed)
    return {"messages": [response]}
```

**方案二：基于 Token 的智能裁剪——trim_messages**

```python
from langchain_core.messages import trim_messages

def chatbot_with_trimming(state: MessagesState) -> dict:
    """基于 Token 数的智能裁剪。

    trim_messages 会精确计算每条消息的 Token 数，
    从最早的消息开始裁剪，直到总 Token 数在限制内。

    优点：精确控制 Token 消耗，不会超过预算
    缺点：仍然会丢失早期信息
    适用：成本敏感的生产环境
    """
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,           # 最大 Token 数（为 LLM 回复留空间）
        strategy="last",           # 保留最近的消息（从末尾开始保留）
        token_counter=llm,         # 用 LLM 的 tokenizer 精确计算
        include_system=True,       # 始终保留系统消息（不计入裁剪）
        start_on="human",          # 确保裁剪后第一条是用户消息（避免孤立的 AI 回复）
        allow_partial=False,       # 不允许截断单条消息（要么保留完整消息，要么整条删除）
    )
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

**trim_messages 参数详解**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `max_tokens` | 裁剪后的最大 Token 数 | 模型上下文窗口的 50%~70% |
| `strategy` | `"last"` 保留最近的，`"first"` 保留最早的 | `"last"` |
| `token_counter` | Token 计算器，传入 LLM 实例或自定义函数 | `llm` |
| `include_system` | 是否始终保留 SystemMessage | `True` |
| `start_on` | 裁剪后第一条消息的类型 | `"human"` |
| `allow_partial` | 是否允许截断单条消息 | `False` |

### 3.3 消息摘要：压缩旧消息，保留关键信息

**现实类比**：就像会议纪要。你不需要记住会议的每一句话，但需要记住关键决策、待办事项和重要数据。摘要策略是滑动窗口和 Token 裁剪的升级版——不是简单丢弃旧消息，而是把它们压缩成摘要。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage

class SummarizableState(TypedDict):
    """支持摘要的对话状态"""
    messages: Annotated[list, add_messages]
    summary: str                               # 历史对话的摘要

def should_summarize(state: SummarizableState) -> str:
    """判断是否需要生成摘要：消息超过 15 条时触发。"""
    if len(state["messages"]) > 15:
        return "summarize"
    return "respond"

def summarize_conversation(state: SummarizableState) -> dict:
    """生成对话摘要，替代旧消息。

    策略：保留最近 4 条消息（2 轮对话），其余压缩为摘要。
    这样既保留了最近的上下文，又大幅减少了 Token 消耗。
    """
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # 构造摘要提示
    summary_prompt = f"""请总结以下对话的关键信息，生成简洁的摘要。包括：
1. 用户的核心需求和意图
2. 已经完成的操作和结果
3. 待处理的事项
4. 重要的上下文信息（用户身份、订单号、关键数据等）

{"已有摘要（请在此基础上更新）：" + existing_summary if existing_summary else ""}

需要总结的对话：
"""
    # 只总结旧消息（保留最近 4 条）
    old_messages = messages[:-4]
    for msg in old_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        summary_prompt += f"\n{role}: {msg.content}"

    summary_response = llm.invoke(summary_prompt)

    # 删除旧消息，用摘要替代
    delete_msgs = [RemoveMessage(id=m.id) for m in old_messages]

    return {
        "summary": summary_response.content,
        "messages": delete_msgs,  # add_messages reducer 会处理删除
    }

def respond(state: SummarizableState) -> dict:
    """生成回复，将摘要注入系统消息。"""
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        # 将摘要作为系统消息的一部分
        system_msg = SystemMessage(
            content=f"你是一个智能助手。以下是之前对话的摘要：\n{summary}"
        )
        all_messages = [system_msg] + messages
    else:
        all_messages = messages

    response = llm.invoke(all_messages)
    return {"messages": [response]}

# 构建带摘要的对话图
graph = StateGraph(SummarizableState)
graph.add_node("respond", respond)
graph.add_node("summarize", summarize_conversation)

graph.add_conditional_edges(START, should_summarize, {
    "summarize": "summarize",
    "respond": "respond",
})
graph.add_edge("summarize", "respond")
graph.add_edge("respond", END)
```

### 3.4 自定义 State 字段：超越纯消息的业务上下文

真实的企业 Agent 不只需要消息列表。它还需要携带业务上下文——客户信息、订单状态、权限级别等。这些信息需要在节点之间传递，但不一定要放进 LLM 的 prompt 里。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class EnterpriseAgentState(TypedDict):
    """企业级 Agent 状态：消息 + 业务上下文。

    设计原则：凡是需要在节点之间传递的数据，都放进 State。
    """
    # ===== 对话相关 =====
    messages: Annotated[list, add_messages]    # 对话历史（必须）

    # ===== 用户上下文 =====
    user_id: str                               # 用户 ID
    user_role: str                             # 用户角色（admin/manager/staff）
    tenant_id: str                             # 租户 ID（多租户隔离）

    # ===== 业务上下文 =====
    intent: str | None                         # 意图路由结果
    rag_context: str | None                    # RAG 检索到的上下文
    rag_sources: list[str]                     # RAG 数据来源
    current_task: dict | None                  # 当前正在处理的任务

    # ===== 记忆相关 =====
    session_summary: str | None                # 上次会话摘要
    recalled_memories: list[str]               # 从长期记忆召回的信息
    customer_profile: dict | None              # 客户画像

    # ===== 流程控制 =====
    requires_approval: bool                    # 是否需要人工审批
    error: str | None                          # 错误信息
```

**为什么不把所有信息都塞进 messages？**

```python
# ❌ 反模式：把业务数据塞进消息
def bad_approach(state: MessagesState) -> dict:
    return {"messages": [
        SystemMessage(content=f"用户ID: {user_id}, 角色: admin, 租户: tenant-001, ...")
    ]}
    # 问题：每轮对话都重复传这些信息，浪费 Token；且不方便在节点间传递结构化数据

# ✅ 正确做法：用 State 字段传递结构化数据
def good_approach(state: EnterpriseAgentState) -> dict:
    # 业务数据通过 State 字段传递，不占用 LLM Token
    user_role = state["user_role"]
    if user_role != "admin":
        return {"error": "权限不足"}
    # 只在需要时才把关键信息注入 prompt
    return {"messages": [llm.invoke(state["messages"])]}
```

---

## 四、LangGraph Checkpointer 深度解析

Checkpointer 是 LangGraph 的状态持久化引擎。它在每个节点执行完毕后，自动将整个 State 快照保存到存储后端。这是实现会话恢复、时间旅行、Human-in-the-Loop 的基础。

### 4.1 Checkpointer 是什么？内部工作原理

**现实类比**：Checkpointer 就像游戏的自动存档系统。每过一个关卡（节点），游戏自动保存进度。如果你挂了（服务器崩溃），可以从最近的存档点恢复，而不是从头开始。

**内部工作流程**：

```
用户输入 → [START]
              │
              ▼
         ┌─────────┐
         │  Node A  │ ← 执行完毕
         └────┬─────┘
              │
              ▼
    ┌──────────────────┐
    │ Checkpointer.put │ ← 自动保存 State 快照
    │ (checkpoint_id,  │    包含：所有 State 字段的当前值
    │  thread_id,      │    标记：哪个节点刚执行完
    │  state_snapshot) │    元数据：时间戳、步骤号
    └────────┬─────────┘
              │
              ▼
         ┌─────────┐
         │  Node B  │ ← 执行完毕
         └────┬─────┘
              │
              ▼
    ┌──────────────────┐
    │ Checkpointer.put │ ← 再次保存快照
    └────────┬─────────┘
              │
              ▼
           [END]
```

**关键概念**：

- **checkpoint_id**：每个快照的唯一标识（UUID），每次保存都生成新的
- **thread_id**：会话标识，同一个 thread_id 下的所有快照形成一条时间线
- **parent_checkpoint_id**：指向上一个快照，形成链表结构，支持时间旅行
- **channel_values**：State 中所有字段的序列化值
- **channel_versions**：每个字段的版本号，用于增量更新

### 4.2 MemorySaver：内存存储（开发调试用）

```python
from langgraph.checkpoint.memory import MemorySaver

# 最简单的 Checkpointer：数据存在内存中
checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

# 特点：
# ✅ 零配置，开箱即用
# ✅ 速度最快（纯内存操作）
# ❌ 进程重启后数据丢失
# ❌ 不支持多进程/多实例共享
# 适用：本地开发、单元测试、快速原型验证
```

### 4.3 SqliteSaver / AsyncSqliteSaver：轻量持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# ===== 同步版（适合 CLI 工具、脚本） =====
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 需要手动初始化表结构（首次使用）
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(input_data, config)

# ===== 异步版（适合 FastAPI 等异步框架） =====
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    result = await app.ainvoke(input_data, config)

# 特点：
# ✅ 数据持久化到文件，重启不丢失
# ✅ 单文件部署，无需额外数据库服务
# ✅ 适合中小规模应用（单机部署）
# ❌ 不支持并发写入（SQLite 的限制）
# ❌ 不适合多实例部署（文件锁冲突）
# 适用：个人项目、小团队内部工具、测试环境
```

### 4.4 PostgresSaver / AsyncPostgresSaver：生产级持久化

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/agent_memory"

# ===== 同步版 =====
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 首次使用时自动创建表结构
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(input_data, config)

# ===== 异步版（生产环境推荐） =====
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
    result = await app.ainvoke(input_data, config)

# ===== 使用连接池（高并发场景） =====
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    min_size=5,       # 最小连接数
    max_size=20,      # 最大连接数
    max_idle=300,     # 空闲连接最大存活时间（秒）
)

async with pool.connection() as conn:
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

# 特点：
# ✅ 生产级可靠性（ACID 事务、WAL 日志、主从复制）
# ✅ 支持高并发读写
# ✅ 支持多实例共享（水平扩展）
# ✅ 丰富的运维工具（备份、监控、慢查询分析）
# ❌ 需要额外部署和维护 PostgreSQL
# 适用：生产环境、多实例部署、高并发场景
```

### 4.5 thread_id 机制：会话隔离

`thread_id` 是 LangGraph 实现多会话隔离的核心机制。每个 `thread_id` 对应一条独立的对话时间线。

```python
# 用户 A 的对话
config_a = {"configurable": {"thread_id": "user-alice-session-001"}}
app.invoke({"messages": [HumanMessage(content="我要查订单 ORD-001")]}, config_a)
app.invoke({"messages": [HumanMessage(content="这个订单的物流状态？")]}, config_a)
# Agent 知道"这个订单"是 ORD-001

# 用户 B 的对话（完全隔离）
config_b = {"configurable": {"thread_id": "user-bob-session-001"}}
app.invoke({"messages": [HumanMessage(content="帮我退货")]}, config_b)
# Agent 不知道 ORD-001，因为这是不同的 thread

# thread_id 命名最佳实践
thread_id_patterns = {
    # 按用户+日期：适合客服场景，每天一个新会话
    "daily_session": "user-{user_id}-{date}",        # user-alice-20260417

    # 按用户+任务：适合任务型 Agent，一个任务一个会话
    "task_session": "user-{user_id}-task-{task_id}",  # user-alice-task-refund-001

    # 按租户+用户+会话：适合多租户 SaaS
    "tenant_session": "t-{tenant_id}-u-{user_id}-s-{session_id}",
}
```

### 4.6 State 快照与时间旅行调试

时间旅行是 Checkpointer 最强大的能力之一——你可以回溯到 Agent 执行过程中的任意一个状态点，查看当时的完整 State，甚至从那个点重新执行。

```python
config = {"configurable": {"thread_id": "debug-session-001"}}

# 执行一次完整的对话
app.invoke({"messages": [HumanMessage(content="帮我分析上月销售数据")]}, config)
app.invoke({"messages": [HumanMessage(content="按区域拆分")]}, config)
app.invoke({"messages": [HumanMessage(content="导出为 Excel")]}, config)

# ===== 查看当前状态 =====
current_state = app.get_state(config)
print(f"当前消息数: {len(current_state.values['messages'])}")
print(f"下一步节点: {current_state.next}")  # 如果图还没结束，显示下一个要执行的节点

# ===== 遍历历史状态（时间旅行） =====
for state_snapshot in app.get_state_history(config):
    step = state_snapshot.metadata.get("step", "N/A")
    source = state_snapshot.metadata.get("source", "N/A")
    ts = state_snapshot.created_at
    msg_count = len(state_snapshot.values.get("messages", []))
    print(f"步骤 {step} | 来源: {source} | 时间: {ts} | 消息数: {msg_count}")

# ===== 回溯到特定状态 =====
# 场景：Agent 在第 3 步做了错误的工具调用，你想从第 2 步重新开始
history = list(app.get_state_history(config))
target_state = history[2]  # 回到第 3 个快照点

# 方式一：从该状态点继续执行（传入 checkpoint_id）
rollback_config = {
    "configurable": {
        "thread_id": "debug-session-001",
        "checkpoint_id": target_state.config["configurable"]["checkpoint_id"],
    }
}
# 从回溯点重新执行
app.invoke({"messages": [HumanMessage(content="换一种方式分析")]}, rollback_config)

# 方式二：修改状态后继续（比如修正错误的中间结果）
app.update_state(config, {"intent": "rag"})  # 修正意图判断
app.invoke(None, config)  # 从修正后的状态继续执行
```

**时间旅行的企业价值**：
- **调试**：Agent 给出了错误回答？回溯到每一步，看看哪个节点出了问题
- **审计**：监管要求查看 Agent 的完整决策过程？每一步都有快照
- **回滚**：Agent 执行了错误的操作？回到操作前的状态，重新执行

### 4.7 Checkpoint 生命周期与垃圾回收

在生产环境中，Checkpoint 数据会持续增长。一个活跃的客服系统，每天可能产生数十万个 Checkpoint。必须有清理策略。

```python
import asyncio
from datetime import datetime, timedelta

async def cleanup_old_checkpoints(
    db_uri: str,
    retention_days: int = 30,
    batch_size: int = 1000,
) -> int:
    """清理过期的 Checkpoint 数据。

    Args:
        db_uri: PostgreSQL 连接字符串
        retention_days: 保留天数，超过此天数的 Checkpoint 将被删除
        batch_size: 每批删除的记录数

    Returns:
        删除的 Checkpoint 总数
    """
    import asyncpg

    conn = await asyncpg.connect(db_uri)
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    total_deleted = 0

    try:
        while True:
            # 分批删除，避免长事务锁表
            deleted = await conn.execute("""
                DELETE FROM checkpoints
                WHERE created_at < $1
                AND ctid IN (
                    SELECT ctid FROM checkpoints
                    WHERE created_at < $1
                    LIMIT $2
                )
            """, cutoff, batch_size)

            count = int(deleted.split()[-1])
            total_deleted += count

            if count < batch_size:
                break

            await asyncio.sleep(0.1)  # 避免对数据库造成过大压力
    finally:
        await conn.close()

    return total_deleted

# 定时任务：每天凌晨清理 30 天前的 Checkpoint
# 可以用 APScheduler 或 Celery Beat 调度
```

**Checkpoint 存储容量估算**：

| 场景 | 每次对话轮次 | 每个 Checkpoint 大小 | 日活用户 | 日增量 |
|------|-------------|---------------------|---------|--------|
| 简单客服 | 5 轮 | ~2KB | 1,000 | ~10MB |
| 复杂工单 | 20 轮 | ~8KB | 5,000 | ~800MB |
| ERP 助手 | 30 轮 | ~15KB | 500 | ~225MB |

---

## 五、短期记忆：跨会话的上下文延续

短期记忆解决的核心问题是：**用户关闭浏览器后再回来，Agent 还能"接上话"**。这不是简单的消息回放，而是智能地恢复上下文。

### 5.1 基于 Checkpoint + thread_id 的会话恢复

最基本的跨会话恢复：使用同一个 `thread_id`，Checkpointer 会自动加载上次的完整 State。

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage

DB_URI = "postgresql://user:pass@localhost:5432/agent_memory"

async def demo_session_recovery():
    """演示跨会话恢复：用户今天和明天的对话无缝衔接。"""
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)

        # 用户 ID + 任务 ID 作为 thread_id，确保同一任务的对话连续
        config = {"configurable": {"thread_id": "user-zhangsan-task-sales-report"}}

        # ===== 周一上午：用户开始分析任务 =====
        await app.ainvoke(
            {"messages": [HumanMessage(content="帮我查上个月华东区的销售数据")]},
            config,
        )
        # Agent 返回了销售数据

        # ===== 周一下午：用户继续（同一个 thread_id） =====
        await app.ainvoke(
            {"messages": [HumanMessage(content="按产品线拆分一下")]},
            config,
        )
        # Agent 知道"拆分"的是华东区销售数据

        # ===== 周二上午：用户回来继续（服务器可能已经重启过） =====
        await app.ainvoke(
            {"messages": [HumanMessage(content="和去年同期对比呢？")]},
            config,
        )
        # Checkpointer 从 PostgreSQL 加载了完整的对话历史
        # Agent 知道要对比的是华东区按产品线拆分的销售数据
```

### 5.2 跨会话记忆传递：加载上次会话摘要

有时候你不想延续同一个 thread（比如客服场景，每次来电是新会话），但希望 Agent 知道上次聊了什么。这就需要**跨会话记忆传递**。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

class CustomerServiceState(TypedDict):
    """客服 Agent 状态：支持跨会话记忆传递。"""
    messages: Annotated[list, add_messages]
    customer_id: str
    session_summary: str | None       # 上一次会话的摘要
    customer_context: dict | None     # 客户上下文信息

async def load_previous_context(state: CustomerServiceState) -> dict:
    """会话开始时，加载上次会话摘要和客户上下文。

    这个节点在每次新会话开始时执行，从数据库加载：
    1. 上次会话的摘要（短期记忆）
    2. 客户的基本信息和最近工单（业务上下文）
    """
    customer_id = state["customer_id"]

    # 从数据库加载上次会话摘要
    last_summary = await db.get_latest_session_summary(customer_id)

    # 加载客户上下文
    context = await db.get_customer_context(customer_id)

    # 构造系统消息，注入上下文
    context_parts = [f"当前客户: {customer_id}"]

    if last_summary:
        context_parts.append(f"上次对话摘要: {last_summary}")

    if context:
        context_parts.append(f"客户等级: {context.get('level', '普通')}")
        context_parts.append(f"最近工单: {context.get('recent_tickets', '无')}")
        context_parts.append(f"历史投诉: {context.get('complaints', '无')}")
        context_parts.append(f"偏好: {context.get('preferences', '无特殊偏好')}")

    system_msg = SystemMessage(content="\n".join(context_parts))

    return {
        "session_summary": last_summary,
        "customer_context": context,
        "messages": [system_msg],
    }
```

### 5.3 会话摘要的生成与存储

每次会话结束时，自动生成摘要并存储，供下次会话使用。

```python
async def save_session_summary(state: CustomerServiceState) -> dict:
    """会话结束时，生成并保存会话摘要。

    摘要会存入数据库，下次该客户来电时自动加载。
    """
    customer_id = state["customer_id"]
    messages = state["messages"]

    # 让 LLM 生成结构化摘要
    summary_prompt = """请为以下客服对话生成结构化摘要，格式如下：
【客户诉求】一句话概括客户的核心需求
【处理结果】已完成的操作和结果
【待跟进】未解决的问题或需要后续跟进的事项
【客户情绪】满意/一般/不满
【关键信息】对话中提到的重要数据（订单号、金额等）

对话内容：
"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            summary_prompt += f"\n客户: {msg.content}"
        elif hasattr(msg, "content") and not isinstance(msg, SystemMessage):
            summary_prompt += f"\n客服: {msg.content}"

    summary = await llm.ainvoke(summary_prompt)

    # 存入数据库
    await db.save_session_summary(
        customer_id=customer_id,
        summary=summary.content,
        session_id=state.get("session_id"),
        created_at=datetime.utcnow(),
    )

    return {}
```

### 5.4 企业模式：客户上下文在会话开始时加载

将上述能力整合成完整的客服 Agent 图：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

def build_customer_service_graph() -> StateGraph:
    """构建带跨会话记忆的客服 Agent 图。

    流程：加载上下文 → Agent 对话（循环） → 保存摘要
    """
    graph = StateGraph(CustomerServiceState)

    graph.add_node("load_context", load_previous_context)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_executor)
    graph.add_node("save_summary", save_session_summary)

    # 会话开始 → 加载上下文
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "agent")

    # Agent 循环：思考 → 调用工具 → 继续思考
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "save_and_end": "save_summary",
    })
    graph.add_edge("tools", "agent")

    # 保存摘要 → 结束
    graph.add_edge("save_summary", END)

    return graph

# 编译并运行
graph = build_customer_service_graph()
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    # 每次来电创建新的 thread_id，但通过 load_context 加载历史
    config = {"configurable": {"thread_id": f"call-{customer_id}-{datetime.now():%Y%m%d%H%M}"}}
    result = await app.ainvoke({
        "messages": [HumanMessage(content="上次那个退款处理好了吗？")],
        "customer_id": "CUST-001",
    }, config)
    # Agent 通过 load_context 知道上次的退款详情，直接回答
```

---

## 六、长期记忆：用户画像与知识积累

长期记忆是 Agent 从"工具"进化为"伙伴"的关键。它让 Agent 能够：
- 记住用户的偏好和习惯（"张三总是要简洁的报表"）
- 积累业务知识（"华东区 Q1 销售通常会下降"）
- 提供个性化服务（"李四对价格敏感，推荐时优先考虑性价比"）

### 6.1 基于向量数据库的语义记忆

**为什么用向量数据库而不是关系型数据库？**

因为记忆的召回是**语义匹配**，不是精确查询。当用户说"帮我看看销售情况"，你需要召回的可能是"用户偏好简洁报表风格"和"用户关注华东区数据"——这些和"销售情况"没有关键词重叠，但语义相关。

```python
import chromadb
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

class SemanticMemory:
    """基于向量数据库的语义长期记忆。

    使用 ChromaDB 存储记忆，支持：
    - 按用户隔离（user_id 过滤）
    - 语义相似度召回
    - 记忆分类（偏好/习惯/事实/反馈）
    - TTL 过期清理
    """

    def __init__(self, persist_dir: str = "./memory_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def store(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        ttl_days: int | None = None,
    ) -> None:
        """存储一条记忆。

        Args:
            user_id: 用户 ID
            content: 记忆内容
            category: 分类（preference/habit/fact/feedback）
            ttl_days: 过期天数，None 表示永不过期
        """
        embedding = self.embeddings.embed_query(content)
        memory_id = f"{user_id}_{hash(content) & 0xFFFFFFFF}"
        expires_at = (
            (datetime.utcnow().timestamp() + ttl_days * 86400)
            if ttl_days
            else None
        )

        self.collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "category": category,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": str(expires_at) if expires_at else "never",
            }],
        )

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """根据语义相似度召回相关记忆。

        Args:
            user_id: 用户 ID
            query: 查询文本
            top_k: 返回的最大记忆数
            category: 可选的分类过滤

        Returns:
            召回的记忆列表，每条包含 content、category、score
        """
        embedding = self.embeddings.embed_query(query)

        # 构建过滤条件
        where_filter: dict = {"user_id": user_id}
        if category:
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"category": category},
                ]
            }

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # 过滤已过期的记忆
                if meta.get("expires_at") != "never":
                    if float(meta["expires_at"]) < datetime.utcnow().timestamp():
                        continue

                memories.append({
                    "content": doc,
                    "category": meta.get("category", "general"),
                    "score": 1 - dist,  # ChromaDB 返回的是距离，转换为相似度
                })

        return memories

    def cleanup_expired(self) -> int:
        """清理已过期的记忆。返回删除的记忆数量。"""
        now = datetime.utcnow().timestamp()
        # ChromaDB 不支持直接按条件删除，需要先查询再删除
        all_items = self.collection.get(
            where={"expires_at": {"$ne": "never"}},
        )

        expired_ids = []
        for item_id, meta in zip(all_items["ids"], all_items["metadatas"]):
            if float(meta["expires_at"]) < now:
                expired_ids.append(item_id)

        if expired_ids:
            self.collection.delete(ids=expired_ids)

        return len(expired_ids)


# 使用示例
memory = SemanticMemory()

# 存储不同类型的记忆
memory.store("user-001", "用户偏好简洁的报表风格，不喜欢太多图表", category="preference")
memory.store("user-001", "每周一上午查看上周销售数据", category="habit")
memory.store("user-001", "华东区域负责人，重点关注上海和杭州", category="fact")
memory.store("user-001", "上次对响应速度不满意，希望更快", category="feedback", ttl_days=90)

# 语义召回
results = memory.recall("user-001", "帮我看看销售数据")
for r in results:
    print(f"[{r['category']}] {r['content']} (相似度: {r['score']:.3f})")
# 输出：
# [habit] 每周一上午查看上周销售数据 (相似度: 0.892)
# [fact] 华东区域负责人，重点关注上海和杭州 (相似度: 0.847)
# [preference] 用户偏好简洁的报表风格，不喜欢太多图表 (相似度: 0.781)
```

### 6.2 LangGraph Store API：基于命名空间的 KV 存储

LangGraph 1.x 提供了内置的 Store API，适合存储结构化的用户偏好和配置信息。

```python
from langgraph.store.memory import InMemoryStore

# 创建 Store（生产环境可替换为持久化实现）
store = InMemoryStore()

# ===== 基于命名空间的层级存储 =====
# 命名空间是一个元组，类似文件系统的目录结构

# 存储用户偏好
store.put(
    namespace=("users", "user-001", "preferences"),
    key="report_style",
    value={"style": "简洁", "charts": False, "format": "表格", "language": "中文"},
)

store.put(
    namespace=("users", "user-001", "preferences"),
    key="notification",
    value={"email": True, "sms": False, "frequency": "daily"},
)

# 存储用户的常用查询
store.put(
    namespace=("users", "user-001", "saved_queries"),
    key="weekly_sales",
    value={"query": "华东区上周销售数据", "schedule": "每周一 9:00"},
)

# ===== 读取 =====
item = store.get(namespace=("users", "user-001", "preferences"), key="report_style")
print(item.value)  # {"style": "简洁", "charts": False, ...}

# ===== 列出命名空间下的所有项 =====
items = store.search(namespace=("users", "user-001", "preferences"))
for item in items:
    print(f"{item.key}: {item.value}")

# ===== 在 Agent 节点中使用 Store =====
# 编译时传入 store
app = graph.compile(checkpointer=checkpointer, store=store)

# 在节点函数中通过 config 访问 store
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

def agent_with_store(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """在 Agent 节点中访问 Store，读取用户偏好。"""
    user_id = config["configurable"].get("user_id", "anonymous")

    # 读取用户偏好
    prefs = store.get(namespace=("users", user_id, "preferences"), key="report_style")

    if prefs:
        style_hint = f"用户偏好: {prefs.value}"
    else:
        style_hint = "无特殊偏好"

    messages = state["messages"] + [SystemMessage(content=style_hint)]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

**Store vs 向量数据库：何时用哪个？**

| 维度 | Store（KV 存储） | 向量数据库 |
|------|------------------|-----------|
| 数据类型 | 结构化（JSON） | 非结构化文本 |
| 查询方式 | 精确键查找 | 语义相似度搜索 |
| 适用场景 | 用户偏好、配置、标签 | 对话记忆、知识片段、用户反馈 |
| 示例 | "张三偏好简洁报表" | "上次张三问了华东区 Q1 销售下降的原因" |

### 6.3 从对话中提取记忆：LLM 作为记忆提取器

长期记忆不是手动录入的，而是从对话中自动提取的。让 LLM 充当"记忆提取器"，识别值得长期记住的信息。

```python
from langchain_core.messages import HumanMessage, AIMessage

MEMORY_EXTRACTION_PROMPT = """你是一个记忆提取专家。请从以下对话中提取值得长期记住的用户信息。

提取规则：
1. 只提取关于用户本人的信息，不提取通用知识
2. 分类为以下类型之一：preference（偏好）、habit（习惯）、fact（事实）、feedback（反馈）
3. 每条记忆独立成句，简洁明了
4. 如果没有值得记住的信息，返回空列表

输出格式（JSON 数组）：
[
  {"content": "记忆内容", "category": "分类"},
  ...
]

对话内容：
{conversation}

已有记忆（避免重复）：
{existing_memories}
"""

async def extract_memories(
    messages: list,
    user_id: str,
    memory: SemanticMemory,
) -> list[dict]:
    """从对话中提取并存储新的长期记忆。

    Args:
        messages: 对话消息列表
        user_id: 用户 ID
        memory: 长期记忆存储实例

    Returns:
        新提取的记忆列表
    """
    # 格式化对话
    conversation = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation += f"助手: {msg.content}\n"

    # 获取已有记忆（用于去重）
    existing = memory.recall(user_id, conversation, top_k=10)
    existing_text = "\n".join(m["content"] for m in existing) if existing else "无"

    # 调用 LLM 提取
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        conversation=conversation,
        existing_memories=existing_text,
    )
    result = await llm.ainvoke(prompt)

    # 解析并存储
    import json
    try:
        new_memories = json.loads(result.content)
    except json.JSONDecodeError:
        return []

    for mem in new_memories:
        memory.store(
            user_id=user_id,
            content=mem["content"],
            category=mem.get("category", "general"),
        )

    return new_memories
```

### 6.4 在 Agent 回复前召回记忆

将长期记忆集成到 Agent 的处理流程中：在 Agent 回复前，先召回相关记忆，注入到上下文中。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

class MemoryAwareState(TypedDict):
    """带长期记忆感知的 Agent 状态。"""
    messages: Annotated[list, add_messages]
    user_id: str
    recalled_memories: list[str]

long_term_memory = SemanticMemory()

async def recall_before_response(state: MemoryAwareState) -> dict:
    """Agent 回复前，召回相关的长期记忆。

    策略：用用户最新的消息作为查询，从长期记忆中召回最相关的信息。
    召回的记忆会作为 SystemMessage 注入，帮助 Agent 提供个性化回答。
    """
    user_id = state["user_id"]

    # 找到最新的用户消息
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_msgs:
        return {"recalled_memories": []}

    query = human_msgs[-1].content
    memories = long_term_memory.recall(user_id, query, top_k=5)

    if memories:
        memory_text = "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)
        return {
            "recalled_memories": [m["content"] for m in memories],
            "messages": [SystemMessage(
                content=f"[长期记忆] 关于该用户的已知信息：\n{memory_text}\n"
                        f"请根据这些信息提供个性化的回答。"
            )],
        }

    return {"recalled_memories": []}

async def save_after_response(state: MemoryAwareState) -> dict:
    """Agent 回复后，提取并保存新的长期记忆。"""
    await extract_memories(state["messages"], state["user_id"], long_term_memory)
    return {}

# 构建带记忆的 Agent 图
graph = StateGraph(MemoryAwareState)
graph.add_node("recall", recall_before_response)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_executor)
graph.add_node("save", save_after_response)

graph.add_edge(START, "recall")
graph.add_edge("recall", "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "save": "save",
})
graph.add_edge("tools", "agent")
graph.add_edge("save", END)
```

### 6.5 记忆去重与 TTL 过期

长期记忆如果不管理，会出现大量重复和过时的信息。

```python
class MemoryManager:
    """记忆管理器：负责去重、过期清理、容量控制。"""

    def __init__(self, memory: SemanticMemory, max_memories_per_user: int = 500):
        self.memory = memory
        self.max_per_user = max_memories_per_user

    def deduplicate_before_store(
        self,
        user_id: str,
        new_content: str,
        similarity_threshold: float = 0.92,
    ) -> bool:
        """存储前去重：如果已有高度相似的记忆，则跳过。

        Args:
            user_id: 用户 ID
            new_content: 新记忆内容
            similarity_threshold: 相似度阈值，超过则认为重复

        Returns:
            True 表示是新记忆（已存储），False 表示重复（已跳过）
        """
        existing = self.memory.recall(user_id, new_content, top_k=3)

        for mem in existing:
            if mem["score"] >= similarity_threshold:
                # 已有高度相似的记忆，跳过
                return False

        # 没有重复，存储新记忆
        self.memory.store(user_id, new_content)
        return True

    def merge_similar_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.88,
    ) -> int:
        """合并相似记忆：将高度相似的多条记忆合并为一条。

        适合定期执行（比如每天凌晨），减少记忆冗余。

        Returns:
            合并的记忆对数
        """
        all_memories = self.memory.recall(user_id, "", top_k=100)
        merged_count = 0

        # 两两比较，找出相似对
        for i, mem_a in enumerate(all_memories):
            for mem_b in all_memories[i + 1:]:
                if mem_b["score"] >= similarity_threshold:
                    # 让 LLM 合并两条记忆
                    merge_prompt = f"""请将以下两条相似的用户记忆合并为一条更完整的记忆：
记忆1: {mem_a['content']}
记忆2: {mem_b['content']}
合并后的记忆（一句话）:"""
                    merged = llm.invoke(merge_prompt)

                    # 存储合并后的记忆（旧的会被 upsert 覆盖）
                    self.memory.store(user_id, merged.content, category=mem_a["category"])
                    merged_count += 1

        return merged_count
```

---

## 七、LangGraph 1.x State 设计模式

State 是 LangGraph 的灵魂。设计好 State，Agent 的数据流就清晰了；设计不好，节点之间的数据传递会变成一团乱麻。

### 7.1 TypedDict vs MessagesState

```python
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState, add_messages

# ===== 方式一：MessagesState（纯对话场景） =====
# 内置 messages 字段 + add_messages reducer
# 适合：简单聊天机器人、问答助手

class SimpleChat(MessagesState):
    """继承 MessagesState，自动获得 messages 字段。
    可以在此基础上添加额外字段。
    """
    user_id: str  # 额外字段

# ===== 方式二：TypedDict（复杂业务场景） =====
# 完全自定义，灵活度最高
# 适合：多步骤工作流、需要大量业务字段的场景

class OrderProcessingState(TypedDict):
    """订单处理 Agent 的状态。"""
    messages: Annotated[list, add_messages]     # 对话历史
    order_id: str                               # 订单号
    order_items: list[dict]                     # 订单商品列表
    total_amount: float                         # 订单总金额
    payment_status: str | None                  # 支付状态
    shipping_status: str | None                 # 物流状态
    approval_required: bool                     # 是否需要审批
    error: str | None                           # 错误信息

# ===== 选择建议 =====
# MessagesState：对话为主，业务字段少（< 5 个额外字段）
# TypedDict：业务逻辑复杂，需要大量结构化字段
```

### 7.2 Annotated Reducers：控制字段更新策略

Reducer 定义了当多个节点更新同一个字段时，如何合并值。这是 LangGraph State 最核心的概念之一。

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import add_messages

# ===== 内置 Reducer =====

class DemoState(TypedDict):
    # 1. add_messages：智能消息合并（支持追加、替换、删除）
    messages: Annotated[list, add_messages]

    # 2. operator.add：简单列表追加
    logs: Annotated[list, add]
    # 节点 A 返回 {"logs": ["step1"]}
    # 节点 B 返回 {"logs": ["step2"]}
    # 最终结果：["step1", "step2"]

    # 3. 无 Reducer：后写入覆盖先写入
    current_step: str
    # 节点 A 返回 {"current_step": "验证"}
    # 节点 B 返回 {"current_step": "处理"}
    # 最终结果："处理"（被覆盖）

# ===== 自定义 Reducer =====

def merge_dicts(existing: dict, new: dict) -> dict:
    """字典合并 Reducer：新值合并到已有字典中。"""
    merged = {**existing} if existing else {}
    merged.update(new)
    return merged

def keep_max(existing: float | None, new: float) -> float:
    """保留最大值 Reducer。"""
    if existing is None:
        return new
    return max(existing, new)

def append_unique(existing: list, new: list) -> list:
    """去重追加 Reducer：只追加不重复的元素。"""
    result = list(existing) if existing else []
    for item in new:
        if item not in result:
            result.append(item)
    return result

class AdvancedState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: Annotated[dict, merge_dicts]           # 字典合并
    max_confidence: Annotated[float | None, keep_max] # 保留最大值
    visited_nodes: Annotated[list, append_unique]     # 去重追加
```

### 7.3 State Channels：数据如何在节点间流转

理解 State 的数据流转机制，是设计复杂 Agent 的基础。

```
                    State（共享数据池）
                 ┌──────────────────────┐
                 │ messages: [...]       │
                 │ intent: "rag"        │
                 │ rag_context: "..."   │
                 │ error: None          │
                 └──────────────────────┘
                    ▲          │
          ┌─────────┘          └─────────┐
          │ 写入（返回 dict）              │ 读取（接收 state）
          │                              ▼
    ┌──────────┐                   ┌──────────┐
    │  Node A  │                   │  Node B  │
    │          │                   │          │
    │ return { │                   │ state[   │
    │  "intent"│                   │  "intent"│
    │  : "rag" │                   │ ]        │
    │ }        │                   │ → "rag"  │
    └──────────┘                   └──────────┘
```

**核心规则**：
1. **节点读取**：通过 `state["field_name"]` 读取任意字段
2. **节点写入**：返回一个 dict，只包含要更新的字段（不需要返回完整 State）
3. **Reducer 合并**：如果字段有 Reducer，新值和旧值通过 Reducer 合并；否则新值覆盖旧值
4. **未返回的字段不变**：节点不返回某个字段，该字段保持原值

```python
# 示例：节点只更新它关心的字段
def intent_router(state: EnterpriseAgentState) -> dict:
    """意图路由节点：只更新 intent 字段。"""
    last_msg = state["messages"][-1].content
    intent = classify_intent(last_msg)
    return {"intent": intent}  # 只返回 intent，其他字段不变

def rag_retriever(state: EnterpriseAgentState) -> dict:
    """RAG 检索节点：更新 rag_context 和 rag_sources。"""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {
        "rag_context": "\n".join(d.page_content for d in docs),
        "rag_sources": [d.metadata.get("source", "") for d in docs],
    }
    # messages、intent 等字段保持不变
```

### 7.4 企业级 AgentState 设计实战

设计一个完整的企业级 Agent State，需要考虑：对话管理、业务上下文、记忆系统、流程控制、可观测性。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from operator import add

class ProductionAgentState(TypedDict):
    """生产级 Agent 状态设计。

    设计原则：
    1. 对话字段用 add_messages reducer（智能合并）
    2. 日志字段用 add reducer（追加）
    3. 业务字段不用 reducer（覆盖更新）
    4. 字段命名清晰，避免歧义
    5. 可选字段用 | None 标注
    """

    # ========== 对话管理 ==========
    messages: Annotated[list, add_messages]     # 对话历史（核心）

    # ========== 用户与租户 ==========
    user_id: str                                # 用户标识
    tenant_id: str                              # 租户标识（多租户隔离）
    user_role: str                              # 用户角色（权限控制）

    # ========== 意图与路由 ==========
    intent: str | None                          # 当前意图（rag/tool/chat）
    sub_intent: str | None                      # 子意图（refund/inquiry/complaint）

    # ========== RAG 上下文 ==========
    rag_context: str | None                     # 检索到的文档内容
    rag_sources: list[str]                      # 文档来源列表
    rag_confidence: float | None                # 检索置信度

    # ========== 记忆系统 ==========
    session_summary: str | None                 # 上次会话摘要
    recalled_memories: list[str]                # 召回的长期记忆
    conversation_summary: str | None            # 当前对话的滚动摘要

    # ========== 业务上下文 ==========
    current_task: dict | None                   # 当前任务详情
    task_result: dict | None                    # 任务执行结果

    # ========== 流程控制 ==========
    requires_approval: bool                     # 是否需要人工审批
    retry_count: int                            # 重试次数
    error: str | None                           # 错误信息

    # ========== 可观测性 ==========
    execution_logs: Annotated[list, add]        # 执行日志（追加模式）
    node_timings: Annotated[dict, merge_dicts]  # 各节点耗时
```

### 7.5 State 不可变性与 Reducer 模式

LangGraph 的 State 遵循**不可变性原则**：节点不应该直接修改传入的 state 对象，而是返回一个新的 dict 来描述变更。

```python
# ❌ 反模式：直接修改 state（可能导致不可预测的行为）
def bad_node(state: ProductionAgentState) -> dict:
    state["messages"].append(new_message)  # 直接修改！
    state["intent"] = "rag"                # 直接修改！
    return state                           # 返回整个 state

# ✅ 正确做法：返回变更的字段
def good_node(state: ProductionAgentState) -> dict:
    return {
        "messages": [new_message],  # add_messages reducer 会处理追加
        "intent": "rag",            # 覆盖更新
    }

# ✅ 正确做法：需要基于旧值计算新值时
def increment_retry(state: ProductionAgentState) -> dict:
    return {
        "retry_count": state["retry_count"] + 1,  # 读旧值，返回新值
        "execution_logs": [f"重试第 {state['retry_count'] + 1} 次"],
    }
```

---

## 八、生产环境记忆架构设计

### 8.1 整体架构：三层记忆统一视图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent 应用层                              │
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐          │
│  │  工作记忆   │     │  短期记忆   │     │  长期记忆   │          │
│  │            │     │            │     │            │          │
│  │ messages   │     │ Checkpoint │     │ 向量检索    │          │
│  │ State 字段  │     │ 会话摘要    │     │ KV Store   │          │
│  │ 消息裁剪    │     │ 上下文加载  │     │ 记忆提取    │          │
│  └─────┬──────┘     └─────┬──────┘     └─────┬──────┘          │
│        │                  │                   │                 │
└────────┼──────────────────┼───────────────────┼─────────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
    ┌─────────┐      ┌───────────┐      ┌────────────┐
    │  内存    │      │PostgreSQL │      │ PostgreSQL  │
    │ (State) │      │           │      │ + pgvector  │
    │         │      │checkpoints│      │             │
    │ 纳秒级  │      │ sessions  │      │ memories    │
    │ 访问    │      │ summaries │      │ embeddings  │
    └─────────┘      │           │      │ user_prefs  │
                     │ 毫秒级    │      │             │
                     │ 访问      │      │ 10~100ms    │
                     └───────────┘      │ 访问        │
                                        └────────────┘
```

### 8.2 PostgreSQL 统一存储方案

在生产环境中，推荐使用 PostgreSQL 作为统一存储后端：Checkpoint 用原生表，长期记忆用 pgvector 扩展。这样只需要维护一个数据库。

```python
"""生产环境记忆架构：PostgreSQL 统一存储。

表结构：
- checkpoints: LangGraph 自动管理（Checkpoint 数据）
- checkpoint_blobs: LangGraph 自动管理（大对象存储）
- checkpoint_writes: LangGraph 自动管理（写入记录）
- session_summaries: 会话摘要（自定义）
- long_term_memories: 长期记忆 + pgvector（自定义）
- user_preferences: 用户偏好（自定义）
"""

# ===== 数据库初始化 SQL =====
INIT_SQL = """
-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 会话摘要表
CREATE TABLE IF NOT EXISTS session_summaries (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(200) NOT NULL,
    summary TEXT NOT NULL,
    sentiment VARCHAR(20),           -- 客户情绪：positive/neutral/negative
    unresolved_issues TEXT,          -- 未解决的问题
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id)
);
CREATE INDEX idx_session_customer ON session_summaries(customer_id, created_at DESC);

-- 长期记忆表（带向量索引）
CREATE TABLE IF NOT EXISTS long_term_memories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'general',
    embedding vector(1536),          -- OpenAI text-embedding-3-small 维度
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,            -- NULL 表示永不过期
    access_count INT DEFAULT 0,      -- 访问次数（用于热度排序）
    last_accessed_at TIMESTAMP
);
CREATE INDEX idx_memory_user ON long_term_memories(user_id);
CREATE INDEX idx_memory_embedding ON long_term_memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_memory_expiry ON long_term_memories(expires_at)
    WHERE expires_at IS NOT NULL;

-- 用户偏好表
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(100) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, key)
);
"""
```

```python
import asyncpg
from langchain_openai import OpenAIEmbeddings

class ProductionMemoryStore:
    """生产级记忆存储：基于 PostgreSQL + pgvector。

    统一管理会话摘要、长期记忆、用户偏好。
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    async def save_session_summary(
        self,
        customer_id: str,
        session_id: str,
        summary: str,
        sentiment: str = "neutral",
        unresolved: str | None = None,
    ) -> None:
        """保存会话摘要。"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO session_summaries
                    (customer_id, session_id, summary, sentiment, unresolved_issues)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    sentiment = EXCLUDED.sentiment,
                    unresolved_issues = EXCLUDED.unresolved_issues
            """, customer_id, session_id, summary, sentiment, unresolved)

    async def get_recent_summaries(
        self,
        customer_id: str,
        limit: int = 3,
    ) -> list[dict]:
        """获取客户最近的会话摘要。"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT summary, sentiment, unresolved_issues, created_at
                FROM session_summaries
                WHERE customer_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, customer_id, limit)
            return [dict(row) for row in rows]

    async def store_memory(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        ttl_days: int | None = None,
    ) -> None:
        """存储长期记忆（带向量嵌入）。"""
        embedding = self.embeddings.embed_query(content)

        async with self.pool.acquire() as conn:
            expires_at = (
                f"NOW() + INTERVAL '{ttl_days} days'" if ttl_days else "NULL"
            )
            await conn.execute(f"""
                INSERT INTO long_term_memories
                    (user_id, content, category, embedding, expires_at)
                VALUES ($1, $2, $3, $4::vector, {expires_at})
            """, user_id, content, category, str(embedding))

    async def recall_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """语义召回长期记忆。"""
        embedding = self.embeddings.embed_query(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT content, category,
                       1 - (embedding <=> $3::vector) AS similarity
                FROM long_term_memories
                WHERE user_id = $1
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> $3::vector
                LIMIT $2
            """, user_id, top_k, str(embedding))

            # 更新访问计数
            if rows:
                ids = [row["content"] for row in rows]
                await conn.execute("""
                    UPDATE long_term_memories
                    SET access_count = access_count + 1,
                        last_accessed_at = NOW()
                    WHERE user_id = $1 AND content = ANY($2)
                """, user_id, ids)

            return [dict(row) for row in rows]
```

### 8.3 记忆管理策略总览

| 策略 | 适用层 | 实现方式 | 触发条件 | 效果 |
|------|--------|----------|----------|------|
| 滑动窗口 | 工作记忆 | 只保留最近 N 条消息 | 每次 LLM 调用前 | 简单粗暴，可能丢失重要信息 |
| Token 限制 | 工作记忆 | `trim_messages` 按 Token 裁剪 | 每次 LLM 调用前 | 精确控制成本，推荐 |
| 摘要压缩 | 工作记忆 | LLM 生成摘要替代旧消息 | 消息数 > 阈值 | 保留关键信息，额外 LLM 调用 |
| 会话摘要 | 短期记忆 | 会话结束时生成结构化摘要 | 会话结束 | 跨会话延续的基础 |
| 语义去重 | 长期记忆 | 存储前检查相似度 > 阈值 | 每次存储前 | 避免记忆冗余 |
| TTL 过期 | 长期记忆 | 设置过期时间，定时清理 | 定时任务 | 清理时效性信息 |
| 热度衰减 | 长期记忆 | 长期未访问的记忆降低优先级 | 召回排序时 | 保持记忆的时效性 |
| Checkpoint GC | 短期记忆 | 定时删除 N 天前的 Checkpoint | 定时任务 | 控制存储增长 |

### 8.4 性能考量

```python
"""记忆系统性能优化要点。"""

# 1. Checkpoint 大小控制
# 问题：State 中存储大量数据会导致 Checkpoint 体积膨胀
# 方案：只在 State 中存储必要的数据，大对象存外部存储

# ❌ 反模式：把完整文档存在 State 中
class BadState(TypedDict):
    messages: Annotated[list, add_messages]
    full_document: str  # 可能有几十 KB，每个 Checkpoint 都会保存一份

# ✅ 正确做法：State 中只存引用，需要时再查
class GoodState(TypedDict):
    messages: Annotated[list, add_messages]
    document_id: str | None  # 只存 ID，需要时从数据库/对象存储查

# 2. 向量检索延迟优化
# - 使用 IVFFlat 索引（PostgreSQL pgvector）或 HNSW 索引
# - 控制向量维度：text-embedding-3-small (1536维) vs text-embedding-3-large (3072维)
# - 预热索引：服务启动时执行一次空查询

# 3. Checkpoint 查询优化
# - 为 thread_id 建立索引（LangGraph 自动创建）
# - 定期 VACUUM 清理已删除的 Checkpoint
# - 使用连接池避免频繁建立连接

# 4. 批量操作
# - 记忆提取和存储使用批量 embedding（减少 API 调用）
# - Checkpoint 清理使用分批删除（避免长事务）
```

### 8.5 多租户记忆隔离

在 SaaS 场景中，不同租户的记忆必须严格隔离。

```python
class MultiTenantMemoryStore(ProductionMemoryStore):
    """多租户记忆存储：在所有操作中强制租户隔离。"""

    async def recall_memories(
        self,
        user_id: str,
        query: str,
        tenant_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """带租户隔离的记忆召回。"""
        embedding = self.embeddings.embed_query(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT content, category,
                       1 - (embedding <=> $4::vector) AS similarity
                FROM long_term_memories
                WHERE user_id = $1
                  AND tenant_id = $2          -- 租户隔离
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> $4::vector
                LIMIT $3
            """, user_id, tenant_id, top_k, str(embedding))
            return [dict(row) for row in rows]

# thread_id 也要包含租户信息
config = {
    "configurable": {
        "thread_id": f"tenant-{tenant_id}-user-{user_id}-session-{session_id}",
    }
}
```

**多租户隔离的三道防线**：
1. **thread_id 前缀**：`tenant-{id}-` 确保 Checkpoint 不会跨租户
2. **SQL WHERE 条件**：所有查询都带 `tenant_id` 过滤
3. **行级安全策略（RLS）**：PostgreSQL 原生支持，即使代码有 bug 也不会泄露

---

## 九、企业级真实案例

### 9.1 案例一：智能客服系统——多轮对话 + 会话恢复

**业务场景**：某电商平台的智能客服，需要处理退款、物流查询、投诉等多种场景。客户可能中途断线，再次来电时希望无缝衔接。

**记忆需求**：
- 工作记忆：维持当前对话的多轮上下文
- 短期记忆：客户断线后恢复对话进度
- 长期记忆：记住客户的历史投诉和偏好

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class CustomerServiceState(TypedDict):
    """电商客服 Agent 状态。"""
    messages: Annotated[list, add_messages]
    customer_id: str
    intent: str | None                     # refund/logistics/complaint/inquiry
    order_id: str | None                   # 当前讨论的订单
    session_summary: str | None            # 上次会话摘要
    recalled_memories: list[str]           # 长期记忆
    resolution_status: str | None          # pending/resolved/escalated

async def load_customer_context(state: CustomerServiceState) -> dict:
    """会话开始：加载客户的完整上下文。

    这是记忆系统的"入口节点"，整合三层记忆：
    1. 从 PostgreSQL 加载上次会话摘要（短期记忆）
    2. 从向量数据库召回客户画像（长期记忆）
    3. 从业务系统加载客户信息（外部数据）
    """
    customer_id = state["customer_id"]

    # 短期记忆：上次会话摘要
    summaries = await memory_store.get_recent_summaries(customer_id, limit=2)
    summary_text = ""
    if summaries:
        latest = summaries[0]
        summary_text = f"上次对话({latest['created_at']:%m-%d})：{latest['summary']}"
        if latest.get("unresolved_issues"):
            summary_text += f"\n未解决问题：{latest['unresolved_issues']}"

    # 长期记忆：客户画像和历史偏好
    last_msg = state["messages"][-1].content if state["messages"] else ""
    memories = await memory_store.recall_memories(customer_id, last_msg, top_k=5)
    memory_text = "\n".join(f"- {m['content']}" for m in memories) if memories else "无"

    # 构造系统提示
    system_content = f"""你是电商平台的智能客服助手。

客户信息：
- 客户ID: {customer_id}
{f"- {summary_text}" if summary_text else "- 首次来访"}

客户画像（长期记忆）：
{memory_text}

服务准则：
1. 如果客户提到"上次""之前"，参考上次对话摘要回答
2. 根据客户画像调整沟通风格
3. 复杂问题无法解决时，主动升级到人工客服
"""
    return {
        "session_summary": summary_text,
        "recalled_memories": [m["content"] for m in memories],
        "messages": [SystemMessage(content=system_content)],
    }

async def save_and_extract(state: CustomerServiceState) -> dict:
    """会话结束：保存摘要 + 提取长期记忆。"""
    customer_id = state["customer_id"]

    # 保存会话摘要（短期记忆）
    summary = await generate_session_summary(state["messages"])
    await memory_store.save_session_summary(
        customer_id=customer_id,
        session_id=f"session-{customer_id}-{datetime.now():%Y%m%d%H%M}",
        summary=summary["summary"],
        sentiment=summary["sentiment"],
        unresolved=summary.get("unresolved"),
    )

    # 提取长期记忆
    await extract_memories(state["messages"], customer_id, long_term_memory)

    return {"resolution_status": "resolved"}

# 完整的客服图
graph = StateGraph(CustomerServiceState)
graph.add_node("load_context", load_customer_context)
graph.add_node("agent", customer_service_agent)
graph.add_node("tools", tool_executor)
graph.add_node("save", save_and_extract)

graph.add_edge(START, "load_context")
graph.add_edge("load_context", "agent")
graph.add_conditional_edges("agent", route_agent, {
    "tools": "tools",
    "end": "save",
})
graph.add_edge("tools", "agent")
graph.add_edge("save", END)
```

**效果对比**：

| 指标 | 无记忆 | 有记忆 |
|------|--------|--------|
| 平均处理时长 | 8 分钟 | 4 分钟 |
| 首次解决率 | 45% | 72% |
| 客户满意度 | 3.2/5 | 4.5/5 |
| 重复信息收集 | 每次都要 | 自动加载 |

### 9.2 案例二：金融理财顾问——长期用户偏好追踪

**业务场景**：某银行的智能理财顾问，需要根据客户的风险偏好、投资历史、人生阶段提供个性化的理财建议。

**记忆需求**：
- 长期记忆是核心：记住客户的风险偏好、资产配置、投资目标
- 记忆需要随时间演化：客户的偏好会变（结婚、生子、退休）

```python
class FinancialAdvisorState(TypedDict):
    """金融理财顾问 Agent 状态。"""
    messages: Annotated[list, add_messages]
    customer_id: str
    risk_profile: dict | None              # 风险画像
    investment_history: list[dict]         # 投资历史
    life_events: list[str]                 # 人生事件（影响投资策略）
    recalled_memories: list[str]
    recommendation: dict | None            # 理财建议

async def build_customer_profile(state: FinancialAdvisorState) -> dict:
    """构建客户的完整投资画像。

    整合长期记忆中的偏好信息和业务系统中的交易数据。
    """
    customer_id = state["customer_id"]

    # 从长期记忆召回投资偏好
    memories = await memory_store.recall_memories(
        customer_id, "投资偏好 风险 理财目标", top_k=10
    )

    # 从业务系统加载交易历史
    history = await trading_system.get_history(customer_id, months=12)

    # 构建风险画像
    profile_prompt = f"""根据以下信息，生成客户的投资风险画像（JSON格式）：

客户记忆：
{chr(10).join(m['content'] for m in memories)}

最近12个月交易记录：
{format_trading_history(history)}

输出格式：
{{"risk_level": "保守/稳健/积极/激进",
  "preferred_products": ["产品类型"],
  "investment_horizon": "短期/中期/长期",
  "key_concerns": ["关注点"],
  "life_stage": "阶段描述"}}"""

    profile = await llm.ainvoke(profile_prompt)

    return {
        "risk_profile": json.loads(profile.content),
        "investment_history": history,
        "recalled_memories": [m["content"] for m in memories],
    }

async def detect_preference_changes(state: FinancialAdvisorState) -> dict:
    """检测客户偏好变化，更新长期记忆。

    场景：客户说"我最近结婚了，想稳健一点"
    → 检测到人生事件变化 → 更新风险偏好记忆
    """
    messages = state["messages"]
    customer_id = state["customer_id"]

    detection_prompt = """分析以下对话，检测客户是否透露了以下变化：
1. 人生事件（结婚、生子、退休、换工作等）
2. 风险偏好变化（更保守或更激进）
3. 投资目标变化（买房、教育、养老等）
4. 财务状况变化（收入增减、大额支出等）

如果检测到变化，输出 JSON 数组；如果没有，输出空数组 []。
格式：[{"type": "变化类型", "content": "具体内容", "impact": "对投资策略的影响"}]
"""
    result = await llm.ainvoke(
        detection_prompt + "\n对话：\n" + format_messages(messages)
    )

    changes = json.loads(result.content)
    for change in changes:
        # 存入长期记忆
        await memory_store.store_memory(
            user_id=customer_id,
            content=f"{change['type']}: {change['content']}。影响: {change['impact']}",
            category="life_event",
        )

    return {
        "life_events": [c["content"] for c in changes],
    }
```

### 9.3 案例三：ERP 智能助手——跨会话任务延续

**业务场景**：某制造企业的 ERP 助手，帮助管理层进行数据分析。分析任务通常跨越多天：第一天看总览，第二天下钻细节，第三天生成报告。

**记忆需求**：
- 短期记忆是核心：必须记住上次分析到哪一步、用了什么筛选条件
- 工作记忆需要摘要：分析对话很长，需要压缩

```python
class ERPAssistantState(TypedDict):
    """ERP 助手 Agent 状态。"""
    messages: Annotated[list, add_messages]
    user_id: str
    analysis_context: dict | None          # 当前分析上下文
    query_history: list[dict]              # 历史查询记录
    conversation_summary: str | None       # 对话摘要
    current_filters: dict | None           # 当前筛选条件

async def restore_analysis_context(state: ERPAssistantState) -> dict:
    """恢复上次的分析上下文。

    场景：用户昨天分析了华东区销售数据，今天回来说"接着昨天的继续"。
    """
    user_id = state["user_id"]

    # 从 Store 加载上次的分析上下文
    last_context = store.get(
        namespace=("users", user_id, "analysis"),
        key="last_context",
    )

    if last_context:
        ctx = last_context.value
        context_msg = f"""上次分析上下文：
- 分析主题：{ctx.get('topic', '未知')}
- 数据范围：{ctx.get('date_range', '未指定')}
- 筛选条件：{ctx.get('filters', '无')}
- 已完成步骤：{ctx.get('completed_steps', '无')}
- 关键发现：{ctx.get('findings', '无')}
- 下一步建议：{ctx.get('next_steps', '无')}"""

        return {
            "analysis_context": ctx,
            "current_filters": ctx.get("filters"),
            "messages": [SystemMessage(content=context_msg)],
        }

    return {"analysis_context": None}

async def save_analysis_context(state: ERPAssistantState) -> dict:
    """保存当前分析上下文，供下次会话恢复。"""
    user_id = state["user_id"]

    # 让 LLM 提取当前分析的结构化上下文
    extract_prompt = """从以下对话中提取数据分析的上下文信息，输出 JSON：
{
  "topic": "分析主题",
  "date_range": "数据时间范围",
  "filters": {"区域": "...", "产品线": "..."},
  "completed_steps": ["已完成的分析步骤"],
  "findings": ["关键发现"],
  "next_steps": ["建议的下一步"]
}"""

    result = await llm.ainvoke(
        extract_prompt + "\n对话：\n" + format_messages(state["messages"])
    )

    context = json.loads(result.content)

    # 存入 Store
    store.put(
        namespace=("users", user_id, "analysis"),
        key="last_context",
        value=context,
    )

    return {}
```

**跨会话效果演示**：

```
# 周一上午
用户: 帮我看看上个月华东区的销售数据
助手: 华东区上月总销售额 ¥2,340 万，环比增长 12%...

# 周一下午（同一 thread_id，工作记忆自动恢复）
用户: 按产品线拆分一下
助手: 好的，基于刚才华东区的数据，按产品线拆分如下...

# 周二上午（新 thread_id，但通过 restore_analysis_context 恢复上下文）
用户: 接着昨天的继续
助手: 好的，昨天我们分析了华东区上月销售数据（总额 ¥2,340 万），
      并按产品线做了拆分。建议的下一步是和去年同期对比，要继续吗？
```

---

## 十、面试高频题与深度解析

### Q1：Agent 的记忆分为哪几个层次？各自解决什么问题？

**答**：Agent 的记忆分为三个层次，对应人类记忆系统：

1. **工作记忆（Working Memory）**：当前对话的消息列表（`messages`），维持多轮对话的连贯性。技术实现是 `MessagesState` + `add_messages` reducer。生命周期是单次会话。

2. **短期记忆（Short-term Memory）**：跨会话的上下文延续，让用户断线后能恢复对话。技术实现是 Checkpoint 持久化 + 会话摘要。生命周期是天到周级别。

3. **长期记忆（Long-term Memory）**：用户画像、偏好、历史知识的积累，提供个性化服务。技术实现是向量数据库（语义检索）+ KV Store（结构化偏好）。生命周期是永久。

三者协作：会话开始时加载长期记忆和短期记忆到工作记忆；会话结束时从工作记忆提取信息存入短期和长期记忆。

---

### Q2：长对话中消息列表越来越长，Token 消耗爆炸怎么办？

**答**：三种策略，按复杂度递增：

1. **滑动窗口**：只保留最近 N 条消息。实现最简单，但会丢失早期重要信息。适合闲聊场景。

2. **Token 限制裁剪**：使用 `trim_messages(max_tokens=4000, strategy="last")`，精确按 Token 数裁剪。比滑动窗口更精确，但同样会丢失信息。适合成本敏感的生产环境。

3. **摘要压缩**：当消息超过阈值时，用 LLM 将旧消息压缩为摘要，保留最近几条原始消息。信息保留最完整，但需要额外的 LLM 调用。适合需要长上下文的复杂任务。

**生产环境推荐**：Token 裁剪 + 摘要压缩结合使用。先尝试 Token 裁剪，如果对话特别长（超过 30 轮），启用摘要压缩。

---

### Q3：Checkpoint 和长期记忆有什么区别？

**答**：这是两个完全不同的概念，解决不同的问题：

| 维度 | Checkpoint | 长期记忆 |
|------|-----------|---------|
| **存储内容** | Agent 的完整执行状态（State 快照） | 提取出的知识和用户画像 |
| **存储粒度** | 每个节点执行后自动保存 | 从对话中选择性提取 |
| **查询方式** | 按 thread_id + checkpoint_id 精确查找 | 按语义相似度模糊检索 |
| **用途** | 会话恢复、时间旅行、Human-in-the-Loop | 个性化服务、知识积累 |
| **生命周期** | 通常保留 7~30 天后清理 | 永久保留（除非设置 TTL） |
| **数据量** | 大（包含完整 State） | 小（只有提取的关键信息） |

**类比**：Checkpoint 像游戏存档（完整的游戏状态），长期记忆像你的笔记本（只记录重要的事情）。

---

### Q4：如何实现跨会话的上下文延续？

**答**：两种方案，根据场景选择：

**方案一：同一 thread_id 延续**
- 适用：同一个任务跨多天完成（如 ERP 数据分析）
- 实现：使用 `user-{id}-task-{task_id}` 作为 thread_id，Checkpoint 自动恢复完整状态
- 优点：零额外开发，状态完整恢复
- 缺点：消息列表会越来越长，需要配合摘要压缩

**方案二：新 thread_id + 上下文加载**
- 适用：每次来电是新会话，但需要知道历史（如客服场景）
- 实现：每次新会话开始时，从数据库加载上次会话摘要和客户画像，注入 SystemMessage
- 优点：每次会话干净，不会积累过多历史消息
- 缺点：需要额外开发摘要生成和加载逻辑

**生产环境推荐**：方案二更灵活，因为你可以精确控制加载哪些上下文，而不是把所有历史消息都带上。

---

### Q5：长期记忆为什么用向量数据库？关系型数据库不行吗？

**答**：核心原因是**查询方式不同**。

关系型数据库擅长精确查询：`SELECT * FROM memories WHERE user_id = 'xxx' AND category = 'preference'`。但长期记忆的召回是**语义匹配**——当用户说"帮我看看销售情况"，你需要召回"用户偏好简洁报表"和"用户关注华东区"，这些和"销售情况"没有关键词重叠，但语义相关。

向量数据库通过 Embedding 将文本转换为高维向量，用余弦相似度计算语义距离，能找到"意思相近但用词不同"的记忆。

**实际选型**：
- 小规模（< 10 万条记忆）：ChromaDB（嵌入式，零运维）
- 中规模（10 万~1000 万）：PostgreSQL + pgvector（统一存储，减少运维）
- 大规模（> 1000 万）：专用向量数据库（Milvus、Pinecone）

---

### Q6：LangGraph Checkpointer 的内部工作原理是什么？

**答**：Checkpointer 的核心是**自动快照 + 链式存储**：

1. **自动触发**：每个节点执行完毕后，LangGraph 自动调用 `checkpointer.put()` 保存当前 State 的完整快照。

2. **链式结构**：每个 Checkpoint 包含 `checkpoint_id`（当前快照 ID）和 `parent_checkpoint_id`（上一个快照 ID），形成链表。这支持时间旅行——沿着链表回溯到任意历史状态。

3. **序列化**：State 中的所有字段被序列化为 JSON/二进制存储。`channel_values` 存储字段值，`channel_versions` 存储版本号（用于增量更新优化）。

4. **恢复机制**：当同一个 `thread_id` 再次调用 `invoke()` 时，Checkpointer 自动加载该 thread 最新的 Checkpoint，恢复完整 State，然后在此基础上继续执行。

5. **存储后端**：`MemorySaver`（内存字典）→ `SqliteSaver`（SQLite 文件）→ `PostgresSaver`（PostgreSQL 表），接口统一，只是持久化策略不同。

---

### Q7：复杂 Agent 的 State 应该怎么设计？

**答**：遵循以下原则：

1. **按职责分组**：对话管理、用户上下文、业务数据、记忆系统、流程控制、可观测性，每组字段清晰分开。

2. **选择合适的 Reducer**：
   - `messages` 用 `add_messages`（智能合并）
   - 日志类字段用 `operator.add`（追加）
   - 业务字段不用 Reducer（覆盖更新）
   - 需要合并的字典用自定义 `merge_dicts` Reducer

3. **只存必要数据**：State 会被 Checkpoint 序列化保存，大对象（完整文档、图片）存外部存储，State 中只存引用 ID。

4. **可选字段用 `| None`**：不是每个节点都会填充所有字段，用 `None` 表示"尚未设置"。

5. **遵循不可变性**：节点不直接修改 `state` 对象，而是返回一个 dict 描述变更。

```python
# 示例：一个设计良好的 State
class WellDesignedState(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer: 智能合并
    user_id: str                             # 无 Reducer: 覆盖
    logs: Annotated[list, add]               # Reducer: 追加
    intent: str | None                       # 可选字段
```

---

### Q8：生产环境的记忆架构应该怎么选？

**答**：推荐的生产架构：

**存储层**：
- **Checkpoint**：PostgreSQL（`AsyncPostgresSaver` + 连接池），支持高并发和多实例
- **长期记忆**：PostgreSQL + pgvector 扩展（统一存储，减少运维复杂度）
- **用户偏好**：PostgreSQL 的 JSONB 字段或 LangGraph Store API

**管理策略**：
- 工作记忆：`trim_messages` Token 裁剪 + 超长对话摘要压缩
- 短期记忆：会话结束时生成结构化摘要，Checkpoint 保留 30 天后 GC
- 长期记忆：对话后自动提取，存储前语义去重（相似度 > 0.92 跳过），时效性信息设置 TTL

**多租户**：thread_id 包含 tenant_id 前缀 + SQL 查询强制 tenant_id 过滤 + PostgreSQL RLS 兜底

**性能优化**：
- 连接池（min=5, max=20）
- pgvector IVFFlat 索引
- Checkpoint 分批 GC（避免长事务）
- State 中不存大对象

---

### Q9：add_messages reducer 和普通的 list append 有什么区别？

**答**：`add_messages` 比简单的 `list.append` 智能得多，它支持三种操作：

1. **追加**：新消息的 `id` 不存在于已有列表中 → 追加到末尾（和 append 一样）
2. **替换**：新消息的 `id` 和已有消息相同 → 替换该消息（用于更新回答）
3. **删除**：传入 `RemoveMessage(id=xxx)` → 从列表中删除指定消息（用于摘要压缩时清理旧消息）

这三种操作使得消息管理非常灵活：你可以追加新消息、更新已有消息、删除不需要的消息，而不需要手动操作列表。

---

### Q10：如果让你从零设计一个企业级 Agent 的记忆系统，你会怎么做？

**答**：分四步走：

**第一步：确定记忆需求**
- 分析业务场景：是纯对话（只需工作记忆）还是需要跨会话延续（需要短期记忆）还是需要个性化（需要长期记忆）？
- 确定数据量级：日活用户数、平均对话轮次、需要保留多久？

**第二步：选择存储方案**
- 开发环境：`MemorySaver`（零配置）
- 测试环境：`SqliteSaver`（单文件持久化）
- 生产环境：`AsyncPostgresSaver` + pgvector（统一存储）

**第三步：设计 State 和记忆流程**
- 定义 `AgentState`（TypedDict），包含消息、业务上下文、记忆字段
- 设计记忆加载节点（会话开始时）和记忆保存节点（会话结束时）
- 实现消息管理策略（Token 裁剪 + 摘要压缩）

**第四步：运维和优化**
- Checkpoint GC 定时任务（保留 30 天）
- 长期记忆去重和 TTL 清理
- 监控指标：Checkpoint 大小、向量检索延迟、记忆命中率
- 多租户隔离（thread_id 前缀 + SQL 过滤 + RLS）

# 02 - LangChain 1.x 深度解析

> LangChain 1.x 是一次重大范式转变。如果你还在用 0.x 的 AgentExecutor，面试官会认为你技术栈过时了。

---

## 一、LangChain 1.x vs 0.x：到底变了什么？

### 1.1 核心变化一览

| 维度 | 0.x 时代 | 1.x 时代 |
|------|----------|----------|
| Agent 创建 | `initialize_agent()` / `AgentExecutor` | `create_agent()` |
| Agent 循环控制 | 黑盒，难以定制 | Middleware 机制，完全可控 |
| 底层编排 | 内置简单循环 | 基于 LangGraph |
| 模型集成 | 统一但受限 | 支持最新内容类型（音频、图片等） |
| 工具定义 | `@tool` 装饰器 | `@tool` + 结构化 schema + provider 特定工具 |
| 包结构 | 大而全的 `langchain` 包 | 拆分为 `langchain-core` + 集成包 |

### 1.2 为什么要升级？

**现实类比**：0.x 时代的 AgentExecutor 就像一辆自动挡汽车——简单好开，但你没法精确控制换挡时机。1.x 的 create_agent + middleware 就像一辆带拨片换挡的赛车——既能自动，也能在关键时刻手动介入。

**企业痛点**：
- 0.x 的 AgentExecutor 是个黑盒，出了问题很难调试
- 无法在 Agent 循环的特定阶段插入业务逻辑（如权限校验、审计日志）
- 不支持复杂的状态管理和持久化

---

## 二、create_agent：新一代 Agent 创建方式

### 2.1 基本用法

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# 1. 初始化模型
llm = init_chat_model("openai:gpt-4o")

# 2. 定义工具
@tool
def search_orders(order_id: str) -> str:
    """根据订单号查询订单详情，包括状态、金额、物流信息"""
    # 实际调用订单系统 API
    return f"订单 {order_id}: 已发货，预计明天到达"

@tool
def send_notification(user_id: str, message: str) -> str:
    """向用户发送通知消息"""
    return f"已向用户 {user_id} 发送通知: {message}"

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[search_orders, send_notification],
    prompt="你是一个电商客服助手，帮助用户查询订单和处理问题。"
)

# 4. 运行
result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我查一下订单 ORD-12345 的状态"}]
})
```

### 2.2 create_agent 的核心参数

```python
agent = create_agent(
    model=llm,                    # 必需：LLM 模型实例
    tools=[...],                  # 必需：工具列表
    prompt="...",                 # 系统提示词（字符串或 ChatPromptTemplate）
    response_format=MySchema,     # 可选：结构化输出 schema
    middleware=[...],             # 可选：中间件列表
    checkpointer=checkpointer,   # 可选：状态持久化
    interrupt_before=["tools"],   # 可选：在工具调用前暂停（Human-in-the-Loop）
    interrupt_after=["tools"],    # 可选：在工具调用后暂停
)
```

### 2.3 真实企业场景：智能 IT 运维助手

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ServerStatus(BaseModel):
    """服务器状态查询结果"""
    hostname: str = Field(description="服务器主机名")
    cpu_usage: float = Field(description="CPU 使用率")
    memory_usage: float = Field(description="内存使用率")
    status: str = Field(description="运行状态")

@tool
def check_server_status(hostname: str) -> str:
    """检查指定服务器的运行状态，包括 CPU、内存、磁盘使用率"""
    # 实际调用监控系统 API（如 Prometheus/Zabbix）
    return f"{hostname}: CPU 85%, 内存 72%, 状态正常"

@tool
def restart_service(hostname: str, service_name: str) -> str:
    """重启指定服务器上的服务"""
    return f"已在 {hostname} 上重启 {service_name}"

@tool
def query_logs(hostname: str, keyword: str, minutes: int = 30) -> str:
    """查询指定服务器最近 N 分钟的日志，支持关键词过滤"""
    return f"在 {hostname} 最近 {minutes} 分钟日志中找到 3 条包含 '{keyword}' 的记录"

llm = init_chat_model("openai:gpt-4o")

ops_agent = create_agent(
    model=llm,
    tools=[check_server_status, restart_service, query_logs],
    prompt="""你是一个高级 IT 运维助手。你的职责是：
1. 诊断服务器和服务的问题
2. 在确认安全的情况下执行运维操作
3. 分析日志找出根因
4. 给出专业的运维建议

注意：重启服务前必须先检查服务器状态和相关日志。"""
)
```

---

## 三、Middleware：Agent 循环的精确控制

### 3.1 什么是 Middleware？

**现实类比**：Middleware 就像机场的安检通道。每个旅客（请求）在登机（执行）前后都要经过安检（中间件）。你可以在安检通道里加各种检查——身份验证、行李扫描、违禁品检测。

在 Agent 的上下文中，Middleware 可以拦截和修改 Agent 循环中的每一步：
- 模型调用前后
- 工具调用前后
- 每次循环迭代前后

### 3.2 Middleware 的核心价值

| 场景 | 没有 Middleware | 有 Middleware |
|------|----------------|---------------|
| 审计日志 | 手动在每个工具里加日志 | 统一拦截，自动记录 |
| 权限校验 | 每个工具自己校验 | 统一拦截，集中校验 |
| 速率限制 | 自己实现计数器 | 中间件统一控制 |
| 敏感信息过滤 | 每个输出都要手动检查 | 统一拦截，自动脱敏 |
| 成本控制 | 事后统计 | 实时监控，超限自动停止 |

### 3.3 实战：企业级审计日志中间件

```python
import logging
from datetime import datetime
from langchain.agents import create_agent, AgentMiddleware

logger = logging.getLogger("agent_audit")

class AuditMiddleware(AgentMiddleware):
    """企业级审计日志中间件 —— 记录 Agent 的每一步操作"""

    async def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs):
        """工具调用前：记录操作意图"""
        logger.info(
            f"[AUDIT] Tool Start | tool={tool_name} | "
            f"input={tool_input} | time={datetime.now().isoformat()}"
        )
        # 可以在这里做权限校验
        if tool_name == "restart_service" and not self.has_permission("ops_admin"):
            raise PermissionError(f"无权执行 {tool_name}")

    async def on_tool_end(self, tool_name: str, tool_output: str, **kwargs):
        """工具调用后：记录执行结果"""
        logger.info(
            f"[AUDIT] Tool End | tool={tool_name} | "
            f"output={tool_output[:200]} | time={datetime.now().isoformat()}"
        )

    async def on_model_start(self, messages: list, **kwargs):
        """模型调用前：可以修改或过滤消息"""
        # 过滤敏感信息
        sanitized = self.sanitize_messages(messages)
        return sanitized

    def has_permission(self, role: str) -> bool:
        """检查当前用户是否有指定角色权限"""
        # 实际项目中从 RBAC 系统查询
        return True

    def sanitize_messages(self, messages: list) -> list:
        """脱敏处理：隐藏手机号、身份证号等"""
        import re
        sanitized = []
        for msg in messages:
            content = msg.get("content", "")
            # 手机号脱敏
            content = re.sub(r'1[3-9]\d{9}', '1**********', content)
            # 身份证号脱敏
            content = re.sub(r'\d{17}[\dXx]', '****', content)
            sanitized.append({**msg, "content": content})
        return sanitized

# 使用中间件
agent = create_agent(
    model=llm,
    tools=[check_server_status, restart_service],
    middleware=[AuditMiddleware()],
    prompt="你是 IT 运维助手"
)
```

### 3.4 实战：成本控制中间件

```python
class CostControlMiddleware(AgentMiddleware):
    """控制 Agent 的 Token 消耗，防止失控"""

    def __init__(self, max_tokens: int = 100_000, max_iterations: int = 20):
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.total_tokens = 0
        self.iteration_count = 0

    async def on_model_end(self, response, **kwargs):
        """模型调用后：统计 Token 消耗"""
        usage = response.usage_metadata
        if usage:
            self.total_tokens += usage.get("total_tokens", 0)

        self.iteration_count += 1

        if self.total_tokens > self.max_tokens:
            raise RuntimeError(
                f"Token 消耗超限: {self.total_tokens}/{self.max_tokens}"
            )
        if self.iteration_count > self.max_iterations:
            raise RuntimeError(
                f"迭代次数超限: {self.iteration_count}/{self.max_iterations}"
            )
```

---

## 四、init_chat_model：统一模型初始化

### 4.1 为什么需要统一初始化？

**企业痛点**：不同项目用不同的模型提供商（OpenAI、Anthropic、Azure、本地模型），每次切换都要改代码。

`init_chat_model` 提供了统一的初始化接口：

```python
from langchain.chat_models import init_chat_model

# OpenAI
llm = init_chat_model("openai:gpt-4o")

# Anthropic
llm = init_chat_model("anthropic:claude-sonnet-4-20250514")

# Azure OpenAI
llm = init_chat_model("azure_openai:gpt-4o", azure_deployment="my-deployment")

# 本地 Ollama
llm = init_chat_model("ollama:llama3")

# 通过配置切换，代码零修改
import os
model_name = os.getenv("LLM_MODEL", "openai:gpt-4o")
llm = init_chat_model(model_name, temperature=0.7)
```

### 4.2 模型参数配置最佳实践

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "openai:gpt-4o",
    temperature=0,          # 企业场景建议 0，确保输出稳定
    max_tokens=4096,         # 根据场景设置上限
    timeout=30,              # 超时时间（秒）
    max_retries=3,           # 自动重试次数
    # rate_limiter=...,      # 速率限制器
)
```

---

## 五、LCEL（LangChain Expression Language）

### 5.1 LCEL 是什么？

LCEL 是 LangChain 的声明式链式调用语法，用 `|` 管道符连接组件。

**现实类比**：就像 Linux 的管道命令 `cat file | grep error | sort | head -10`，每个组件处理完数据传给下一个。

### 5.2 核心用法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")

# 简单链：Prompt → LLM → 解析输出
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{domain}分析师"),
        ("human", "{question}")
    ])
    | llm
    | StrOutputParser()
)

result = chain.invoke({"domain": "金融", "question": "分析一下当前的市场趋势"})
```

### 5.3 LCEL 的高级模式

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# 并行执行多个链
parallel_chain = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
    keywords=keyword_chain,
)

# 条件路由
def route(input_dict: dict) -> str:
    if input_dict["category"] == "technical":
        return technical_chain
    return general_chain

routed_chain = RunnableLambda(route)

# 带回退的链（主模型失败时用备用模型）
primary = init_chat_model("openai:gpt-4o")
fallback = init_chat_model("anthropic:claude-sonnet-4-20250514")

robust_chain = prompt | primary.with_fallbacks([fallback]) | parser
```

---

## 六、结构化输出（Structured Output）

### 6.1 为什么需要结构化输出？

**企业痛点**：LLM 返回的是自然语言文本，但下游系统需要结构化数据（JSON）。手动解析不可靠，正则匹配容易出错。

**现实场景**：客服 Agent 需要从用户描述中提取工单信息（类型、优先级、描述），然后写入工单系统。

### 6.2 使用 Pydantic 定义输出 Schema

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class TicketInfo(BaseModel):
    """客服工单信息"""
    category: str = Field(description="工单类型：退款/换货/咨询/投诉")
    priority: str = Field(description="优先级：高/中/低")
    summary: str = Field(description="问题摘要，不超过50字")
    customer_emotion: str = Field(description="客户情绪：愤怒/焦虑/平静/满意")

llm = init_chat_model("openai:gpt-4o")

# 方式一：with_structured_output
structured_llm = llm.with_structured_output(TicketInfo)
ticket = structured_llm.invoke("我买的手机屏幕碎了，才用了三天！我要退款！太气人了！")
# ticket.category = "退款"
# ticket.priority = "高"
# ticket.summary = "手机屏幕碎裂，使用三天，要求退款"
# ticket.customer_emotion = "愤怒"

# 方式二：在 create_agent 中使用
agent = create_agent(
    model=llm,
    tools=[...],
    response_format=TicketInfo,  # Agent 最终输出为结构化数据
)
```

---

## 七、本章面试要点

1. **LangChain 1.x 相比 0.x 最大的变化是什么？**
   → create_agent 替代 AgentExecutor，引入 Middleware 机制，底层基于 LangGraph

2. **Middleware 能解决什么问题？举个企业级的例子**
   → 统一拦截 Agent 循环，实现审计日志、权限校验、成本控制、敏感信息脱敏等横切关注点

3. **LCEL 的设计理念是什么？和直接写 Python 函数调用有什么区别？**
   → 声明式、可组合、天然支持流式和异步、支持 fallback 和并行

4. **如何实现模型的灵活切换？**
   → init_chat_model 统一接口 + 环境变量配置，代码零修改

5. **结构化输出在企业场景中为什么重要？**
   → LLM 输出需要对接下游系统，结构化输出保证数据格式可靠，避免解析失败

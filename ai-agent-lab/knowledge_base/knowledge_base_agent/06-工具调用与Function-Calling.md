# 04 - 工具调用与 Function Calling

> 没有工具的 Agent 只是一个聊天机器人。工具是 Agent 的手和脚，让它能真正"做事"。

---

## 一、工具调用的本质

### 1.1 什么是 Tool Calling？

Tool Calling（工具调用）是 LLM 根据用户需求，**自主决定**调用哪个外部函数、传什么参数的能力。

**现实类比**：你是一个项目经理，手下有几个工程师（工具）。客户提了一个需求，你（LLM）分析后决定：
- 这个需求需要查数据库 → 派数据库工程师（DB 查询工具）
- 查完后需要生成报表 → 派前端工程师（报表生成工具）
- 最后需要发邮件通知 → 派运维工程师（邮件工具）

你不需要事先写死"先查数据库再生成报表再发邮件"的流程，LLM 会根据具体情况自己决定。

### 1.2 Tool Calling 的工作流程

```
用户输入 → LLM 分析 → 决定调用哪个工具 → 生成工具调用参数
                                              ↓
                                        执行工具函数
                                              ↓
                                        返回工具结果
                                              ↓
                                    LLM 根据结果生成回答
```

---

## 二、定义工具的三种方式

### 2.1 方式一：@tool 装饰器（最常用）

```python
from langchain_core.tools import tool

@tool
def query_customer_info(customer_id: str) -> str:
    """根据客户 ID 查询客户详细信息，包括姓名、等级、历史订单数。

    Args:
        customer_id: 客户唯一标识，格式为 CUST-XXXXX
    """
    # 实际调用 CRM 系统 API
    customer = crm_api.get_customer(customer_id)
    return f"客户: {customer.name}, 等级: {customer.level}, 历史订单: {customer.order_count}"
```

**关键点**：docstring 非常重要！LLM 通过 docstring 理解工具的用途和参数含义。写得好，LLM 调用就准确；写得差，LLM 就会乱调。

### 2.2 方式二：Pydantic Schema（精确控制参数）

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class OrderQueryInput(BaseModel):
    """订单查询参数"""
    order_id: str = Field(description="订单号，格式 ORD-XXXXX")
    include_logistics: bool = Field(
        default=False,
        description="是否包含物流信息"
    )
    include_payment: bool = Field(
        default=False,
        description="是否包含支付信息"
    )

@tool(args_schema=OrderQueryInput)
def query_order(order_id: str, include_logistics: bool = False, include_payment: bool = False) -> str:
    """查询订单详情，支持选择性包含物流和支付信息"""
    result = order_service.get(order_id)
    if include_logistics:
        result += f"\n物流: {logistics_service.track(order_id)}"
    if include_payment:
        result += f"\n支付: {payment_service.get(order_id)}"
    return result
```

### 2.3 方式三：StructuredTool（完全自定义）

```python
from langchain_core.tools import StructuredTool

def execute_sql(query: str, database: str = "main") -> str:
    """执行 SQL 查询"""
    # 安全检查：只允许 SELECT
    if not query.strip().upper().startswith("SELECT"):
        return "错误：只允许执行 SELECT 查询"
    result = db_service.execute(query, database)
    return str(result)

sql_tool = StructuredTool.from_function(
    func=execute_sql,
    name="sql_query",
    description="在指定数据库上执行 SQL 查询。仅支持 SELECT 语句，用于数据分析和报表查询。",
    args_schema=SQLQueryInput,
    return_direct=False,  # False: 结果返回给 LLM 继续处理; True: 直接返回给用户
)
```

---

## 三、工具绑定与模型集成

### 3.1 bind_tools：将工具绑定到模型

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")

# 绑定工具
llm_with_tools = llm.bind_tools([
    query_customer_info,
    query_order,
    sql_tool,
])

# 调用时 LLM 会自动决定是否调用工具
response = llm_with_tools.invoke("帮我查一下客户 CUST-00123 的信息")

# 检查是否有工具调用
if response.tool_calls:
    for call in response.tool_calls:
        print(f"工具: {call['name']}")
        print(f"参数: {call['args']}")
```

### 3.2 在 LangGraph 中使用工具

```python
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage

# 工具列表
tools = [query_customer_info, query_order, sql_tool]

# 创建工具执行节点
tool_node = ToolNode(tools)

# Agent 节点
def agent(state: MessagesState) -> dict:
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 路由函数
def should_use_tools(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # 工具执行完回到 Agent

app = graph.compile()
```

---

## 四、高级工具模式

### 4.1 工具错误处理

```python
from langchain_core.tools import tool, ToolException

@tool
def risky_operation(param: str) -> str:
    """执行一个可能失败的操作"""
    try:
        result = external_api.call(param)
        return result
    except ConnectionError:
        raise ToolException(
            "外部服务连接失败，请稍后重试。"
            "建议：检查网络连接或联系运维团队。"
        )
    except ValueError as e:
        raise ToolException(f"参数错误: {e}。请检查输入格式。")

# ToolException 会被优雅地返回给 LLM，LLM 可以据此调整策略
# 比如换一个工具、修改参数重试、或者告诉用户问题所在
```

### 4.2 异步工具（I/O 密集场景）

```python
import httpx
from langchain_core.tools import tool

@tool
async def async_search(query: str) -> str:
    """异步搜索外部知识库"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.knowledge-base.com/search",
            params={"q": query}
        )
        return response.json()["results"]

@tool
async def async_translate(text: str, target_lang: str = "en") -> str:
    """异步翻译文本"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.translate.com/v1/translate",
            json={"text": text, "target": target_lang}
        )
        return response.json()["translated_text"]
```

### 4.3 动态工具选择

**企业场景**：不同角色的用户能使用不同的工具。管理员能重启服务，普通用户只能查看状态。

```python
def get_tools_for_role(role: str) -> list:
    """根据用户角色返回可用工具列表"""
    base_tools = [query_status, view_logs]

    if role == "admin":
        return base_tools + [restart_service, modify_config]
    elif role == "operator":
        return base_tools + [restart_service]
    else:
        return base_tools

# 在 Agent 节点中动态绑定工具
def agent_node(state: AgentState) -> dict:
    user_role = state["user_role"]
    available_tools = get_tools_for_role(user_role)
    llm_with_tools = llm.bind_tools(available_tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 4.4 工具结果验证

```python
from pydantic import BaseModel, field_validator

class StockQueryResult(BaseModel):
    """股票查询结果验证"""
    symbol: str
    price: float
    change_percent: float

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("股价必须为正数")
        return v

@tool
def query_stock(symbol: str) -> str:
    """查询股票实时价格"""
    raw_data = stock_api.get_quote(symbol)
    # 验证数据质量
    validated = StockQueryResult(**raw_data)
    return f"{validated.symbol}: ¥{validated.price:.2f} ({validated.change_percent:+.2f}%)"
```

---

## 五、真实企业场景：供应链智能助手

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class InventoryAlert(BaseModel):
    sku: str = Field(description="商品 SKU")
    warehouse: str = Field(description="仓库编号")
    current_stock: int = Field(description="当前库存")
    threshold: int = Field(description="预警阈值")

@tool
def check_inventory(sku: str, warehouse: str | None = None) -> str:
    """查询指定商品的库存情况。可指定仓库，不指定则查询所有仓库汇总。"""
    # 调用 WMS（仓储管理系统）API
    return inventory_service.query(sku, warehouse)

@tool
def create_purchase_order(sku: str, quantity: int, supplier_id: str) -> str:
    """创建采购订单。当库存低于阈值时，自动触发补货。

    Args:
        sku: 商品 SKU
        quantity: 采购数量
        supplier_id: 供应商 ID
    """
    po = procurement_service.create_order(sku, quantity, supplier_id)
    return f"采购订单已创建: {po.id}, 预计到货: {po.eta}"

@tool
def query_supplier_performance(supplier_id: str) -> str:
    """查询供应商的历史表现，包括交货准时率、质量合格率、价格竞争力。"""
    perf = supplier_service.get_performance(supplier_id)
    return (
        f"供应商 {supplier_id}: "
        f"准时率 {perf.on_time_rate:.1%}, "
        f"合格率 {perf.quality_rate:.1%}, "
        f"价格评分 {perf.price_score}/10"
    )

@tool
def forecast_demand(sku: str, days: int = 30) -> str:
    """预测指定商品未来 N 天的需求量，基于历史销售数据和季节性因素。"""
    forecast = ml_service.predict_demand(sku, days)
    return f"SKU {sku} 未来 {days} 天预测需求: {forecast.quantity} 件"

# 供应链 Agent
supply_chain_agent = create_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[check_inventory, create_purchase_order, query_supplier_performance, forecast_demand],
    prompt="""你是一个供应链智能助手，负责：
1. 监控库存水平，及时预警
2. 根据需求预测和供应商表现，推荐最优采购方案
3. 自动创建采购订单（需确认后执行）

决策原则：
- 库存低于安全库存时，优先选择准时率最高的供应商
- 采购量 = 预测需求 × 1.2（留 20% 安全余量）
- 单笔采购超过 10 万元需要人工审批"""
)
```

---

## 六、本章面试要点

1. **Tool Calling 的工作原理是什么？**
   → LLM 根据工具的 name + description + schema 决定调用哪个工具、传什么参数。工具执行结果返回给 LLM 继续推理

2. **如何写好工具的 docstring？为什么它很重要？**
   → docstring 是 LLM 理解工具用途的唯一依据。要写清楚：做什么、什么时候用、参数含义、返回什么

3. **工具调用失败怎么处理？**
   → 使用 ToolException 返回友好错误信息给 LLM，LLM 可以据此调整策略（重试、换工具、告知用户）

4. **如何实现基于角色的工具权限控制？**
   → 动态工具绑定：根据用户角色返回不同的工具列表，bind_tools 时只绑定该角色可用的工具

5. **同步工具 vs 异步工具，什么时候用哪个？**
   → I/O 密集（HTTP 请求、数据库查询）用异步；CPU 密集（本地计算）用同步

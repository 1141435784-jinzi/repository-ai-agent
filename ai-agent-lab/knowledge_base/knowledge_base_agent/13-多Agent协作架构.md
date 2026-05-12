# 07 - 多 Agent 协作架构

> 一个 Agent 搞不定的事，让一群 Agent 协作完成。这是企业级 AI 系统的终极形态。

---

## 一、为什么需要多 Agent？

### 1.1 单 Agent 的瓶颈

**现实类比**：一个全栈工程师能写前端、后端、运维，但效率不如一个专业团队。当任务复杂到一定程度，专业分工 + 协作比"一个人干所有事"更高效。

单 Agent 的问题：
- **Prompt 过载**：一个 Prompt 塞太多职责，LLM 容易混乱
- **工具过多**：绑定 20+ 工具，LLM 选择准确率下降
- **上下文污染**：不同任务的上下文混在一起，互相干扰
- **难以维护**：一个巨大的 Agent 难以测试和迭代

### 1.2 多 Agent 的优势

| 维度 | 单 Agent | 多 Agent |
|------|----------|----------|
| 职责 | 一个 Agent 干所有事 | 每个 Agent 专精一个领域 |
| Prompt | 又长又复杂 | 短小精悍，职责清晰 |
| 工具 | 全部绑定 | 按需分配 |
| 可维护性 | 牵一发动全身 | 独立开发、测试、部署 |
| 可扩展性 | 加功能就加复杂度 | 加一个新 Agent 即可 |

---

## 二、多 Agent 协作模式

### 2.1 Supervisor 模式（主管模式）

**现实类比**：就像一个部门经理。客户的需求先到经理这里，经理判断该派谁去处理，处理完了汇报给经理，经理再决定下一步。

```
         ┌─────────────────┐
         │   Supervisor     │  ← 接收任务、分配工作、汇总结果
         │   (调度 Agent)   │
         └────────┬────────┘
          ┌───────┼───────┐
          ▼       ▼       ▼
     ┌────────┐ ┌────────┐ ┌────────┐
     │ 客服   │ │ 技术   │ │ 财务   │
     │ Agent  │ │ Agent  │ │ Agent  │
     └────────┘ └────────┘ └────────┘
```

**真实企业场景：电商客服中心**

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")

# ========== 定义各专业 Agent ==========

def customer_service_agent(state: MessagesState) -> dict:
    """客服 Agent：处理退换货、投诉、咨询"""
    response = llm.invoke([
        {"role": "system", "content": "你是电商客服专家，负责处理退换货、投诉和一般咨询。态度友好专业。"},
        *state["messages"]
    ])
    return {"messages": [response]}

def technical_support_agent(state: MessagesState) -> dict:
    """技术支持 Agent：处理产品使用问题、故障排查"""
    tech_llm = llm.bind_tools([query_product_manual, diagnose_issue])
    response = tech_llm.invoke([
        {"role": "system", "content": "你是技术支持专家，负责产品使用指导和故障排查。"},
        *state["messages"]
    ])
    return {"messages": [response]}

def finance_agent(state: MessagesState) -> dict:
    """财务 Agent：处理退款、发票、账单问题"""
    finance_llm = llm.bind_tools([process_refund, generate_invoice])
    response = finance_llm.invoke([
        {"role": "system", "content": "你是财务专家，负责退款处理、发票开具和账单查询。"},
        *state["messages"]
    ])
    return {"messages": [response]}

# ========== Supervisor：任务调度 ==========

def supervisor(state: MessagesState) -> dict:
    """Supervisor 决定下一步交给哪个 Agent"""
    response = llm.with_structured_output(RouterOutput).invoke([
        {"role": "system", "content": """你是客服中心主管。根据用户问题，决定交给哪个团队处理：
- customer_service：退换货、投诉、一般咨询
- technical_support：产品使用问题、故障排查
- finance：退款、发票、账单
- FINISH：问题已解决，结束对话"""},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[路由到: {response.next_agent}]")]}

class RouterOutput(BaseModel):
    next_agent: Literal["customer_service", "technical_support", "finance", "FINISH"]
    reason: str

def route_to_agent(state: MessagesState) -> str:
    """根据 Supervisor 的决策路由"""
    last_msg = state["messages"][-1].content
    if "customer_service" in last_msg:
        return "customer_service"
    elif "technical_support" in last_msg:
        return "technical_support"
    elif "finance" in last_msg:
        return "finance"
    return "end"

# ========== 构建图 ==========

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor)
graph.add_node("customer_service", customer_service_agent)
graph.add_node("technical_support", technical_support_agent)
graph.add_node("finance", finance_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_to_agent, {
    "customer_service": "customer_service",
    "technical_support": "technical_support",
    "finance": "finance",
    "end": END,
})

# 各 Agent 处理完后回到 Supervisor
graph.add_edge("customer_service", "supervisor")
graph.add_edge("technical_support", "supervisor")
graph.add_edge("finance", "supervisor")

app = graph.compile()
```

### 2.2 Swarm 模式（群体智能模式）

**现实类比**：就像一个开放式办公室。没有固定的经理，谁擅长就谁来。Agent A 处理到一半发现需要 Agent B 的专业能力，直接把任务"递"给 Agent B。

**核心特点**：Agent 之间直接交接（Handoff），没有中央调度。

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 定义 Handoff 工具
@tool
def transfer_to_technical():
    """当用户遇到技术问题时，转接给技术支持专家"""
    pass  # LangGraph 会处理实际的转接逻辑

@tool
def transfer_to_finance():
    """当用户需要处理退款或发票时，转接给财务专家"""
    pass

@tool
def transfer_to_customer_service():
    """当需要回到一般客服处理时，转接给客服"""
    pass

# 每个 Agent 都有"转接"工具
cs_agent = create_react_agent(
    model=llm,
    tools=[search_faq, transfer_to_technical, transfer_to_finance],
    prompt="你是客服专家。如果遇到技术问题或财务问题，转接给对应专家。"
)

tech_agent = create_react_agent(
    model=llm,
    tools=[query_product_manual, diagnose_issue, transfer_to_customer_service],
    prompt="你是技术支持专家。解决技术问题后，转回客服。"
)

finance_agent = create_react_agent(
    model=llm,
    tools=[process_refund, generate_invoice, transfer_to_customer_service],
    prompt="你是财务专家。处理完财务问题后，转回客服。"
)
```

### 2.3 层级模式（Hierarchical）

**现实类比**：大公司的组织架构。CEO → VP → 部门经理 → 员工。每一层只和直接上下级沟通。

**适用场景**：大型复杂系统，需要多层决策。

```
              ┌──────────────┐
              │  总调度 Agent  │
              └──────┬───────┘
           ┌─────────┼─────────┐
           ▼         ▼         ▼
      ┌────────┐ ┌────────┐ ┌────────┐
      │ 售前   │ │ 售中   │ │ 售后   │  ← 二级 Supervisor
      │ 主管   │ │ 主管   │ │ 主管   │
      └───┬────┘ └───┬────┘ └───┬────┘
       ┌──┼──┐    ┌──┼──┐    ┌──┼──┐
       ▼  ▼  ▼    ▼  ▼  ▼    ▼  ▼  ▼
      产品 方案 报价 订单 物流 支付 客服 退换 投诉
```

```python
# 售后子图
def build_after_sales_subgraph():
    after_sales_graph = StateGraph(MessagesState)

    after_sales_graph.add_node("after_sales_supervisor", after_sales_supervisor)
    after_sales_graph.add_node("customer_service", cs_agent)
    after_sales_graph.add_node("returns", returns_agent)
    after_sales_graph.add_node("complaints", complaints_agent)

    after_sales_graph.add_edge(START, "after_sales_supervisor")
    after_sales_graph.add_conditional_edges(
        "after_sales_supervisor",
        route_after_sales,
        {"customer_service": "customer_service", "returns": "returns", "complaints": "complaints", "end": END}
    )
    # ... 各 Agent 回到 supervisor 的边

    return after_sales_graph.compile()

# 主图
main_graph = StateGraph(MessagesState)
main_graph.add_node("main_supervisor", main_supervisor)
main_graph.add_node("pre_sales", build_pre_sales_subgraph())
main_graph.add_node("in_sales", build_in_sales_subgraph())
main_graph.add_node("after_sales", build_after_sales_subgraph())
# ...
```

---

## 三、Agent 间通信机制

### 3.1 通过共享状态通信

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list, add]
    current_agent: str
    task_results: dict          # 各 Agent 的处理结果
    shared_context: dict        # 共享上下文

def agent_a(state: MultiAgentState) -> dict:
    # Agent A 处理完后，把结果写入共享状态
    result = do_something()
    return {
        "task_results": {**state["task_results"], "agent_a": result},
        "messages": [AIMessage(content=f"Agent A 完成: {result}")]
    }

def agent_b(state: MultiAgentState) -> dict:
    # Agent B 可以读取 Agent A 的结果
    agent_a_result = state["task_results"].get("agent_a")
    result = do_something_with(agent_a_result)
    return {
        "task_results": {**state["task_results"], "agent_b": result},
        "messages": [AIMessage(content=f"Agent B 完成: {result}")]
    }
```

### 3.2 通过消息传递通信

```python
def agent_with_message_passing(state: MessagesState) -> dict:
    """Agent 通过消息列表传递信息"""
    # 读取其他 Agent 的消息
    previous_messages = state["messages"]

    # 处理并添加自己的消息
    response = llm.invoke([
        {"role": "system", "content": "你是分析师。基于之前的信息进行分析。"},
        *previous_messages
    ])

    return {"messages": [response]}
```

---

## 四、真实企业场景：智能招聘系统

```python
class RecruitmentState(TypedDict):
    messages: Annotated[list, add]
    job_description: str
    candidate_resume: str
    screening_result: dict | None
    interview_questions: list[str] | None
    evaluation: dict | None

# 简历筛选 Agent
def resume_screener(state: RecruitmentState) -> dict:
    """分析简历与 JD 的匹配度"""
    prompt = f"""分析候选人简历与职位要求的匹配度。

职位要求：{state['job_description']}
候选人简历：{state['candidate_resume']}

请从以下维度评分（1-10）：
1. 技术技能匹配度
2. 工作经验匹配度
3. 教育背景匹配度
4. 综合推荐指数

输出 JSON 格式。"""

    result = llm.with_structured_output(ScreeningResult).invoke(prompt)
    return {
        "screening_result": result.model_dump(),
        "messages": [AIMessage(content=f"简历筛选完成，综合评分: {result.overall_score}/10")]
    }

# 面试题生成 Agent
def interview_designer(state: RecruitmentState) -> dict:
    """根据简历和 JD 生成针对性面试题"""
    screening = state["screening_result"]
    prompt = f"""基于简历筛选结果，生成 5 个针对性面试问题。

重点考察候选人的薄弱环节：
{screening}

要求：
1. 技术深度题 2 个
2. 项目经验题 2 个
3. 软技能/文化匹配题 1 个"""

    result = llm.invoke(prompt)
    questions = result.content.strip().split("\n")
    return {
        "interview_questions": questions,
        "messages": [AIMessage(content=f"已生成 {len(questions)} 个面试问题")]
    }

# 综合评估 Agent
def final_evaluator(state: RecruitmentState) -> dict:
    """综合评估，给出录用建议"""
    prompt = f"""综合以下信息，给出录用建议：

简历筛选结果：{state['screening_result']}
面试问题：{state['interview_questions']}

请给出：
1. 录用建议（推荐/待定/不推荐）
2. 建议薪资范围
3. 关键风险点
4. 入职后重点培养方向"""

    result = llm.invoke(prompt)
    return {
        "evaluation": {"recommendation": result.content},
        "messages": [AIMessage(content=result.content)]
    }

# 构建招聘流程图
graph = StateGraph(RecruitmentState)
graph.add_node("screener", resume_screener)
graph.add_node("interviewer", interview_designer)
graph.add_node("evaluator", final_evaluator)

graph.add_edge(START, "screener")

def after_screening(state: RecruitmentState) -> str:
    score = state["screening_result"]["overall_score"]
    if score >= 6:
        return "interviewer"  # 达标，进入面试环节
    return "evaluator"        # 不达标，直接评估（不推荐）

graph.add_conditional_edges("screener", after_screening)
graph.add_edge("interviewer", "evaluator")
graph.add_edge("evaluator", END)

recruitment_app = graph.compile()
```

---

## 五、多 Agent 架构选型指南

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| Supervisor | 任务类型明确，需要集中调度 | 控制力强，易于监控 | Supervisor 是瓶颈 |
| Swarm | 任务边界模糊，需要灵活协作 | 灵活，去中心化 | 难以追踪，可能死循环 |
| 层级 | 大型复杂系统 | 可扩展，职责清晰 | 架构复杂，延迟高 |
| 顺序流水线 | 固定流程，步骤明确 | 简单可靠 | 不灵活 |
| 并行扇出 | 多个独立子任务 | 高效，并行执行 | 需要结果合并逻辑 |

---

## 六、本章面试要点

1. **什么时候该用多 Agent？什么时候单 Agent 就够了？**
   → 单一领域、工具 <10 个 → 单 Agent；跨领域、工具多、流程复杂 → 多 Agent

2. **Supervisor 模式和 Swarm 模式的核心区别？**
   → Supervisor 有中央调度，控制力强但有瓶颈；Swarm 去中心化，灵活但难追踪

3. **多 Agent 之间怎么通信？**
   → 共享状态（State）或消息传递（Messages），LangGraph 中通过 StateGraph 的状态流转实现

4. **多 Agent 系统的常见问题？**
   → 死循环（Agent 互相踢皮球）、上下文丢失、延迟累积、调试困难

5. **如何防止多 Agent 死循环？**
   → 设置最大迭代次数、超时机制、Supervisor 兜底、状态中记录已访问的 Agent

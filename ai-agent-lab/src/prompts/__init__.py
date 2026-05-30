"""
=== Prompt 管理系统 — 含安全防护（输入校验 + Prompt 注入防护 + 输出过滤）===

【功能】：
1. 集中管理所有 Agent 使用的 Prompt 模板
2. 支持多语言、多场景的 Prompt 变体
3. 提供 Prompt 版本管理和热更新
4. 内置安全防护：输入校验、Prompt 注入防护、输出过滤

【设计原则】：
1. 模板化：使用 Jinja2 或 f-string 模板
2. 可配置：支持动态参数注入
3. 可测试：每个 Prompt 都有对应的测试用例
4. 可维护：按功能模块组织 Prompt
5. 安全性：内置多重安全防护

【Prompt 分类】：
1. Agent 技术助手 Prompt
2. Supervisor 路由 Prompt
3. 旅游助手 Prompt
4. 工具调用 Prompt
5. 安全防护函数

【使用方式】：
```python
from src.prompts import AGENT_PROMPT, SUPERVISOR_PROMPT, TRAVEL_TECH_PROMPT
from src.prompts import sanitize_input, sanitize_output

# 使用 Prompt 模板
prompt_messages = AGENT_PROMPT.invoke({"messages": [HumanMessage(content="问题")]})

# 安全处理输入输出
safe_input = sanitize_input(user_input)
safe_output = sanitize_output(agent_response)
```
"""

# 导入各个 Prompt 模块
from .supervisor import COMPLEXITY_EVALUATION_PROMPT, ROUTER_PROMPT, TASK_DECOMPOSITION_PROMPT
from .agent_tech import AGENT_PROMPT
from .agent_plan import PLAN_PROMPT
from .agent_sights import SIGHTS_PROMPT
from .agent_food import FOOD_PROMPT
from .agent_transport import TRANSPORT_PROMPT
from .security import sanitize_input, sanitize_output

# 导出所有公共接口
__all__ = [
    "ROUTER_PROMPT",
    "COMPLEXITY_EVALUATION_PROMPT",
    "TASK_DECOMPOSITION_PROMPT",
    "AGENT_PROMPT",
    "PLAN_PROMPT",
    "SIGHTS_PROMPT",
    "FOOD_PROMPT",
    "TRANSPORT_PROMPT",
    "sanitize_input",
    "sanitize_output",
]
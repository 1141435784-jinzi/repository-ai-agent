"""
=== Agent Workflow 示例 ===

演示如何直接使用 Agent Workflow
"""

import asyncio
from src.agents.workflow import get_async_agent
from langchain_core.messages import HumanMessage


async def run_workflow_example():
    """运行 Workflow 示例"""
    print("=== Agent Workflow 示例 ===")
    
    # 获取工作流
    workflow = await get_async_agent()
    
    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="如何设计一个 RAG 系统？")],
        "execution_plan": [],
        "iteration_count": 0,
        "task_errors": []
    }
    
    # 运行工作流
    print("正在处理请求...")
    result = await workflow.ainvoke(input_data)
    
    # 输出结果
    print("\n处理完成！")
    print("=" * 50)
    print("最终响应:", result["messages"][-1].content)
    
    if "execution_plan" in result and result["execution_plan"]:
        print("\n执行计划:")
        for task in result["execution_plan"]:
            status = task.get("status", "unknown")
            print(f"- {task.get('task', '')} [{status}]")


if __name__ == "__main__":
    asyncio.run(run_workflow_example())

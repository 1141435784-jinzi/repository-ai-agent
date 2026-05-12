"""
=== Agent Workflow 示例 ===

演示如何直接使用 Agent Workflow
"""

from src.agents.workflow import create_workflow
from langchain_core.messages import HumanMessage


def run_workflow_example():
    """运行 Workflow 示例"""
    print("=== Agent Workflow 示例 ===")
    
    # 创建工作流
    workflow = create_workflow()
    
    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="如何设计一个 RAG 系统？")]
    }
    
    # 运行工作流
    print("正在处理请求...")
    result = workflow.invoke(input_data)
    
    # 输出结果
    print("\n处理完成！")
    print("=" * 50)
    print("最终响应:", result["messages"][-1].content)
    
    if "rag_sources" in result and result["rag_sources"]:
        print("\nRAG 来源:")
        for source in result["rag_sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    run_workflow_example()
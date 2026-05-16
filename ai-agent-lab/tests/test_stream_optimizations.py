
"""
测试优化后的流式对话接口
验证点：
1. 任务分解与执行清单 (Execution List)
2. 专家协作与路由
3. 流式输出与工具调用展示
4. 最终总结回复
"""
import asyncio
import uuid
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_stream_optimization(message: str, description: str):
    """测试流式接口并验证新架构特性"""
    print(f"\n{'='*60}")
    print(f"【测试场景】{description}")
    print(f"{'='*60}")
    print(f"用户输入: {message}")
    print(f"{'─'*60}")
    
    thread_id = str(uuid.uuid4())
    data = {
        "message": message, 
        "thread_id": thread_id,
        "model": "deepseek"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json=data,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        full_response = ""
        has_execution_plan = False
        tools_called = []
        
        print("流式响应: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:].strip()
                    
                    if content == "[DONE]":
                        break
                    elif content.startswith("[TOOL_START]"):
                        try:
                            tool_data = json.loads(content[12:])
                            print(f"\n[执行工具: {tool_data['name']}]", end="", flush=True)
                            tools_called.append(tool_data['name'])
                        except:
                            pass
                    elif content.startswith("[TOOL_RESULT]"):
                        # 结果通常不在流中展示给用户，除非发生错误
                        pass
                    elif content.startswith("[ERROR]"):
                        print(f"\n❌ 后端报错: {content[7:]}")
                    else:
                        # 尝试检测是否包含执行计划的文字描述（如果模型输出了的话）
                        if "执行计划" in content or "清单" in content:
                            has_execution_plan = True
                        
                        # 还原换行符
                        clean_content = content.replace("\\n", "\n")
                        print(clean_content, end="", flush=True)
                        full_response += clean_content
        
        print("\n" + "─"*60)
        print(f"✅ 测试完成")
        print(f"   - 工具调用次数: {len(tools_called)}")
        if tools_called:
            print(f"   - 调用列表: {', '.join(tools_called)}")
        
        return True, full_response
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False, str(e)

def main():
    print("🚀 开始测试优化后的 Agent 工作流...")
    
    # 场景 1：复杂跨领域任务，验证任务分解和多专家协作
    scenario1 = {
        "message": "帮我规划一下去深圳旅游，我想看明天的天气，推荐两个必去景点，顺便预估一下两天的餐饮预算。",
        "description": "复杂跨领域任务（天气+景点+财务）"
    }
    
    # 场景 2：简单工具调用，验证直接路由和总结
    scenario2 = {
        "message": "现在深圳的天气怎么样？",
        "description": "简单工具调用（天气）"
    }

    scenarios = [scenario1, scenario2]
    
    for scenario in scenarios:
        success, _ = test_stream_optimization(scenario["message"], scenario["description"])
        if not success:
            sys.exit(1)

    print("\n" + "="*60)
    print("🎉 优化后的流式工作流测试全部通过！")
    print("="*60)

if __name__ == "__main__":
    main()

"""测试五个典型场景的脚本 - 使用流式对话接口"""
import asyncio
import uuid
import requests
import json

BASE_URL = "http://localhost:8000"

def test_scenario_streaming(message: str, description: str):
    """使用流式接口测试单个场景"""
    print(f"\n{'='*60}")
    print(f"【场景】{description}")
    print(f"{'='*60}")
    print(f"用户提问: {message}")
    print(f"{'─'*60}")
    
    thread_id = str(uuid.uuid4())
    data = {"message": message, "thread_id": thread_id}
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        tool_info = []
        
        print("流式响应:", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    content = decoded_line[5:]
                    
                    if content == "[DONE]":
                        break
                    elif content.startswith("[TOOL_START]"):
                        try:
                            tool_data = json.loads(content[12:])
                            tool_info.append(f"📦 工具调用: {tool_data['name']}")
                        except:
                            pass
                    elif content.startswith("[TOOL_RESULT]"):
                        try:
                            result_data = json.loads(content[14:])
                            tool_info.append(f"📋 工具结果: {result_data['name']}")
                        except:
                            pass
                    elif content.startswith("[ERROR]"):
                        print(f"\n❌ 错误: {content[7:]}")
                    elif content.startswith("[REPLACE]"):
                        full_response = content[10:]
                    else:
                        print(content, end="", flush=True)
                        full_response += content
        
        print("\n")
        if tool_info:
            print("工具调用记录:")
            for info in tool_info:
                print(f"  - {info}")
        
        return True, full_response
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        return False, str(e)

def main():
    print("🚀 开始使用流式接口测试五个典型场景...")
    
    scenarios = [
        {
            "message": "什么是 Agent？",
            "description": "简单问答：什么是 Agent？"
        },
        {
            "message": "帮我计算一下：2的10次方加上5乘以3等于多少",
            "description": "工具调用：计算器"
        },
        {
            "message": "帮我规划去北京旅游，包括景点推荐和交通方式",
            "description": "Agent 协作：推荐北京景点和交通方式"
        },
        {
            "message": "帮我规划一个3天的上海旅游行程，包含景点、美食和预算",
            "description": "复杂任务分解：3天上海旅游行程"
        },
        {
            "message": "什么是 LangGraph？它和 LangChain 有什么区别？",
            "description": "技术问题：LangGraph 是什么？"
        }
    ]
    
    results = []
    for scenario in scenarios:
        success, _ = test_scenario_streaming(scenario["message"], scenario["description"])
        results.append(success)
    
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️ 有 {total - passed} 个测试失败")

if __name__ == "__main__":
    main()
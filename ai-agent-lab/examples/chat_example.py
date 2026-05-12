"""
=== Agent Lab 聊天示例 ===

演示如何调用 Agent API 进行聊天
"""

import requests
import json


def chat_with_agent(message: str, session_id: str = None, stream: bool = False):
    """
    与 Agent 进行聊天
    
    Args:
        message: 用户消息
        session_id: 会话ID（可选）
        stream: 是否使用流式响应
    
    Returns:
        响应结果
    """
    url = "http://localhost:8000/chat"
    
    if stream:
        url = "http://localhost:8000/chat/stream"
    
    payload = {
        "message": message,
        "stream": stream
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        if stream:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    print(data.get("chunk", ""), end="")
            print()
            return None
        else:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def create_new_session():
    """创建新会话"""
    url = "http://localhost:8000/session/new"
    
    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json().get("session_id")
    except requests.exceptions.RequestException as e:
        print(f"创建会话失败: {e}")
        return None


if __name__ == "__main__":
    print("=== Agent Lab 聊天示例 ===")
    
    # 创建会话
    session_id = create_new_session()
    print(f"创建会话成功: {session_id}")
    
    # 发送消息
    response = chat_with_agent("你好，我想了解一下 RAG 系统", session_id)
    print("\n响应结果:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
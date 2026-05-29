"""测试流式响应"""
import requests
import json
import time

def test_stream():
    # 创建 session
    session_res = requests.post(
        "http://localhost:8000/chat/session/new",
        json={"user_id": "test_user"}
    )
    session_data = session_res.json()
    thread_id = session_data["thread_id"]
    print(f"Session created: {thread_id}")

    # 测试流式接口
    start_time = time.time()
    last_print_time = start_time
    chunk_count = 0

    print("\n开始流式测试...")
    print("-" * 50)

    res = requests.post(
        "http://localhost:8000/chat/stream",
        json={"message": "你好，请介绍一下你自己", "user_id": "test_user", "thread_id": thread_id},
        stream=True
    )

    full_text = ""

    for chunk in res.iter_content(chunk_size=None):
        if chunk:
            current_time = time.time()
            chunk_count += 1
            elapsed = current_time - start_time

            # 解析 SSE 数据
            text = chunk.decode('utf-8')
            print(f"[{elapsed:.2f}s] 第{chunk_count}个数据包: {text[:100]}...")

            # 尝试提取实际内容
            if text.startswith('data: '):
                data_content = text[6:].strip()
                if data_content and data_content != '[DONE]':
                    full_text += data_content

    print("-" * 50)
    print(f"流式结束！总耗时: {time.time() - start_time:.2f}s")
    print(f"总数据包数: {chunk_count}")
    print(f"总内容长度: {len(full_text)} 字符")
    print(f"完整内容:\n{full_text[:500]}...")

if __name__ == "__main__":
    test_stream()

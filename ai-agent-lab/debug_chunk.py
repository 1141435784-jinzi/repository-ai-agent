"""调试 astream_events 事件结构"""
import asyncio
import json
from src.agents.workflow import get_async_graph


async def debug_events():
    print("=" * 80)
    print("🔍 调试 astream_events 事件结构")
    print("=" * 80)

    try:
        graph_executor = await get_async_graph()
        config = {"recursion_limit": 100}

        print("\n📡 发送测试消息...")
        user_message = "你好"

        event_count = 0
        stream_count = 0
        all_event_types = set()
        chunk_structures = []

        async for event in graph_executor.astream_events(
            {"messages": [("user", user_message)]},
            config=config,
            version="v1"
        ):
            event_count += 1
            event_type = event.get("event", "???")
            all_event_types.add(event_type)

            if event_type == "on_chat_model_stream":
                stream_count += 1
                chunk_data = event.get("data", {}).get("chunk", {})
                
                # 记录 chunk 结构
                structure_info = {
                    "event_num": event_count,
                    "type": type(chunk_data).__name__,
                    "has_content_attr": hasattr(chunk_data, 'content'),
                    "is_dict": isinstance(chunk_data, dict),
                    "dict_keys": list(chunk_data.keys()) if isinstance(chunk_data, dict) else [],
                    "content_type": type(chunk_data.content).__name__ if hasattr(chunk_data, 'content') else None,
                    "content_value": str(chunk_data.content)[:50] if hasattr(chunk_data, 'content') else None
                }
                chunk_structures.append(structure_info)

                # 每10个事件打印一次
                if stream_count <= 5 or stream_count % 10 == 0:
                    print(f"\n📦 Stream Event #{stream_count}")
                    print(f"   Type: {type(chunk_data).__name__}")
                    print(f"   Has content attr: {hasattr(chunk_data, 'content')}")
                    if hasattr(chunk_data, 'content'):
                        print(f"   Content type: {type(chunk_data.content).__name__}")
                        print(f"   Content: {repr(str(chunk_data.content)[:100])}")
                    if isinstance(chunk_data, dict):
                        print(f"   Dict keys: {list(chunk_data.keys())}")
                        if 'delta' in chunk_data:
                            print(f"   delta: {repr(chunk_data['delta'])}")

        print("\n" + "=" * 80)
        print(f"📊 统计结果:")
        print(f"   总事件数: {event_count}")
        print(f"   Stream 事件数: {stream_count}")
        print(f"   事件类型: {sorted(all_event_types)}")
        
        # 输出 chunk 结构汇总
        print("\n📋 Chunk 结构汇总:")
        for i, struct in enumerate(chunk_structures[:5]):
            print(f"   {i+1}. {struct}")

    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        print(f"\n详细堆栈:")
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(debug_events())

# FastAPI 与异步编程实战

> 项目应用：api.py 使用 FastAPI + SSE 流式输出 + 异步 Agent 调用
> 整理来源：基于 [FastAPI 官方文档](https://fastapi.tiangolo.com/)、[Async Streaming Responses in FastAPI](https://dasroot.net/posts/2026/03/async-streaming-responses-fastapi-comprehensive-guide/)、[Server-Sent Events with FastAPI](https://medium.com/@nandagopal05/server-sent-events-with-python-fastapi-f1960e0c8e4b) 归纳改写
> 最后更新：2026 年 4 月

---

## 一、FastAPI 是什么

FastAPI 是一个现代的 Python Web 框架，基于 Starlette（ASGI 框架）和 Pydantic（数据验证）构建。它是当前 Python 生态中构建 AI/LLM API 服务的首选框架。

**核心优势**：
- **异步优先**：原生支持 `async/await`，适合 I/O 密集型的 LLM 调用
- **自动文档**：自动生成 Swagger UI（`/docs`）和 ReDoc（`/redoc`）
- **类型安全**：基于 Python type hints + Pydantic 自动验证请求/响应
- **高性能**：性能接近 Node.js 和 Go，远超 Flask/Django

---

## 二、同步 vs 异步：为什么 AI 服务必须用异步

### 2.1 同步模式的问题

```python
# 同步：一个请求阻塞整个线程
@app.post("/chat")
def chat(request: ChatRequest):
    result = agent.invoke(...)  # 等待 3-10 秒
    return result               # 这期间其他请求全部排队
```

LLM 调用通常需要 3-10 秒，同步模式下一个请求就会阻塞一个线程。10 个并发用户就需要 10 个线程，100 个并发就需要 100 个线程——资源浪费严重。

### 2.2 异步模式的优势

```python
# 异步：等待 LLM 时释放线程去处理其他请求
@app.post("/chat")
async def chat(request: ChatRequest):
    result = await agent.ainvoke(...)  # 等待时线程去处理其他请求
    return result                       # 一个线程能处理大量并发
```

异步模式下，当一个请求在等待 LLM 响应时，线程会去处理其他请求。一个线程就能处理成百上千的并发连接。

### 2.3 直觉理解

- **同步**：餐厅只有一个服务员，点完菜后站在厨房门口等，其他桌的客人干等着
- **异步**：服务员点完菜后去服务其他桌，菜好了再端过来

---

## 三、Python 异步编程基础

### 3.1 核心概念

```python
import asyncio

# async def 定义协程函数
async def fetch_data():
    await asyncio.sleep(1)  # 模拟 I/O 等待
    return "data"

# await 等待协程完成（期间释放控制权）
async def main():
    result = await fetch_data()
    print(result)

# 运行
asyncio.run(main())
```

### 3.2 并发执行多个协程

```python
async def main():
    # 串行：总耗时 = 3 秒
    result1 = await fetch_data()  # 1 秒
    result2 = await fetch_data()  # 1 秒
    result3 = await fetch_data()  # 1 秒

    # 并发：总耗时 = 1 秒
    result1, result2, result3 = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data(),
    )
```

### 3.3 async for：异步迭代

```python
async def stream_tokens():
    """模拟 LLM 流式输出"""
    for token in ["你", "好", "，", "世", "界"]:
        await asyncio.sleep(0.1)
        yield token

async def main():
    async for token in stream_tokens():
        print(token, end="", flush=True)
```

---

## 四、FastAPI 核心用法

### 4.1 基本路由

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Agent API", version="1.0")

# 请求模型（自动验证）
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

# 响应模型
class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []

# 异步路由
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await agent.ainvoke(
        {"messages": [("user", request.message)]},
        {"configurable": {"thread_id": request.thread_id}}
    )
    return ChatResponse(
        response=result["messages"][-1].content,
        sources=result.get("rag_sources", [])
    )
```

### 4.2 生命周期事件（预加载资源）

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时预加载重型资源"""
    # 启动时执行
    print("加载 Embedding 模型...")
    load_embedding_model()
    print("加载 RAG 引擎...")
    init_rag_engine()
    yield
    # 关闭时执行
    print("清理资源...")

app = FastAPI(lifespan=lifespan)
```

这确保第一个用户请求就能毫秒级响应，符合项目的预加载原则。

### 4.3 依赖注入

```python
from fastapi import Depends

async def get_agent():
    """获取异步 Agent 实例"""
    return await get_async_agent()

@app.post("/chat")
async def chat(request: ChatRequest, agent=Depends(get_agent)):
    result = await agent.ainvoke(...)
    return result
```

---

## 五、SSE 流式输出

### 5.1 什么是 SSE

SSE（Server-Sent Events）是一种服务器向客户端单向推送数据的协议。与 WebSocket 的区别：

| 特性 | SSE | WebSocket |
|---|---|---|
| 方向 | 服务器 → 客户端（单向） | 双向 |
| 协议 | HTTP | 独立协议 |
| 复杂度 | 简单 | 复杂 |
| 重连 | 自动重连 | 需手动实现 |
| 适用场景 | LLM 流式输出、实时通知 | 聊天室、游戏 |

LLM 流式输出是典型的单向推送场景，SSE 是最合适的选择。

### 5.2 FastAPI 中实现 SSE

```python
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式输出"""

    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}

        async for event in agent.astream(
            {"messages": [("user", request.message)]},
            config,
            stream_mode="messages"
        ):
            # 每个 token 作为一个 SSE 事件发送
            yield {
                "event": "token",
                "data": json.dumps({
                    "content": event.content,
                    "type": event.type
                }, ensure_ascii=False)
            }

        # 发送结束信号
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
```

### 5.3 前端消费 SSE

```javascript
const eventSource = new EventSource("/chat/stream", {
    method: "POST",
    body: JSON.stringify({ message: "你好", thread_id: "user-1" })
});

eventSource.addEventListener("token", (event) => {
    const data = JSON.parse(event.data);
    document.getElementById("output").textContent += data.content;
});

eventSource.addEventListener("done", () => {
    eventSource.close();
});
```

---

## 六、错误处理

### 6.1 全局异常处理

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "服务内部错误", "detail": str(exc)}
    )
```

### 6.2 自定义业务异常

```python
from fastapi import HTTPException

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        result = await agent.ainvoke(...)
        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent 响应超时")
```

---

## 七、CORS 跨域配置

前端和后端不在同一个域时需要配置 CORS：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 八、在 agent-lab 项目中的应用

你项目的 api.py 采用的架构：

```
客户端（static/index.html）
    ↓ HTTP POST / SSE
FastAPI（api.py）
    ↓ await agent.ainvoke() / agent.astream()
异步 Agent（agent.py 的 get_async_agent）
    ↓ AsyncSqliteSaver（memory.py）
LangGraph StateGraph
```

关键设计决策：
- **api.py 全部使用异步**：`async def` + `await agent.ainvoke()`
- **main.py 使用同步**：`agent.invoke()` + `SqliteSaver`（命令行调试用）
- **Checkpointer 分离**：同步/异步版统一在 memory.py 管理
- **预加载**：Embedding 模型、向量数据库在服务启动时加载

---

## 九、生产部署建议

### 9.1 启动命令

```bash
# 开发环境
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 生产环境（多 worker）
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn + uvicorn worker
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 9.2 性能调优

| 参数 | 建议 |
|---|---|
| workers | CPU 核心数 × 2 + 1 |
| 超时时间 | LLM 调用慢，设置 120-300 秒 |
| 连接限制 | 根据内存设置最大并发连接数 |
| Keep-Alive | SSE 场景需要较长的 keep-alive 时间 |

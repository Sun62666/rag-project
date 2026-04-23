import json,redis,os
from fastapi import FastAPI,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from src.config import Config
from src.retriever import OpsRetriever
from src.graph import build_graph

app = FastAPI(title="SmartOps API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
print("创建配置中。。。。")
cfg = Config()
pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","文档2.pdf")

print("\n初始话retriever中。。。。")
retriever = OpsRetriever(pdf_path)
print("\n构建图中。。。。")
graph = build_graph(retriever)
print("\n查看redis缓存中。。。。")
cache = redis.from_url(cfg.REDIS_URL,decode_responses=True)

print("修复了bug")
class Query(BaseModel):
    query: str
# def get_session_history(user_id,session_id):
#     session_id = user_id + "_" + session_id
#     return RedisChatMessageHistory(session_id,url=cfg.REDIS_URL)
# graph = RunnableWithMessageHistory(
#     graph,
#     get_session_history,
#     input_messages_key="query",
#     history_messages_key="context"
# )
@app.post("/ask")
async def ask(req:Query,bg_tasks: BackgroundTasks):
    try:
        cached = cache.get(f"ops:{req.query}")
        if cached:
            print("="*60)
            print(f"answer:{cached},from_cache:True")

            async def cached_stream():
                yield f"data: {json.dumps({'type':'status','message':'从缓存中获取'})}\n\n"
                for char in cached:
                    yield f"data: {json.dumps({'type':'token','content':char})}\n\n"
                print("\n 已完成缓存输出！")
                yield f"data: {json.dumps({'type':'done','from_cache':True})}\n\n"
            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"

                }
            )
    except Exception as e:
        print(f"Redis缓存异常： {e}")
        raise ValueError(f"Redis缓存异常： {e}")

    async def stream_gen():
        try:
            state = {"query":req.query,"context":[],"answer":""}
            full_answer = []

            yield f"data: {json.dumps({'type': 'status', 'message': '正在检索知识库...'})}\n\n"
            async for msg,metadata in graph.astream(state,stream_mode="messages"):
                # print(f"\nmsg: {msg}")
                # print(f"\nmetadata: {metadata}")
                if metadata.get("langgraph_node") == "generate" and isinstance(msg,AIMessage): #and msg.type == "ai"
                    token = msg.content
                    if token:
                        full_answer.append(token)
                        # print(f"token: {token}")
                        # print(f"full_answer: {full_answer}")
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            answer_text = "".join(full_answer)
            print(f"\nanswer_text: {answer_text}")
            bg_tasks.add_task(cache.setex,f"ops:{req.query}",cfg.CACHE_TTL,answer_text)
            yield f"data: {json.dumps({'type': 'done','from_cache':False})}\n\n"
        except Exception as e:
            # 向前端返回错误
            yield f"data: {json.dumps({'type': 'status', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# SSE(Server-Sent Events) 是一种单向实时通信协议，允许服务器主动向客户端推送数据
# 数据格式位 data: <消息内容>\n\n
# stream_mode    "updates"            "values"            "messages"
# 返回内容        每个节点的完整输出        完整的状态快照        LLM 的每个 token
# 流式时机        节点执行完后才返回        节点执行完后才返回     ✅ LLM 每生成一个 token 就返回
#








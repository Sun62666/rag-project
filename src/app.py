import json, redis, os, uuid
import logging
import sys
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel
from src.config import Config
from src.retriever import OpsRetriever
from src.graph import build_graph

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"

)
logger = logging.getLogger(__name__)

# class PrintToLogger:
#     _in_write = False
#     def write(self,msg):
#         if PrintToLogger._in_write:
#             return
#         PrintToLogger._in_write = True
#         try:
#             msg = msg.strip()
#             if msg:
#                 logger.info(msg)
#         finally:
#             PrintToLogger._in_write = False
#     def flush(self):
#         """必须实现flush方法，否则sys.stdout重定向会报错"""
#         pass
#
# sys.stdout = PrintToLogger()
# sys.stderr = PrintToLogger()
FALLBACK_MESSAGE = "当前知识库未覆盖该问题，建议转交人工运维专家。"

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
pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "文档2.pdf")

print("\n初始化retriever中。。。。")
retriever = OpsRetriever(pdf_path)
print("\n构建图中。。。。")
graph = build_graph(retriever)
print("\n查看redis缓存中。。。。")
cache = redis.from_url(cfg.REDIS_URL, decode_responses=True)


class Query(BaseModel):
    query: str
    session_id: str = "default"


def get_chat_history(session_id: str, limit: int = 6):
    try:
        history_key = f"chat_history:{session_id}"
        raw = cache.lrange(history_key, -limit, -1)
        return [json.loads(h) for h in raw] if raw else []
    except Exception:
        return []


def save_chat_history(session_id: str, user_msg: str, assistant_msg: str):
    try:
        history_key = f"chat_history:{session_id}"
        cache.rpush(history_key, json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False))
        cache.rpush(history_key, json.dumps({"role": "assistant", "content": assistant_msg}, ensure_ascii=False))
        cache.expire(history_key, cfg.CACHE_TTL)
    except Exception as e:
        print(f"保存对话历史失败: {e}")


@app.post("/ask")
async def ask(req: Query, bg_tasks: BackgroundTasks):
    try:
        cached = cache.get(f"ops:{req.query}")
        if cached:
            print("=" * 60)
            print(f"answer:{cached}, from_cache:True")

            async def cached_stream():
                yield f"data: {json.dumps({'type': 'status', 'message': '从缓存中获取'})}\n\n"
                for char in cached:
                    yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                print("\n已完成缓存输出！")
                yield f"data: {json.dumps({'type': 'done', 'from_cache': True})}\n\n"

            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )
    except Exception as e:
        print(f"Redis缓存异常：{e}")

    chat_history = get_chat_history(req.session_id)
    history_strs = [f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in chat_history]

    async def stream_gen():
        try:
            state = {
                "query": req.query,
                "intent": "",
                "rewritten_query": "",
                "retrieved_context": "",
                "tool_results": {},
                "answer": "",
                "chat_history": history_strs
            }
            full_answer = []

            yield f"data: {json.dumps({'type': 'status', 'message': '正在分析问题...'})}\n\n"

            async for msg, metadata in graph.astream(state, stream_mode="messages"):
                node = metadata.get("langgraph_node")
                msg_type = type(msg).__name__
                print(f"msg_type: {msg_type}")
                print(f"metadata: {metadata}")
                if node == "classify":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在分类意图...'})}\n\n"
                elif node == "rewrite_query":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在优化检索词...'})}\n\n"
                elif node == "retrieve":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在检索知识库...'})}\n\n"
                elif node == "execute_tools":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在执行系统工具...'})}\n\n"
                elif node == "generate" and isinstance(msg, AIMessageChunk) and msg.content:
                    token = msg.content
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            answer_text = "".join(full_answer)

            if not answer_text.strip():
                print("\n[DEBUG] 流式未捕获答案，尝试直接获取...")
                final_state = await graph.ainvoke(state)
                answer_text = final_state.get("answer", "").strip()
                if answer_text:
                    for char in answer_text:
                        yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

            print(f"\nanswer_text: {answer_text}")

            if answer_text and answer_text != FALLBACK_MESSAGE:
                bg_tasks.add_task(cache.setex, f"ops:{req.query}", cfg.CACHE_TTL, answer_text)

            bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text)

            yield f"data: {json.dumps({'type': 'done', 'from_cache': False})}\n\n"
        except Exception as e:
            print(f"\n[ERROR] stream_gen 异常: {e}")
            yield f"data: {json.dumps({'type': 'status', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())[:8]
    return {"session_id": session_id}


@app.post("/clear_history")
async def clear_history(session_id: str = "default"):
    try:
        cache.delete(f"chat_history:{session_id}")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

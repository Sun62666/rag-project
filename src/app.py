import json, redis, os, uuid
import logging
import sys
from datetime import datetime
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

logger.info("创建配置中。。。。")
cfg = Config()
pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "文档2.pdf")

logger.info("初始化retriever中。。。。")
retriever = OpsRetriever(pdf_path)

logger.info("构建图中。。。。")
graph = build_graph(retriever)

logger.info("查看redis缓存中。。。。")
cache = redis.from_url(cfg.REDIS_URL, decode_responses=True)


class Query(BaseModel):
    query: str
    session_id: str = "default"


def get_chat_history(session_id: str, limit: int = 0):
    try:
        history_key = f"chat_history:{session_id}"
        if limit > 0:
            raw = cache.lrange(history_key, -limit, -1)
        else:
            raw = cache.lrange(history_key, 0, -1)
        return [json.loads(h) for h in raw] if raw else []
    except Exception:
        return []


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _session_key(sid: str) -> str:
    return f"session:{sid}"

SESSIONS_ZSET = "sessions_list"

def _get_all_sessions() -> list:
    try:
        ids = cache.zrevrange(SESSIONS_ZSET, 0, -1)
        if not ids:
            return []
        # result = []
        # for sid in ids:
        #     data = cache.hgetall(_session_key(sid))
        #     if data:
        #         result.append(data)
        # 优化：使用Pipeline批量获取
        pipe = cache.pipeline()
        for sid in ids:
            pipe.hgetall(f"session:{sid}")
        results = pipe.execute()  # 一次网络往返获取所有数据
        return [r for r in results if r]
    except Exception:
        return []


def save_chat_history(session_id: str, user_msg: str, assistant_msg: str):
    try:
        history_key = f"chat_history:{session_id}"
        history_len = cache.llen(history_key)
        cache.rpush(history_key, json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False))
        cache.rpush(history_key, json.dumps({"role": "assistant", "content": assistant_msg}, ensure_ascii=False))
        cache.expire(history_key, cfg.CACHE_TTL_SHOT)

        if history_len == 0:
            title = user_msg[:20] if len(user_msg) > 20 else user_msg
            cache.hset(_session_key(session_id), mapping={
                "session_id": session_id,
                "title": title,
                "updated_at": _now(),
            })
            cache.expire(_session_key(session_id), cfg.CACHE_TTL_LONG)
    except Exception as e:
        logger.error(f"保存对话历史失败: {e}")


@app.post("/ask")
async def ask(req: Query, bg_tasks: BackgroundTasks):
    try:
        cached = cache.get(f"ops:{req.query}")
        if cached:
            print("=" * 60)
            logger.info(f"answer:{cached}, from_cache:True")

            async def cached_stream():
                yield f"data: {json.dumps({'type': 'status', 'message': '从缓存中获取'})}\n\n"
                for char in cached:
                    yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                logger.info("\n已完成缓存输出！")
                yield f"data: {json.dumps({'type': 'done', 'from_cache': True})}\n\n"

            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )
    except Exception as e:
        logger.error(f"Redis缓存异常：{e}")

    chat_history = get_chat_history(req.session_id)
    history_strs = [f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in chat_history]
    logger.info(f"历史消息： {history_strs}")
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
                # msg_type = type(msg).__name__
                # logger.info(f"msg_type: {msg_type}")
                # logger.info(f"node: {metadata.get("langgraph_node")}")
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
                logger.debug("\n 流式未捕获答案，尝试直接获取...")
                final_state = await graph.ainvoke(state)
                answer_text = final_state.get("answer", "").strip()
                if answer_text:
                    for char in answer_text:
                        yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

            logger.info(f"answer_text: {answer_text}")

            if answer_text and answer_text != FALLBACK_MESSAGE:
                bg_tasks.add_task(cache.setex, f"ops:{req.query}", cfg.CACHE_TTL_SHOT, answer_text)

            bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text)

            yield f"data: {json.dumps({'type': 'done', 'from_cache': False})}\n\n"
        except Exception as e:
            logger.error(f"\n stream_gen 异常: {e}")
            yield f"data: {json.dumps({'type': 'status', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/new_session")
async def new_session():
    session_id = str(uuid.uuid4())[:8]
    now = _now()
    cache.hset(_session_key(session_id), mapping={
        "session_id": session_id,
        "title": "新会话",
        "created_at": now,
        "updated_at": now,
    })
    cache.expire(_session_key(session_id), cfg.CACHE_TTL_LONG)
    cache.zadd(SESSIONS_ZSET, {session_id: datetime.now().timestamp()})
    cache.expire(SESSIONS_ZSET,cfg.CACHE_TTL_LONG)
    return {"session_id": session_id}


@app.post("/clear_history")
async def clear_history(session_id: str = "default"):
    try:
        cache.delete(f"chat_history:{session_id}")
        cache.hset(_session_key(session_id), mapping={
            "title": "新会话",
            "updated_at": _now(),
        })
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/sessions")
async def get_sessions():
    try:
        sessions = _get_all_sessions()
        return {"status": "ok", "sessions": sessions}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        cache.delete(f"chat_history:{session_id}")
        cache.delete(_session_key(session_id))
        cache.zrem(SESSIONS_ZSET, session_id)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    try:
        history = get_chat_history(session_id)
        return {"status": "ok", "history": history}
    except Exception as e:
        return {"status": "error", "message": str(e)}

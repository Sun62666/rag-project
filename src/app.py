import json, redis, os, uuid, hashlib, secrets
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel
from src.config import Config
from src.retriever import OpsRetriever
from src.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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

class UserAuth(BaseModel):
    username: str
    password: str

class RenameSession(BaseModel):
    title: str


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _session_key(sid: str) -> str:
    return f"session:{sid}"

def _user_sessions_key(username: str) -> str:
    return f"user_sessions:{username}"

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()

def _get_current_user(authorization: Optional[str] = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    token = authorization[7:]
    username = cache.get(f"token:{token}")
    return username or "anonymous"

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

def _get_user_sessions(username: str) -> list:
    try:
        zset_key = _user_sessions_key(username)
        ids = cache.zrevrange(zset_key, 0, -1)
        if not ids:
            return []
        pipe = cache.pipeline()
        for sid in ids:
            pipe.hgetall(f"session:{sid}")
        results = pipe.execute()
        return [r for r in results if r]
    except Exception:
        return []

def _session_queries_key(sid: str) -> str:
    return f"session_queries:{sid}"

def save_chat_history(session_id: str, user_msg: str, assistant_msg: str, username: str = "anonymous"):
    try:
        history_key = f"chat_history:{session_id}"
        history_len = cache.llen(history_key)
        cache.rpush(history_key, json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False))
        cache.rpush(history_key, json.dumps({"role": "assistant", "content": assistant_msg}, ensure_ascii=False))
        cache.expire(history_key, cfg.CACHE_TTL_LONG)
        cache.sadd(_session_queries_key(session_id), user_msg)
        cache.expire(_session_queries_key(session_id), cfg.CACHE_TTL_LONG)
        if history_len == 0:
            title = user_msg[:20] if len(user_msg) > 20 else user_msg
            cache.hset(_session_key(session_id), mapping={
                "session_id": session_id,
                "title": title,
                "updated_at": _now(),
            })
        cache.zadd(_user_sessions_key(username), {session_id: datetime.now().timestamp()})
        cache.expire(_session_key(session_id), cfg.CACHE_TTL_LONG)
    except Exception as e:
        logger.error(f"保存对话历史失败: {e}")


@app.post("/auth/register")
async def register(user: UserAuth):
    if not user.username.strip() or not user.password.strip():
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(user.username) < 2 or len(user.username) > 20:
        raise HTTPException(status_code=400, detail="用户名长度需2-20个字符")
    if len(user.password) < 4:
        raise HTTPException(status_code=400, detail="密码长度至少4个字符")
    user_key = f"user:{user.username}"
    if cache.exists(user_key):
        raise HTTPException(status_code=400, detail="用户名已存在")
    salt = secrets.token_hex(8) # 生成十六进制字符串16个（这是生成8个字节，一个字节8位，十六进制4位组成一个字符串）
    password_hash = _hash_password(user.password, salt)
    cache.hset(user_key, mapping={
        "username": user.username,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": _now(),
    })
    token = secrets.token_hex(32)
    cache.setex(f"token:{token}", 86400, user.username)
    return {"status": "ok", "token": token, "username": user.username}


@app.post("/auth/login")
async def login(user: UserAuth):
    user_key = f"user:{user.username}"
    if not cache.exists(user_key):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    salt = cache.hget(user_key, "salt")
    stored_hash = cache.hget(user_key, "password_hash")
    input_hash = _hash_password(user.password, salt)
    if input_hash != stored_hash:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = secrets.token_hex(32)
    cache.setex(f"token:{token}", 86400, user.username)
    return {"status": "ok", "token": token, "username": user.username}


@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        cache.delete(f"token:{token}")
    return {"status": "ok"}


@app.get("/auth/me")
async def get_me(authorization: Optional[str] = Header(None)):
    username = _get_current_user(authorization)
    if username == "anonymous":
        raise HTTPException(status_code=401, detail="未登录")
    return {"status": "ok", "username": username}


@app.post("/ask")
async def ask(req: Query, bg_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    username = _get_current_user(authorization)

    try:
        cached = cache.get(f"ops:{req.query}")
        if cached:
            logger.info(f"answer from cache: True")
            bg_tasks.add_task(save_chat_history, req.session_id, req.query, cached, username)

            async def cached_stream():
                yield f"data: {json.dumps({'type': 'status', 'message': '从缓存中获取'})}\n\n"
                for char in cached:
                    yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
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

            bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text, username)

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
async def new_session(authorization: Optional[str] = Header(None)):
    username = _get_current_user(authorization)
    session_id = str(uuid.uuid4())[:8]
    now = _now()
    cache.hset(_session_key(session_id), mapping={
        "session_id": session_id,
        "title": "新会话",
        "created_at": now,
        "updated_at": now,
        "user_id": username,
    })
    cache.expire(_session_key(session_id), cfg.CACHE_TTL_LONG)
    zset_key = _user_sessions_key(username)
    cache.zadd(zset_key, {session_id: datetime.now().timestamp()})
    cache.expire(zset_key, cfg.CACHE_TTL_LONG)
    return {"session_id": session_id}


@app.post("/clear_history")
async def clear_history(session_id: str = "default"):
    try:
        queries_key = _session_queries_key(session_id)
        queries = cache.smembers(queries_key)
        if queries:
            pipe = cache.pipeline()
            for q in queries:
                pipe.delete(f"ops:{q}")
            pipe.execute()
        pipe = cache.pipeline()
        pipe.delete(f"chat_history:{session_id}")
        pipe.delete(queries_key)
        pipe.execute()
        cache.hset(_session_key(session_id), mapping={
            "updated_at": _now(),
        })
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/sessions")
async def get_sessions(authorization: Optional[str] = Header(None)):
    username = _get_current_user(authorization)
    try:
        sessions = _get_user_sessions(username)
        return {"status": "ok", "sessions": sessions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        user_id = cache.hget(_session_key(session_id), "user_id")
        queries_key = _session_queries_key(session_id)
        queries = cache.smembers(queries_key)
        if queries:
            pipe = cache.pipeline()
            for q in queries:
                pipe.delete(f"ops:{q}")
            pipe.execute()
        pipe = cache.pipeline()
        pipe.delete(f"chat_history:{session_id}")
        pipe.delete(queries_key)
        pipe.delete(_session_key(session_id))
        pipe.execute()
        if user_id:
            cache.zrem(_user_sessions_key(user_id), session_id)
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


@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, body: RenameSession):
    try:
        if not body.title or not body.title.strip():
            return {"status": "error", "message": "标题不能为空"}
        cache.hset(_session_key(session_id), mapping={
            "title": body.title.strip()[:50],
            "updated_at": _now(),
        })
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

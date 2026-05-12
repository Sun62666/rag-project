import json
import logging
from datetime import datetime
from typing import List, Optional
from src.core.redis import get_cache
from src.core.config import get_settings

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _session_key(sid: str) -> str:
    return f"session:{sid}"


def _user_sessions_key(username: str) -> str:
    return f"user_sessions:{username}"


def _session_queries_key(sid: str) -> str:
    return f"session_queries:{sid}"


def create_session(username: str) -> dict:
    import uuid
    cache = get_cache()
    cfg = get_settings()
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


def get_user_sessions(username: str) -> list:
    cache = get_cache()
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


def get_chat_history(session_id: str, limit: int = 0) -> list:
    cache = get_cache()
    try:
        history_key = f"chat_history:{session_id}"
        if limit > 0:
            raw = cache.lrange(history_key, -limit, -1)
        else:
            raw = cache.lrange(history_key, 0, -1)
        return [json.loads(h) for h in raw] if raw else []
    except Exception:
        return []


def save_chat_history(session_id: str, user_msg: str, assistant_msg: str, username: str = "anonymous"):
    cache = get_cache()
    cfg = get_settings()
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


def clear_history(session_id: str, stm) -> dict:
    cache = get_cache()
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
        if stm:
            stm.clear(session_id)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_session(session_id: str, stm) -> dict:
    cache = get_cache()
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
        if stm:
            stm.clear(session_id)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def rename_session(session_id: str, title: str) -> dict:
    cache = get_cache()
    try:
        if not title or not title.strip():
            return {"status": "error", "message": "标题不能为空"}
        cache.hset(_session_key(session_id), mapping={
            "title": title.strip()[:50],
            "updated_at": _now(),
        })
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

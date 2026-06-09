import hashlib
import secrets
import logging
import redis as redis_lib
from src.core.redis import get_cache

logger = logging.getLogger(__name__)


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def get_current_user(authorization: str = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    token = authorization[7:]
    cache = get_cache()
    username = cache.get(f"token:{token}")
    return username or "anonymous"


def create_token(username: str) -> str:
    token = secrets.token_hex(32)
    cache = get_cache()
    cache.setex(f"token:{token}", 86400 * 7, username)
    return token


def verify_password(username: str, password: str) -> bool:
    cache = get_cache()
    user_key = f"user:{username}"
    if not cache.exists(user_key):
        return False
    try:
        key_type = cache.type(user_key)
        if key_type != "hash":
            # 旧数据是 string 类型，无法用 hget，删除损坏数据让用户重新注册
            logger.warning(f"[Auth] key '{user_key}' 类型为 {key_type}，预期 hash，已删除损坏数据")
            cache.delete(user_key)
            return False
        salt = cache.hget(user_key, "salt")
        stored_hash = cache.hget(user_key, "password_hash")
        if not salt or not stored_hash:
            return False
        input_hash = hash_password(password, salt)
        return input_hash == stored_hash
    except redis_lib.ResponseError as e:
        logger.error(f"[Auth] verify_password Redis 错误: {e}")
        return False
import hashlib
import secrets
from src.core.redis import get_cache


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
    salt = cache.hget(user_key, "salt")
    stored_hash = cache.hget(user_key, "password_hash")
    if not salt or not stored_hash:
        return False
    input_hash = hash_password(password, salt)
    return input_hash == stored_hash

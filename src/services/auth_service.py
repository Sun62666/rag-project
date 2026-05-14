import logging
from datetime import datetime
from src.core.redis import get_cache
from src.core.security import hash_password, create_token, verify_password

logger = logging.getLogger(__name__)


def register_user(username: str, password: str) -> dict:
    if not username.strip() or not password.strip():
        raise ValueError("用户名和密码不能为空")
    if len(username) < 2 or len(username) > 20:
        raise ValueError("用户名长度需2-20个字符")
    if len(password) < 4:
        raise ValueError("密码长度至少4个字符")

    cache = get_cache()
    user_key = f"user:{username}"
    if cache.exists(user_key):
        raise ValueError("用户名已存在")

    import secrets
    salt = secrets.token_hex(8)
    password_hash = hash_password(password, salt)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache.hset(user_key, mapping={
        "username": username,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": now,
    })
    token = create_token(username)
    return {"status": "ok", "token": token, "username": username}


def login_user(username: str, password: str) -> dict:
    cache = get_cache()
    user_key = f"user:{username}"
    if not cache.exists(user_key):
        raise ValueError("用户名或密码错误")
    if not verify_password(username, password):
        raise ValueError("用户名或密码错误")
    token = create_token(username)
    return {"status": "ok", "token": token, "username": username}


def logout_user(authorization: str = None):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        cache = get_cache()
        cache.delete(f"token:{token}")
    return {"status": "ok"}


def get_user_info(authorization: str = None) -> dict:
    from src.core.security import get_current_user
    username = get_current_user(authorization)
    if username == "anonymous":
        raise ValueError("未登录")
    return {"status": "ok", "username": username}

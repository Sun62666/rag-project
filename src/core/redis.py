import logging
import redis
from src.core.config import get_settings

_cache_instance = None


def get_cache() -> redis.Redis:
    global _cache_instance
    if _cache_instance is None:
        cfg = get_settings()
        _cache_instance = redis.from_url(cfg.REDIS_URL, decode_responses=True)
        logging.getLogger(__name__).info("[Redis] 连接成功")
    return _cache_instance

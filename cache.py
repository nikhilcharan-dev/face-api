import os
import redis.asyncio as aioredis

CACHE_TTL = 300  # 5 minutes

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )
    return _redis


def embedding_key(organization_id: str) -> str:
    return f"face:embeddings:{organization_id}"


async def cache_get(key: str) -> str | None:
    try:
        return await _get_redis().get(key)
    except Exception:
        return None


async def cache_set(key: str, value: str, ttl: int = CACHE_TTL) -> None:
    try:
        await _get_redis().setex(key, ttl, value)
    except Exception:
        pass


async def cache_del(key: str) -> None:
    try:
        await _get_redis().delete(key)
    except Exception:
        pass

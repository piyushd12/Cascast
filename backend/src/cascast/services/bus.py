"""Thin helpers over Redis Streams. Every message carries one orjson-encoded field `d`."""

from __future__ import annotations

from typing import Any

import numpy as np
import orjson
from redis.asyncio import Redis

from cascast.config import settings

MAXLEN = 2000  # approximate cap per stream; old entries are trimmed


def connect() -> Redis:
    return Redis.from_url(settings.redis_url)


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)


async def publish(r: Redis, stream: str, payload: dict) -> bytes:
    return await r.xadd(stream, {"d": dumps(payload)}, maxlen=MAXLEN, approximate=True)


async def latest(r: Redis, stream: str) -> dict | None:
    rows = await r.xrevrange(stream, count=1)
    return orjson.loads(rows[0][1][b"d"]) if rows else None


def decode(fields: dict) -> dict:
    return orjson.loads(fields[b"d"])


def r3(a: np.ndarray) -> list[float]:
    """Round for compact JSON on the wire."""
    return np.round(np.asarray(a, dtype=np.float64), 3).tolist()

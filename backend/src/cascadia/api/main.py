"""Cascadia API gateway: REST, WebSocket fan-out of the live streams, and the Copilot (SSE).

Redis and Postgres are optional: without them the API still serves the grid, what-if
analysis (on a nominal operating point) and model metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import orjson
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cascadia.api import copilot, ops
from cascadia.api.db import Store
from cascadia.config import COMMAND_STREAM, EVENT_STREAM, RISK_STREAM, STATE_STREAM, settings
from cascadia.grid.model import load_grid
from cascadia.ml.inference import get_model
from cascadia.services import bus

log = logging.getLogger("cascadia.api")


class Hub:
    """Live state + WebSocket subscribers. Slow clients drop messages instead of blocking."""

    def __init__(self):
        self.live = ops.LiveState()
        self.clients: set[asyncio.Queue] = set()
        self.redis = None
        self.store: Store | None = None

    def broadcast(self, kind: str, data: dict) -> None:
        msg = orjson.dumps({"type": kind, "data": data})
        for q in self.clients:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)

    async def pump(self) -> None:
        """Follow the Redis streams, update live state, persist and fan out."""
        r = self.redis
        cursors = {STATE_STREAM: "$", RISK_STREAM: "$", EVENT_STREAM: "$"}
        if latest := await bus.latest(r, STATE_STREAM):
            self.live.state, self.live.live = latest, True
        self.live.risk = await bus.latest(r, RISK_STREAM)
        for _, fields in reversed(await r.xrevrange(EVENT_STREAM, count=50)):
            self.live.events.append(bus.decode(fields))
        while True:
            try:
                rows = await r.xread(cursors, count=100, block=5000)
            except Exception as e:  # keep the API up if Redis blips
                log.warning("stream read failed: %s", e)
                await asyncio.sleep(2)
                continue
            for stream, msgs in rows:
                name = stream.decode()
                cursors[name] = msgs[-1][0]
                if name == STATE_STREAM:
                    self.live.state, self.live.live = bus.decode(msgs[-1][1]), True
                    self.broadcast("state", self.live.state)
                elif name == RISK_STREAM:
                    risk = bus.decode(msgs[-1][1])
                    self.live.risk = risk
                    self.broadcast("risk", risk)
                    if self.store and risk["tick"] == self.live.state["tick"]:
                        await self._safe(self.store.add_point(self.live.state, risk))
                else:
                    for _, f in msgs:
                        ev = bus.decode(f)
                        self.live.events.append(ev)
                        self.broadcast("event", ev)
                        if self.store and ev["kind"] in ("cascade", "fault"):
                            await self._safe(self.store.add_incident(ev))

    @staticmethod
    async def _safe(coro) -> None:
        try:
            await coro
        except Exception as e:
            log.warning("persistence failed: %s", e)


hub = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    load_grid(), get_model()  # warm caches
    tasks = []
    try:
        hub.redis = bus.connect()
        await asyncio.wait_for(hub.redis.ping(), 2)
        tasks.append(asyncio.create_task(hub.pump()))
    except Exception as e:
        log.warning("Redis unavailable (%s): live feed disabled", e)
        hub.redis = None
    try:
        hub.store = await asyncio.wait_for(Store.connect(), 5)
    except Exception as e:
        log.warning("Postgres unavailable (%s): persistence disabled", e)
        hub.store = None
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Cascadia API", version="1.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# --------------------------------------------------------------------------- REST
@app.get("/api/health")
async def health():
    return {"ok": True, "live": hub.live.live, "redis": hub.redis is not None,
            "postgres": hub.store is not None, "copilot": bool(settings.anthropic_api_key)}


@app.get("/api/grid")
async def grid():
    g = load_grid()
    return {"name": g.name, "buses": g.raw["buses"], "branches": g.raw["branches"],
            "gens": g.raw["gens"]}


@app.get("/api/state")
async def state():
    return {"state": hub.live.state, "risk": hub.live.risk or ops.screen(hub.live),
            "events": list(hub.live.events)[-30:], "live": hub.live.live}


class WhatIf(BaseModel):
    lines: list[int] = Field(min_length=1, max_length=10)
    load_scale: float = Field(1.0, ge=0.5, le=1.6)


@app.post("/api/whatif")
async def whatif(req: WhatIf):
    n = load_grid().n_branch
    if any(not 0 <= e < n for e in req.lines):
        raise HTTPException(422, f"branch ids must be in [0, {n - 1}]")
    return await asyncio.to_thread(ops.whatif, hub.live, req.lines, req.load_scale)


@app.get("/api/lines/{line}")
async def line(line: int):
    if not 0 <= line < load_grid().n_branch:
        raise HTTPException(404, "no such branch")
    return ops.line_info(hub.live, line)


@app.get("/api/incidents")
async def incidents(limit: int = 50, kind: str | None = None):
    if hub.store:
        return await hub.store.incidents(min(limit, 500), kind)
    evs = [e for e in hub.live.events if e["kind"] in ("cascade", "fault")]
    return list(reversed(evs))[:limit]


@app.get("/api/timeline")
async def timeline(limit: int = 720):
    return await hub.store.timeline(min(limit, 5000)) if hub.store else []


@app.get("/api/model")
async def model_card():
    path = settings.artifacts_dir / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics not generated yet")
    return json.loads(path.read_text())


class Command(BaseModel):
    cmd: Literal["trip", "restore_all", "weather"]
    lines: list[int] = Field(default_factory=list, max_length=10)
    value: Literal["clear", "heatwave", "storm"] | None = None


@app.post("/api/commands", status_code=202)
async def command(c: Command):
    n = load_grid().n_branch
    if any(not 0 <= e < n for e in c.lines):
        raise HTTPException(422, f"branch ids must be in [0, {n - 1}]")
    if not hub.redis:
        raise HTTPException(503, "live simulation is not running")
    await bus.publish(hub.redis, COMMAND_STREAM, c.model_dump())
    return {"queued": True}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


@app.post("/api/copilot")
async def chat(req: ChatRequest):
    async def sse():
        async for ev in copilot.stream(hub.live, [m.model_dump() for m in req.messages]):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- WebSocket
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=16)
    hub.clients.add(q)
    try:
        await ws.send_text(orjson.dumps({"type": "snapshot", "data": {
            "state": hub.live.state, "risk": hub.live.risk,
            "events": list(hub.live.events)[-30:], "live": hub.live.live}}).decode())
        while True:
            await ws.send_text((await q.get()).decode())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.clients.discard(q)

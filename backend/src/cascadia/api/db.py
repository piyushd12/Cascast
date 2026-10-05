"""Postgres persistence: incidents (cascades, faults) and a per-tick risk timeline."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, BigInteger, DateTime, Float, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cascadia.config import settings


class Base(DeclarativeBase):
    pass


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(UTC))
    minute: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(200))
    shed_pct: Mapped[float] = mapped_column(Float, default=0.0)
    shed_mw: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSON)


class TimelinePoint(Base):
    __tablename__ = "timeline"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tick: Mapped[int] = mapped_column(BigInteger)
    minute: Mapped[float] = mapped_column(Float)
    demand_mw: Mapped[float] = mapped_column(Float)
    served_mw: Mapped[float] = mapped_column(Float)
    worst_shed_pct: Mapped[float] = mapped_column(Float)
    n_critical: Mapped[int] = mapped_column(Integer)
    max_loading: Mapped[float] = mapped_column(Float)


class Store:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.session = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    async def connect(cls) -> Store:
        engine = create_async_engine(settings.database_url, pool_size=5,
                                     connect_args={"timeout": 3})
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return cls(engine)

    async def add_incident(self, ev: dict) -> None:
        async with self.session() as s, s.begin():
            s.add(Incident(minute=ev["minute"], kind=ev["kind"], title=ev["title"],
                           shed_pct=ev.get("shed_pct", 0.0), shed_mw=ev.get("shed_mw", 0.0),
                           detail=ev))

    async def add_point(self, state: dict, risk: dict) -> None:
        async with self.session() as s, s.begin():
            s.add(TimelinePoint(
                tick=state["tick"], minute=state["minute"], demand_mw=state["demand_mw"],
                served_mw=state["served_mw"], worst_shed_pct=risk["worst_shed_pct"],
                n_critical=risk["n_critical"], max_loading=max(state["loading"])))

    async def incidents(self, limit: int = 50, kind: str | None = None) -> list[dict]:
        q = select(Incident).order_by(Incident.id.desc()).limit(limit)
        if kind:
            q = q.where(Incident.kind == kind)
        async with self.session() as s:
            rows = (await s.execute(q)).scalars().all()
        return [{**r.detail, "id": r.id, "created_at": r.created_at.isoformat()} for r in rows]

    async def timeline(self, limit: int = 720) -> list[dict]:
        q = select(TimelinePoint).order_by(TimelinePoint.id.desc()).limit(limit)
        async with self.session() as s:
            rows = (await s.execute(q)).scalars().all()
        return [{"tick": r.tick, "minute": r.minute, "demand_mw": r.demand_mw,
                 "served_mw": r.served_mw, "worst_shed_pct": r.worst_shed_pct,
                 "n_critical": r.n_critical, "max_loading": r.max_loading}
                for r in reversed(rows)]

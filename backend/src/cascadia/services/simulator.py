"""Grid digital twin: advances a simulated clock and publishes the grid state to Redis.

Demand follows a daily profile with spatially-correlated noise. A weather process switches
between clear skies, heatwaves (higher demand) and storms (more line faults). Faults trip
lines, the physics engine plays out any resulting cascade, and crews repair lines later.
Operators (the dashboard) can inject commands through the `grid.commands` stream.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from cascadia.config import COMMAND_STREAM, EVENT_STREAM, STATE_STREAM, settings
from cascadia.grid.cascade import settle, simulate_cascade
from cascadia.grid.dcpf import FlowResult, dc_power_flow
from cascadia.grid.model import Grid, load_grid
from cascadia.services import bus

log = logging.getLogger("cascadia.simulator")

WEATHER = {  # demand multiplier, fault-rate multiplier, (min, max) duration in hours
    "clear": (1.0, 1.0, (6, 14)),
    "heatwave": (1.16, 1.3, (3, 7)),
    "storm": (1.0, 5.0, (1, 3)),
}
_event_ids = count(1)


def daily_profile(minute: float) -> float:
    """Typical two-peak demand shape, ~0.72 at night to ~1.05 at the evening peak."""
    h = (minute / 60.0) % 24
    return (0.72 + 0.18 * math.exp(-(((h - 10) / 3.0) ** 2))
            + 0.33 * math.exp(-(((h - 19) / 2.5) ** 2)))


@dataclass
class GridTwin:
    grid: Grid
    seed: int = 7
    minute: float = 6 * 60.0
    weather: str = "clear"
    weather_until: float = 9 * 60.0
    alive: np.ndarray = field(init=False)
    repair_at: dict[int, float] = field(default_factory=dict)
    noise: np.ndarray = field(init=False)
    state: FlowResult = field(init=False)
    load: np.ndarray = field(init=False)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.alive = np.ones(self.grid.n_branch, dtype=bool)
        self.noise = np.zeros(self.grid.n_bus)
        self.load = self.grid.load * self.load_factor
        self.state = dc_power_flow(self.grid, self.alive, self.load)

    @property
    def load_factor(self) -> float:
        return daily_profile(self.minute) * WEATHER[self.weather][0]

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, title: str, **extra) -> dict:
        ev = {"id": next(_event_ids), "minute": self.minute, "kind": kind, "title": title, **extra}
        self.events.append(ev)
        return ev

    def set_weather(self, name: str, hours: float | None = None) -> None:
        lo, hi = WEATHER[name][2]
        changed, self.weather = name != self.weather, name
        self.weather_until = self.minute + 60 * (hours or self.rng.uniform(lo, hi))
        if changed:
            self._event("weather", f"Weather: {name}", weather=name)

    def trip(self, lines: list[int], cause: str) -> dict | None:
        lines = [int(e) for e in lines if self.alive[e]]
        if not lines:
            return None
        res = simulate_cascade(self.grid, self.load, lines, alive=self.alive, pre=self.state,
                               record=True, rng=self.rng)
        self.alive = res.alive
        self.state = res.final
        for e in lines:  # damaged equipment: hours to repair
            self.repair_at[e] = self.minute + self.rng.uniform(60, 180)
        cascaded = np.flatnonzero(res.failed).tolist()
        for e in cascaded:  # protection trips: crews re-close sooner
            self.repair_at[e] = self.minute + self.rng.uniform(20, 60)
        sev = "cascade" if cascaded else "fault"
        title = (f"{cause}: line {lines[0]}" + (f" +{len(lines) - 1}" if len(lines) > 1 else "")
                 + (f" → {len(cascaded)} cascading trips" if cascaded else ""))
        n_hidden = int(res.hidden.sum())
        if n_hidden:
            title += f" ({n_hidden} relay misoperation{'s' if n_hidden > 1 else ''})"
        return self._event(
            sev, title, cause=cause, lines=lines, cascade=cascaded,
            hidden=np.flatnonzero(res.hidden).tolist(),
            stages=[s.tripped for s in res.stages],
            shed_mw=round(res.served_before_mw - res.served_after_mw, 1),
            shed_pct=round(100 * res.load_shed_frac, 2),
        )

    def restore_all(self) -> None:
        self.repair_at = {e: self.minute for e in self.repair_at}

    def apply(self, cmd: dict) -> None:
        match cmd.get("cmd"):
            case "trip":
                self.trip(cmd.get("lines", []), "Operator test")
            case "restore_all":
                self.restore_all()
            case "weather" if cmd.get("value") in WEATHER:
                self.set_weather(cmd["value"], cmd.get("hours"))

    # ------------------------------------------------------------------ dynamics
    def step(self, dt: float) -> None:
        self.minute += dt
        if self.minute >= self.weather_until:
            nxt = "clear" if self.weather != "clear" else self.rng.choice(
                ["clear", "heatwave", "storm"], p=[0.4, 0.3, 0.3])
            self.set_weather(str(nxt))

        # Spatially smooth AR(1) demand noise
        self.noise = 0.97 * self.noise + self.rng.normal(0, 0.006, self.grid.n_bus)
        self.load = self.grid.load * self.load_factor * (1 + np.clip(self.noise, -0.15, 0.15))

        # Repairs
        done = [e for e, t in self.repair_at.items() if t <= self.minute]
        if done:
            for e in done:
                del self.repair_at[e]
            self.alive[done] = True
            self._event("restored", f"Crews restored {len(done)} line(s)", lines=done)

        # Random faults (Poisson); storms multiply the rate
        rate = settings.fault_rate_per_hour * WEATHER[self.weather][1] * dt / 60
        for _ in range(self.rng.poisson(rate)):
            self.state = dc_power_flow(self.grid, self.alive, self.load)
            cause = "Storm damage" if self.weather == "storm" else "Fault"
            self.trip([int(self.rng.choice(np.flatnonzero(self.alive)))], cause)

        # Demand growth can overload lines on its own
        before = self.alive.copy()
        self.alive, self.state = settle(self.grid, self.alive, self.load)
        tripped = np.flatnonzero(before & ~self.alive).tolist()
        if tripped:
            # Same demand, pre-trip topology: otherwise demand drift reads as load shed
            served_before = dc_power_flow(self.grid, before, self.load).served.sum()
            for e in tripped:
                self.repair_at[e] = self.minute + self.rng.uniform(20, 60)
            self._event("cascade", f"Overload trips: {len(tripped)} line(s)", cause="Overload",
                        lines=tripped, cascade=[], stages=[tripped],
                        shed_mw=round(served_before - self.state.served.sum(), 1),
                        shed_pct=round(100 * (served_before - self.state.served.sum())
                                       / self.load.sum(), 2))

    def snapshot(self, tick: int) -> dict:
        g, s = self.grid, self.state
        m = int(self.minute)
        return {
            "tick": tick,
            "minute": self.minute,
            "clock": f"Day {m // 1440 + 1} · {m % 1440 // 60:02d}:{m % 60:02d}",
            "weather": self.weather,
            "load_factor": round(self.load_factor, 4),
            "demand_mw": round(float(self.load.sum()), 1),
            "served_mw": round(float(s.served.sum()), 1),
            "n_islands": int(s.n_islands),
            "alive": self.alive.astype(np.int8).tolist(),
            "flow": bus.r3(s.flow),
            "loading": bus.r3(np.abs(s.flow) / g.rating),
            "load": bus.r3(self.load),
            "served": bus.r3(s.served),
            "repairs": {str(e): round(t - self.minute) for e, t in self.repair_at.items()},
        }


async def run() -> None:
    r = bus.connect()
    twin = GridTwin(load_grid())
    last = await r.xrevrange(COMMAND_STREAM, count=1)
    cmd_id = last[0][0] if last else "0-0"
    log.info("simulator started: tick=%.1fs, %s sim-min/tick", settings.tick_seconds,
             settings.sim_minutes_per_tick)
    for tick in count():
        t0 = asyncio.get_running_loop().time()
        rows = await r.xread({COMMAND_STREAM: cmd_id}, count=20, block=None)
        for _, msgs in rows:
            for mid, fields in msgs:
                cmd_id = mid
                twin.apply(bus.decode(fields))
        twin.step(settings.sim_minutes_per_tick)
        await bus.publish(r, STATE_STREAM, twin.snapshot(tick))
        for ev in twin.events:
            await bus.publish(r, EVENT_STREAM, ev)
        twin.events.clear()
        elapsed = asyncio.get_running_loop().time() - t0
        await asyncio.sleep(max(0.0, settings.tick_seconds - elapsed))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()

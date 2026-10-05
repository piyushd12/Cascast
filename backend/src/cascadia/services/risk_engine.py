"""Risk engine: consumes grid states, screens every N-1 contingency with the GNN in one
batch, and publishes a risk assessment. Scales horizontally through a Redis consumer group.

Every `verify_every` ticks it replays a sample of contingencies through the physics
simulator and tracks how well the GNN agrees in production ("live accuracy").
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass

import numpy as np
from redis.exceptions import ResponseError

from cascadia.config import RISK_STREAM, STATE_STREAM, settings
from cascadia.grid.cascade import simulate_cascade
from cascadia.grid.dcpf import FlowResult
from cascadia.grid.model import Grid, load_grid
from cascadia.ml.inference import RiskModel, Screening, get_model
from cascadia.services import bus

log = logging.getLogger("cascadia.risk")
FAIL_P = 0.5
CRITICAL_SHED = 0.005  # contingencies shedding > 0.5% of demand are "critical"
TOP_K = 15


@dataclass
class LiveAccuracy:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    checked: int = 0
    shed_abs_err: float = 0.0

    def as_dict(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0
        return {"checked": self.checked, "precision": round(p, 4), "recall": round(r, 4),
                "shed_mae_pct": round(100 * self.shed_abs_err / max(self.checked, 1), 3)}


def state_arrays(s: dict) -> tuple[np.ndarray, np.ndarray, FlowResult]:
    load = np.asarray(s["load"])
    alive = np.asarray(s["alive"], dtype=bool)
    pre = FlowResult(flow=np.asarray(s["flow"]), served=np.asarray(s["served"]),
                     pg=np.empty(0), theta=np.empty(0), n_islands=s["n_islands"])
    return load, alive, pre


def assess(grid: Grid, sc: Screening) -> dict:
    n_fail = (sc.prob >= FAIL_P).sum(1)
    severity = sc.shed + 0.01 * n_fail  # rank by blackout size, then by cascade breadth
    order = np.argsort(-severity)[:TOP_K]
    critical = (sc.shed > CRITICAL_SHED) | (n_fail > 0)
    worst = float(sc.shed.max(initial=0.0))
    n_crit = int(critical.sum())
    status = "secure" if n_crit == 0 else ("emergency" if worst > 0.05 or n_crit > 8 else "alert")
    return {
        "status": status,
        "worst_shed_pct": round(100 * worst, 2),
        "n_critical": n_crit,
        "n_screened": int(sc.contingency.size),
        "latency_ms": round(sc.latency_ms, 2),
        "top": [
            {"line": int(sc.contingency[i]), "shed_pct": round(100 * float(sc.shed[i]), 2),
             "n_fail": int(n_fail[i]),
             "cascade": np.flatnonzero(sc.prob[i] >= FAIL_P).tolist(),
             "max_prob": round(float(sc.prob[i].max()), 3)}
            for i in order if critical[i]
        ],
        "vulnerability": bus.r3(sc.vulnerability),
    }


def verify(grid: Grid, load, alive, pre, sc: Screening, acc: LiveAccuracy,
           rng: np.random.Generator, k: int = 6) -> None:
    """Replay the riskiest and some random contingencies through physics."""
    idx = np.unique(np.r_[np.argsort(-sc.shed)[: k // 2],
                          rng.choice(sc.contingency.size, k // 2, replace=False)])
    for i in idx:
        res = simulate_cascade(grid, load, [sc.contingency[i]], alive=alive, pre=pre)
        pred = sc.prob[i] >= FAIL_P
        acc.tp += int((pred & res.failed).sum())
        acc.fp += int((pred & ~res.failed).sum())
        acc.fn += int((~pred & res.failed).sum())
        acc.shed_abs_err += abs(float(sc.shed[i]) - res.load_shed_frac)
        acc.checked += 1


async def run() -> None:
    r = bus.connect()
    grid, model = load_grid(), get_model()
    name = f"{socket.gethostname()}-{os.getpid()}"
    try:
        await r.xgroup_create(STATE_STREAM, settings.risk_group, id="$", mkstream=True)
    except ResponseError as e:  # group already exists
        if "BUSYGROUP" not in str(e):
            raise
    acc, rng = LiveAccuracy(), np.random.default_rng(0)
    log.info("risk engine %s ready", name)
    while True:
        rows = await r.xreadgroup(settings.risk_group, name, {STATE_STREAM: ">"},
                                 count=50, block=5000)
        if not rows:
            continue
        msgs = rows[0][1]
        await r.xack(STATE_STREAM, settings.risk_group, *[mid for mid, _ in msgs])
        state = bus.decode(msgs[-1][1])  # if we fell behind, only the newest state matters
        out = await asyncio.to_thread(process, grid, model, state, acc, rng)
        await bus.publish(r, RISK_STREAM, out)


def process(grid, model: RiskModel, state, acc, rng) -> dict:
    load, alive, pre = state_arrays(state)
    sc = model.screen(load, alive, pre)
    out = assess(grid, sc)
    if state["tick"] % settings.verify_every == 0:
        verify(grid, load, alive, pre, sc, acc, rng)
    out.update(tick=state["tick"], minute=state["minute"], live_accuracy=acc.as_dict())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()

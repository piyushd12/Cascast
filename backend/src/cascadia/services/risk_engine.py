"""Risk engine: consumes grid states, screens every N-1 contingency with the GNN in one
batch, and publishes a risk assessment. Scales horizontally through a Redis consumer group.

Every `verify_every` ticks it replays a few contingencies through Monte Carlo physics and
tracks how closely the GNN's probabilities match in production ("live accuracy").
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
from cascadia.grid.cascade import monte_carlo
from cascadia.grid.dcpf import FlowResult
from cascadia.grid.model import Grid, load_grid
from cascadia.ml.inference import RiskModel, Screening, get_model
from cascadia.services import bus

log = logging.getLogger("cascadia.risk")
FAIL_P = 0.5
CRITICAL_SHED = 0.005  # contingencies shedding > 0.5% of demand are "critical"
TOP_K = 15


MC_RUNS = 40


@dataclass
class LiveAccuracy:
    """Running agreement between GNN probabilities and Monte Carlo estimates."""

    checked: int = 0
    prob_abs_err: float = 0.0
    prob_n: int = 0
    shed_abs_err: float = 0.0
    hits: int = 0  # most likely failure (by MC) also ranked first by the GNN

    def as_dict(self) -> dict:
        return {"checked": self.checked, "mc_runs": MC_RUNS,
                "prob_mae": round(self.prob_abs_err / max(self.prob_n, 1), 4),
                "shed_mae_pct": round(100 * self.shed_abs_err / max(self.checked, 1), 3),
                "top1_agreement": round(self.hits / max(self.checked, 1), 3)}


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
           rng: np.random.Generator, k: int = 2) -> None:
    """Check the riskiest and one random contingency against Monte Carlo physics."""
    idx = np.unique(np.r_[np.argsort(-sc.shed)[:k], rng.choice(sc.contingency.size, 1)])
    for i in idx:
        mc = monte_carlo(grid, load, [sc.contingency[i]], alive, pre, MC_RUNS, rng)
        active = (mc.fail_prob > 0) | (sc.prob[i] > 0.02)
        acc.prob_abs_err += float(np.abs(sc.prob[i] - mc.fail_prob)[active].sum())
        acc.prob_n += int(active.sum())
        acc.shed_abs_err += abs(float(sc.shed[i]) - mc.expected_shed)
        if mc.fail_prob.max() > 0:
            acc.hits += int(np.argmax(sc.prob[i]) == np.argmax(mc.fail_prob))
        else:
            acc.hits += int(sc.prob[i].max() < FAIL_P)
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

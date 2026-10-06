"""Grid operations shared by the REST API and the Copilot's tools.

`LiveState` holds the newest state/risk from the streams. Without Redis it falls back to a
static nominal operating point, so what-if analysis works stand-alone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cascast.grid.cascade import monte_carlo, simulate_cascade
from cascast.grid.dcpf import FlowResult, dc_power_flow
from cascast.grid.model import load_grid
from cascast.ml.inference import get_model
from cascast.services.bus import r3
from cascast.services.risk_engine import FAIL_P, assess


def nominal_state() -> dict:
    g = load_grid()
    alive = np.ones(g.n_branch, dtype=bool)
    res = dc_power_flow(g, alive, g.load)
    return {"tick": -1, "minute": 0, "clock": "Nominal", "weather": "clear", "load_factor": 1.0,
            "demand_mw": round(float(g.load.sum()), 1),
            "served_mw": round(float(res.served.sum()), 1),
            "n_islands": int(res.n_islands), "alive": alive.astype(int).tolist(),
            "flow": r3(res.flow), "loading": r3(np.abs(res.flow) / g.rating), "load": r3(g.load),
            "served": r3(res.served), "repairs": {}}


@dataclass
class LiveState:
    state: dict = field(default_factory=nominal_state)
    risk: dict | None = None
    events: deque = field(default_factory=lambda: deque(maxlen=100))
    live: bool = False

    def arrays(self, load_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, FlowResult]:
        g = load_grid()
        load = np.asarray(self.state["load"]) * load_scale
        alive = np.asarray(self.state["alive"], dtype=bool)
        if load_scale == 1.0:
            pre = FlowResult(np.asarray(self.state["flow"]), np.asarray(self.state["served"]),
                             np.empty(0), np.empty(0), self.state["n_islands"])
        else:
            pre = dc_power_flow(g, alive, load)
        return load, alive, pre


def whatif(live: LiveState, lines: list[int], load_scale: float = 1.0, runs: int = 100,
           seed: int | None = None) -> dict:
    """Trip `lines` on the current operating point.

    Returns the GNN's failure probabilities, a Monte Carlo estimate from `runs` physics
    simulations, and one recorded random realisation of the cascade for replay.
    """
    g, model = load_grid(), get_model()
    load, alive, pre = live.arrays(load_scale)
    overloaded = alive & (np.abs(pre.flow) > g.rating)
    trip = np.zeros(g.n_branch, dtype=bool)
    trip[lines] = True
    trip &= alive
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    prob, shed = model.predict(load, pre.served, pre.flow, alive, trip[None])
    gnn_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    mc = monte_carlo(g, load, trip, alive, pre, runs, rng)
    mc_ms = (time.perf_counter() - t0) * 1000
    sample = simulate_cascade(g, load, trip, alive=alive, pre=pre, record=True, rng=rng)

    p = prob[0]
    active = (mc.fail_prob > 0) | (p > 0.02)
    likely_gnn, likely_mc = p >= FAIL_P, mc.fail_prob >= FAIL_P
    union = int((likely_gnn | likely_mc).sum())
    return {
        "lines": np.flatnonzero(trip).tolist(),
        "load_scale": load_scale,
        "pre_overloaded": np.flatnonzero(overloaded).tolist(),
        "gnn": {"prob": r3(p), "shed_pct": round(100 * float(shed[0]), 2),
                "likely": np.flatnonzero(likely_gnn).tolist(), "latency_ms": round(gnn_ms, 2)},
        "monte_carlo": {
            "runs": runs, "prob": r3(mc.fail_prob),
            "likely": np.flatnonzero(likely_mc).tolist(),
            "expected_shed_pct": round(100 * mc.expected_shed, 2),
            "shed_p50_pct": round(100 * float(np.median(mc.shed)), 2),
            "shed_p90_pct": round(100 * float(np.percentile(mc.shed, 90)), 2),
            "cascade_prob": round(float((mc.shed > 0).mean()), 3),
            "latency_ms": round(mc_ms, 1),
        },
        "sample": {
            "failed": np.flatnonzero(sample.failed).tolist(),
            "hidden": np.flatnonzero(sample.hidden).tolist(),
            "stage": sample.stage.tolist(),
            "shed_pct": round(100 * sample.load_shed_frac, 2),
            "n_stages": sample.n_stages,
            "islands": int(sample.final.n_islands),
            "stages": [{"tripped": s.tripped, "loading": r3(s.loading),
                        "served_mw": round(float(s.served_mw), 1)} for s in sample.stages],
        },
        "agreement": {
            "prob_mae": round(float(np.abs(p - mc.fail_prob)[active].mean()), 4)
            if active.any() else 0.0,
            "likely_jaccard": round(int((likely_gnn & likely_mc).sum()) / union, 3)
            if union else 1.0,
            "shed_error_pct": round(abs(100 * float(shed[0]) - 100 * mc.expected_shed), 2),
        },
    }


def line_info(live: LiveState, line: int) -> dict:
    g = load_grid()
    names = [b["name"] for b in g.raw["buses"]]
    s = live.state
    info = {
        "line": line,
        "from_bus": names[g.f[line]], "to_bus": names[g.t[line]],
        "transformer": bool(g.is_trafo[line]),
        "rating_mw": float(g.rating[line]),
        "flow_mw": s["flow"][line],
        "loading_pct": round(100 * s["loading"][line], 1),
        "in_service": bool(s["alive"][line]),
        "repair_in_min": s["repairs"].get(str(line)),
    }
    if live.risk:
        info["vulnerability"] = live.risk["vulnerability"][line]
    return info


def screen(live: LiveState, load_scale: float = 1.0) -> dict:
    """Full N-1 screen of the current state, optionally with scaled demand."""
    g, model = load_grid(), get_model()
    load, alive, pre = live.arrays(load_scale)
    return assess(g, model.screen(load, alive, pre))

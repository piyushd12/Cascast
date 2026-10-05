"""Grid operations shared by the REST API and the Copilot's tools.

`LiveState` holds the newest state/risk from the streams. Without Redis it falls back to a
static nominal operating point, so what-if analysis works stand-alone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cascadia.grid.cascade import simulate_cascade
from cascadia.grid.dcpf import FlowResult, dc_power_flow
from cascadia.grid.model import load_grid
from cascadia.ml.inference import get_model
from cascadia.services.bus import r3
from cascadia.services.risk_engine import FAIL_P, assess


def nominal_state() -> dict:
    g = load_grid()
    alive = np.ones(g.n_branch, dtype=bool)
    res = dc_power_flow(g, alive, g.load)
    return {"tick": -1, "minute": 0, "clock": "Nominal", "weather": "clear", "load_factor": 1.0,
            "demand_mw": round(float(g.load.sum()), 1), "served_mw": round(float(res.served.sum()), 1),
            "n_islands": res.n_islands, "alive": alive.astype(int).tolist(), "flow": r3(res.flow),
            "loading": r3(np.abs(res.flow) / g.rating), "load": r3(g.load),
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


def whatif(live: LiveState, lines: list[int], load_scale: float = 1.0) -> dict:
    """Trip `lines` on the current operating point: GNN prediction vs physics ground truth."""
    g, model = load_grid(), get_model()
    load, alive, pre = live.arrays(load_scale)
    overloaded = alive & (np.abs(pre.flow) > g.rating)
    trip = np.zeros(g.n_branch, dtype=bool)
    trip[lines] = True
    trip &= alive

    t0 = time.perf_counter()
    prob, shed = model.predict(load, pre.served, pre.flow, alive, trip[None])
    gnn_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    res = simulate_cascade(g, load, trip, alive=alive, pre=pre, record=True)
    phys_ms = (time.perf_counter() - t0) * 1000

    pred = prob[0] >= FAIL_P
    tp = int((pred & res.failed).sum())
    union = int((pred | res.failed).sum())
    return {
        "lines": np.flatnonzero(trip).tolist(),
        "load_scale": load_scale,
        "pre_overloaded": np.flatnonzero(overloaded).tolist(),
        "gnn": {"prob": r3(prob[0]), "shed_pct": round(100 * float(shed[0]), 2),
                "predicted": np.flatnonzero(pred).tolist(), "latency_ms": round(gnn_ms, 2)},
        "physics": {
            "failed": np.flatnonzero(res.failed).tolist(),
            "stage": res.stage.tolist(),
            "shed_pct": round(100 * res.load_shed_frac, 2),
            "shed_mw": round(res.served_before_mw - res.served_after_mw, 1),
            "n_stages": res.n_stages,
            "islands": res.final.n_islands,
            "stages": [{"tripped": s.tripped, "loading": r3(s.loading),
                        "served_mw": round(float(s.served_mw), 1)} for s in res.stages],
            "latency_ms": round(phys_ms, 2),
        },
        "agreement": {"jaccard": round(tp / union, 3) if union else 1.0,
                      "true_positive": tp,
                      "false_positive": int((pred & ~res.failed).sum()),
                      "false_negative": int((~pred & res.failed).sum())},
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

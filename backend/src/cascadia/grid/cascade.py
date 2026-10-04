"""Quasi-static cascading-failure simulator.

Starting from a stable operating point, a set of branches is tripped. The DC power flow is
re-solved and every branch loaded above its thermal rating trips simultaneously. This repeats
until no branch is overloaded. Islands that lose generation shed load (see dcpf.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cascadia.grid.dcpf import FlowResult, dc_power_flow
from cascadia.grid.model import Grid

MAX_STAGES = 30


@dataclass
class Stage:
    tripped: list[int]  # branches that tripped at this stage
    loading: np.ndarray  # (E,) loading after this stage's re-dispatch
    served_mw: float


@dataclass
class CascadeResult:
    alive: np.ndarray  # (E,) final in-service mask
    stage: np.ndarray  # (E,) -1 = never failed, 0 = initial trip, k = failed at stage k
    failed: np.ndarray  # (E,) bool, cascaded failures only (excludes initial trips)
    demand_mw: float
    served_before_mw: float
    served_after_mw: float
    final: FlowResult
    stages: list[Stage] = field(default_factory=list)

    @property
    def load_shed_frac(self) -> float:
        """Load lost because of this event, as a fraction of total demand."""
        return max(0.0, self.served_before_mw - self.served_after_mw) / self.demand_mw

    @property
    def n_stages(self) -> int:
        return int(self.stage.max(initial=0))


def settle(grid: Grid, alive: np.ndarray, load: np.ndarray) -> tuple[np.ndarray, FlowResult]:
    """Trip overloaded branches until the network is stable. Returns (alive, flow)."""
    alive = alive.copy()
    for _ in range(MAX_STAGES):
        res = dc_power_flow(grid, alive, load)
        over = alive & (np.abs(res.flow) > grid.rating)
        if not over.any():
            return alive, res
        alive &= ~over
    return alive, dc_power_flow(grid, alive, load)


def simulate_cascade(
    grid: Grid,
    load: np.ndarray,
    trip: np.ndarray,
    alive: np.ndarray | None = None,
    pre: FlowResult | None = None,
    record: bool = False,
) -> CascadeResult:
    """Trip `trip` (bool mask or index array) on a stable state `alive` and run the cascade."""
    alive = np.ones(grid.n_branch, dtype=bool) if alive is None else alive.copy()
    trip_mask = np.zeros(grid.n_branch, dtype=bool)
    trip_mask[trip] = True
    trip_mask &= alive
    pre = pre or dc_power_flow(grid, alive, load)

    stage = np.full(grid.n_branch, -1, dtype=np.int64)
    stage[trip_mask] = 0
    alive &= ~trip_mask
    stages: list[Stage] = []

    for k in range(1, MAX_STAGES + 1):
        res = dc_power_flow(grid, alive, load)
        if record:
            stages.append(
                Stage(np.flatnonzero(stage == k - 1).tolist(), res.loading(grid), res.served.sum())
            )
        over = alive & (np.abs(res.flow) > grid.rating)
        if not over.any():
            break
        stage[over] = k
        alive &= ~over
    else:
        res = dc_power_flow(grid, alive, load)

    return CascadeResult(
        alive=alive,
        stage=stage,
        failed=stage > 0,
        demand_mw=float(load.sum()),
        served_before_mw=float(pre.served.sum()),
        served_after_mw=float(res.served.sum()),
        final=res,
        stages=stages,
    )

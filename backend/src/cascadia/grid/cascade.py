"""Quasi-static cascading-failure simulator with stochastic protection.

Starting from a stable operating point, a set of branches is tripped. The DC power flow is
re-solved and protection acts on every in-service branch, stage by stage, until nothing
else trips. Islands that lose generation shed load (see dcpf.py).

Protection model (after the hidden-failure models of Thorp, Phadke et al. and Chen, Thorp &
Dobson, "Cascading dynamics and mitigation assessment in power system disturbances via a
hidden failure model", 2005):

* Overload: each branch trips when its loading exceeds a threshold τ = 1 + σ·|z|, z ~ N(0, 1),
  drawn once per event. Real relays and conductors tolerate short overloads by varying amounts.
* Hidden failures: when a branch trips, relays on branches sharing a bus with it are exposed;
  each misoperates (trips) with probability p_hf. Hidden failures are what turned local
  faults into the 1996 WSCC and 2003 Northeast blackouts.

Because the outcome is random, the quantity an operator needs is a probability, and physics
can only estimate it by Monte Carlo: many full cascade simulations per contingency.
Passing `rng=None` gives the deterministic special case (τ = 1, no hidden failures).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cascadia.grid.dcpf import FlowResult, dc_power_flow
from cascadia.grid.model import Grid

MAX_STAGES = 30


@dataclass(frozen=True)
class Protection:
    threshold_sigma: float = 0.06  # spread of overload trip thresholds (fraction of rating)
    hidden_failure_p: float = 0.02  # misoperation probability of an exposed adjacent relay


PROTECTION = Protection()


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
    hidden: np.ndarray  # (E,) bool, failures caused by relay misoperation
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
    """Trip overloaded branches (nominal ratings) until stable. Returns (alive, flow)."""
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
    rng: np.random.Generator | None = None,
    protection: Protection = PROTECTION,
) -> CascadeResult:
    """Trip `trip` (bool mask or index array) on a stable state `alive` and run the cascade."""
    alive = np.ones(grid.n_branch, dtype=bool) if alive is None else alive.copy()
    trip_mask = np.zeros(grid.n_branch, dtype=bool)
    trip_mask[trip] = True
    trip_mask &= alive
    pre = pre or dc_power_flow(grid, alive, load)

    stochastic = rng is not None
    limit = grid.rating
    if stochastic:
        limit = grid.rating * (1 + protection.threshold_sigma * np.abs(
            rng.standard_normal(grid.n_branch)))

    stage = np.full(grid.n_branch, -1, dtype=np.int64)
    stage[trip_mask] = 0
    hidden = np.zeros(grid.n_branch, dtype=bool)
    alive &= ~trip_mask
    last = trip_mask
    stages: list[Stage] = []

    for k in range(1, MAX_STAGES + 1):
        res = dc_power_flow(grid, alive, load)
        if record:
            stages.append(Stage(np.flatnonzero(last).tolist(), res.loading(grid),
                                float(res.served.sum())))
        over = alive & (np.abs(res.flow) > limit)
        if stochastic and protection.hidden_failure_p > 0:
            exposed_bus = np.zeros(grid.n_bus, dtype=bool)
            exposed_bus[grid.f[last]] = exposed_bus[grid.t[last]] = True
            exposed = alive & ~over & (exposed_bus[grid.f] | exposed_bus[grid.t])
            misop = exposed & (rng.random(grid.n_branch) < protection.hidden_failure_p)
            hidden |= misop
            over |= misop
        if not over.any():
            break
        stage[over] = k
        alive &= ~over
        last = over
    else:
        res = dc_power_flow(grid, alive, load)

    return CascadeResult(
        alive=alive,
        stage=stage,
        failed=stage > 0,
        hidden=hidden,
        demand_mw=float(load.sum()),
        served_before_mw=float(pre.served.sum()),
        served_after_mw=float(res.served.sum()),
        final=res,
        stages=stages,
    )


@dataclass
class MonteCarlo:
    fail_prob: np.ndarray  # (E,) fraction of runs in which each branch failed
    shed: np.ndarray  # (n,) load-shed fraction of each run
    runs: int

    @property
    def expected_shed(self) -> float:
        return float(self.shed.mean())


def monte_carlo(
    grid: Grid,
    load: np.ndarray,
    trip: np.ndarray,
    alive: np.ndarray,
    pre: FlowResult,
    runs: int,
    rng: np.random.Generator,
    protection: Protection = PROTECTION,
) -> MonteCarlo:
    """Estimate per-branch failure probability and the load-shed distribution."""
    count = np.zeros(grid.n_branch)
    shed = np.empty(runs)
    for i in range(runs):
        res = simulate_cascade(grid, load, trip, alive=alive, pre=pre, rng=rng,
                               protection=protection)
        count += res.failed
        shed[i] = res.load_shed_frac
    return MonteCarlo(fail_prob=count / runs, shed=shed, runs=runs)

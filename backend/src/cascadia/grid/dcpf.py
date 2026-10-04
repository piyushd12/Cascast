"""DC power flow with island detection and capacity-limited generator dispatch.

Model: lossless DC approximation, P = B θ. Each connected island balances itself:
generators share the island's demand in proportion to their capacity, and if the island
lacks capacity (or has no generator at all) its load is curtailed uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cascadia.grid.model import Grid


@dataclass
class FlowResult:
    flow: np.ndarray  # (E,) branch flow [MW], 0 for out-of-service branches
    served: np.ndarray  # (N,) demand actually served [MW]
    pg: np.ndarray  # (G,) generator output [MW]
    theta: np.ndarray  # (N,) voltage angles (arbitrary units, slack = 0 per island)
    n_islands: int

    def loading(self, grid: Grid) -> np.ndarray:
        return np.abs(self.flow) / grid.rating


def islands(grid: Grid, alive: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the in-service network."""
    f, t = grid.f[alive], grid.t[alive]
    adj = coo_matrix((np.ones(f.size), (f, t)), shape=(grid.n_bus, grid.n_bus))
    return connected_components(adj, directed=False)


def dc_power_flow(grid: Grid, alive: np.ndarray, load: np.ndarray) -> FlowResult:
    n_comp, label = islands(grid, alive)

    # Island-level balance: serve min(demand, capacity), dispatch generators pro rata to capacity.
    gen_comp = label[grid.gen_bus]
    cap = np.bincount(gen_comp, weights=grid.gen_pmax, minlength=n_comp)
    demand = np.bincount(label, weights=load, minlength=n_comp)
    served_frac = np.divide(cap, demand, out=np.ones(n_comp), where=demand > 0).clip(max=1.0)
    served = load * served_frac[label]
    gen_util = np.divide(demand * served_frac, cap, out=np.zeros(n_comp), where=cap > 0)
    pg = grid.gen_pmax * gen_util[gen_comp]

    p = np.bincount(grid.gen_bus, weights=pg, minlength=grid.n_bus) - served
    flow, theta = solve_flows(grid, alive, p, label, n_comp)
    return FlowResult(flow=flow, served=served, pg=pg, theta=theta, n_islands=n_comp)


def solve_flows(
    grid: Grid, alive: np.ndarray, p: np.ndarray, label: np.ndarray, n_comp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve B θ = p for island-balanced injections `p` [MW]; return (branch flows, θ)."""
    # One slack bus per island (its largest generator, else its first bus); solve the rest.
    slack = np.zeros(grid.n_bus, dtype=bool)
    cap_bus = grid.gen_capacity_at_bus
    for c in range(n_comp):
        members = np.flatnonzero(label == c)
        slack[members[np.argmax(cap_bus[members])]] = True

    A = grid.incidence[:, alive]
    B = (A * grid.b[alive]) @ A.T
    keep = ~slack
    theta = np.zeros(grid.n_bus)
    if keep.any():
        theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], p[keep])

    flow = np.zeros(grid.n_branch)
    flow[alive] = grid.b[alive] * (theta[grid.f[alive]] - theta[grid.t[alive]])
    return flow, theta

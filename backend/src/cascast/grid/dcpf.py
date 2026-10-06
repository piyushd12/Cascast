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

from cascast.grid.model import Grid


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


def _reduced_laplacian(
    grid: Grid, alive: np.ndarray, label: np.ndarray, n_comp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Laplacian B = A diag(b) A^T and the mask of non-slack buses."""
    # One slack bus per island (its largest generator, else its first bus).
    slack = np.zeros(grid.n_bus, dtype=bool)
    cap_bus = grid.gen_capacity_at_bus
    for c in range(n_comp):
        members = np.flatnonzero(label == c)
        slack[members[np.argmax(cap_bus[members])]] = True

    # Assembled by scatter-add, much faster than a dense matmul.
    n, f, t, b = grid.n_bus, grid.f[alive], grid.t[alive], grid.b[alive]
    idx = np.concatenate([f * n + f, t * n + t, f * n + t, t * n + f])
    B = np.bincount(idx, weights=np.concatenate([b, b, -b, -b]), minlength=n * n).reshape(n, n)
    return B, ~slack


def solve_flows(
    grid: Grid, alive: np.ndarray, p: np.ndarray, label: np.ndarray, n_comp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve B θ = p for island-balanced injections `p` [MW]; return (branch flows, θ)."""
    B, keep = _reduced_laplacian(grid, alive, label, n_comp)
    theta = np.zeros(grid.n_bus)
    if keep.any():
        theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], p[keep])

    flow = np.zeros(grid.n_branch)
    flow[alive] = grid.b[alive] * (theta[grid.f[alive]] - theta[grid.t[alive]])
    return flow, theta


def n1_flows(
    grid: Grid, alive: np.ndarray, load: np.ndarray, pre: FlowResult
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-stage flows right after each single in-service branch outage.

    Uses line outage distribution factors (exact under the DC model), so one matrix inverse
    covers every contingency. An outage of a bridge (a branch whose loss splits an island)
    also changes the dispatch: the PTDF row of a bridge is ±1 exactly on the buses it cuts
    off, which lets us re-balance both sides and re-solve with the same PTDF, vectorised.
    Returns (contingency ids (K,), flows (K, E), served demand (K, N)).
    """
    n_comp, label = islands(grid, alive)
    B, keep = _reduced_laplacian(grid, alive, label, n_comp)
    X = np.zeros((grid.n_bus, grid.n_bus))
    X[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])

    cont = np.flatnonzero(alive)
    k = np.arange(cont.size)
    H = np.where(alive, grid.b, 0.0)[:, None] * (X[grid.f] - X[grid.t])  # PTDF (E, N)
    phi = H[:, grid.f[cont]] - H[:, grid.t[cont]]  # (E, K) flow change per unit transfer
    denom = 1.0 - phi[cont, k]
    bridge = np.abs(denom) < 1e-8
    w = np.where(bridge, 0.0, pre.flow[cont] / np.where(bridge, 1.0, denom))
    flow = pre.flow[None, :] + (phi * w).T
    flow[k, cont] = 0.0
    served = np.repeat(pre.served[None, :], cont.size, axis=0)

    if bridge.any():
        bi = np.flatnonzero(bridge)
        c = cont[bi]
        far = np.abs(H[c]) > 0.5  # (nb, N) buses cut off from their island's slack
        near = (label[None, :] == label[grid.f[c]][:, None]) & ~far
        cap = grid.gen_capacity_at_bus
        p_pre = grid.incidence @ pre.flow  # net injections (KCL)
        pg = p_pre + pre.served  # generation per bus before the outage
        new_served, new_pg = served[bi].copy(), np.repeat(pg[None, :], bi.size, axis=0)
        for side in (far, near):
            demand = (side * load).sum(1, keepdims=True)
            capacity = (side * cap).sum(1, keepdims=True)
            frac = np.divide(capacity, demand, out=np.ones_like(demand), where=demand > 0)
            frac = frac.clip(max=1.0)
            util = np.divide(demand * frac, capacity, out=np.zeros_like(demand),
                             where=capacity > 0)
            new_served = np.where(side, load * frac, new_served)
            new_pg = np.where(side, cap * util, new_pg)
        flow[bi] = (new_pg - new_served) @ H.T
        flow[bi, c] = 0.0
        served[bi] = new_served
    return cont, flow, served

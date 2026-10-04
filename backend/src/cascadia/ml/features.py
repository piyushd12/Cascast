"""Feature construction shared by training and serving (numpy only, works batched).

Inputs describe a stable pre-contingency operating point plus the contingency to test:
    load     (..., N)  demand [MW]
    served   (..., N)  demand served before the contingency [MW]
    flow     (..., E)  branch flows before the contingency [MW]
    alive    (..., E)  bool, branches in service before the contingency
    trip     (..., E)  bool, branches tripped by the contingency
"""

from __future__ import annotations

import numpy as np

from cascadia.grid.model import Grid

NODE_FEATURES = ["served_pu", "demand_pu", "gen_capacity_pu", "net_injection_pu"]
EDGE_FEATURES = [
    "log_susceptance",
    "rating_pu",
    "signed_loading",
    "loading",
    "out_of_service",
    "contingency",
    "is_transformer",
]


def build_features(
    grid: Grid,
    load: np.ndarray,
    served: np.ndarray,
    flow: np.ndarray,
    alive: np.ndarray,
    trip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (node_x (..., N, 4), edge_x (..., E, 7)) as float32."""
    base = grid.base_mva
    batch = np.broadcast_shapes(load.shape[:-1], flow.shape[:-1], trip.shape[:-1])
    net_inj = flow @ grid.incidence.T  # KCL: injection = sum of outgoing flows
    node = np.stack(
        np.broadcast_arrays(
            served / base,
            load / base,
            np.broadcast_to(grid.gen_capacity_at_bus / base, (*batch, grid.n_bus)),
            net_inj / base,
        ),
        axis=-1,
    )
    signed = flow / grid.rating
    edge = np.stack(
        np.broadcast_arrays(
            np.log1p(grid.b) / 5.0,
            grid.rating / base,
            signed,
            np.abs(signed),
            ~alive,
            trip,
            grid.is_trafo,
        ),
        axis=-1,
    )
    return node.astype(np.float32), edge.astype(np.float32)

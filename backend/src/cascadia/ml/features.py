"""Feature construction shared by training and serving (numpy only, works batched).

Inputs describe a stable pre-contingency operating point plus the contingency to test:
    load     (..., N)  demand [MW]
    served   (..., N)  demand served before the contingency [MW]
    flow     (..., E)  branch flows before the contingency [MW]
    alive    (..., E)  bool, branches in service before the contingency
    trip     (..., E)  bool, branches tripped by the contingency
    flow1    (..., E)  flows right after the contingency, before any further trips [MW]
    served1  (..., N)  demand served right after the contingency [MW]

The first-stage quantities are exact DC physics computed cheaply with line outage
distribution factors (see grid.dcpf.n1_flows). The GNN learns everything after that:
which overloads trip, how flow redistributes again, and how far the cascade spreads.
"""

from __future__ import annotations

import numpy as np

from cascadia.grid.model import Grid

NODE_FEATURES = [
    "served_pu", "demand_pu", "gen_capacity_pu", "net_injection_pu", "served_after_trip_pu"
]
EDGE_FEATURES = [
    "log_susceptance",
    "rating_pu",
    "signed_loading",
    "loading",
    "out_of_service",
    "contingency",
    "is_transformer",
    "signed_loading_after_trip",
    "loading_after_trip",
    "overload_after_trip",
]


def build_features(
    grid: Grid,
    load: np.ndarray,
    served: np.ndarray,
    flow: np.ndarray,
    alive: np.ndarray,
    trip: np.ndarray,
    flow1: np.ndarray,
    served1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (node_x (..., N, 5), edge_x (..., E, 10)) as float32."""
    base = grid.base_mva
    batch = np.broadcast_shapes(load.shape[:-1], flow.shape[:-1], trip.shape[:-1],
                                flow1.shape[:-1], served1.shape[:-1])
    net_inj = flow @ grid.incidence.T  # KCL: injection = sum of outgoing flows
    node = np.stack(
        np.broadcast_arrays(
            served / base,
            load / base,
            np.broadcast_to(grid.gen_capacity_at_bus / base, (*batch, grid.n_bus)),
            net_inj / base,
            served1 / base,
        ),
        axis=-1,
    )
    signed = flow / grid.rating
    signed1 = flow1 / grid.rating
    edge = np.stack(
        np.broadcast_arrays(
            np.log1p(grid.b) / 5.0,
            grid.rating / base,
            signed,
            np.abs(signed),
            ~alive,
            trip,
            grid.is_trafo,
            signed1,
            np.abs(signed1),
            np.clip(np.abs(signed1) - 1.0, 0.0, 3.0),
        ),
        axis=-1,
    )
    return node.astype(np.float32), edge.astype(np.float32)

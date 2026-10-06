"""ONNX Runtime inference for CascadeGNN (no torch required)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cache

import numpy as np
import onnxruntime as ort

from cascast.config import settings
from cascast.grid.dcpf import FlowResult, dc_power_flow, n1_flows
from cascast.grid.model import Grid, load_grid
from cascast.ml.features import build_features


@dataclass
class Screening:
    """N-1 screen of a stable operating point."""

    contingency: np.ndarray  # (K,) branch tripped in each scenario
    prob: np.ndarray  # (K, E) failure probability of each branch under each contingency
    shed: np.ndarray  # (K,) predicted load-shed fraction
    vulnerability: np.ndarray  # (E,) worst-case failure probability of each branch
    latency_ms: float


class RiskModel:
    def __init__(self, grid: Grid, path=None, threads: int = 2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.grid = grid
        self.session = ort.InferenceSession(
            str(path or settings.artifacts_dir / "model.onnx"), opts,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, load, served, flow, alive, trip, flow1=None, served1=None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Batched prediction. `trip` is (B, E) bool; other inputs broadcast. -> (prob, shed).

        `flow1`/`served1` (first-stage physics) are computed here when not supplied.
        """
        if flow1 is None:
            first = [dc_power_flow(self.grid, alive & ~t, load) for t in trip]
            flow1 = np.stack([r.flow for r in first])
            served1 = np.stack([r.served for r in first])
        xv, xe = build_features(self.grid, load, served, flow, alive, trip, flow1, served1)
        if xv.shape[0] != xe.shape[0]:
            xv = np.broadcast_to(xv, (xe.shape[0], *xv.shape[1:])).copy()
        prob, shed = self.session.run(None, {"node_x": xv, "edge_x": xe})
        dead = ~alive | trip  # out-of-service or tripped branches can't fail again
        return np.where(dead, 0.0, prob), shed

    def screen(self, load: np.ndarray, alive: np.ndarray, pre: FlowResult) -> Screening:
        t0 = time.perf_counter()
        cont, flow1, served1 = n1_flows(self.grid, alive, load, pre)
        trip = np.zeros((cont.size, self.grid.n_branch), dtype=bool)
        trip[np.arange(cont.size), cont] = True
        prob, shed = self.predict(load, pre.served, pre.flow, alive, trip, flow1, served1)
        return Screening(
            contingency=cont,
            prob=prob,
            shed=shed,
            vulnerability=prob.max(0) if cont.size else np.zeros(self.grid.n_branch),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


@cache
def get_model() -> RiskModel:
    return RiskModel(load_grid())

"""In-memory grid model loaded from artifacts/grid.json (no pandapower needed at runtime)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path

import numpy as np

from cascadia.config import settings


@dataclass(frozen=True)
class Grid:
    name: str
    n_bus: int
    n_branch: int
    f: np.ndarray  # (E,) from-bus index
    t: np.ndarray  # (E,) to-bus index
    b: np.ndarray  # (E,) series susceptance 1/x [p.u.]
    rating: np.ndarray  # (E,) thermal limit [MW]
    load: np.ndarray  # (N,) nominal demand [MW]
    gen_bus: np.ndarray  # (G,) bus index of each generator
    gen_pmax: np.ndarray  # (G,) capacity [MW]
    xy: np.ndarray  # (N, 2) layout coordinates in [0, 1]
    is_trafo: np.ndarray  # (E,) bool
    base_mva: float = 100.0
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @cached_property
    def gen_capacity_at_bus(self) -> np.ndarray:
        return np.bincount(self.gen_bus, weights=self.gen_pmax, minlength=self.n_bus)

    @cached_property
    def incidence(self) -> np.ndarray:
        """Signed bus-branch incidence matrix A (N, E): +1 at from-bus, -1 at to-bus."""
        A = np.zeros((self.n_bus, self.n_branch), dtype=np.float32)
        idx = np.arange(self.n_branch)
        A[self.f, idx] = 1.0
        A[self.t, idx] = -1.0
        return A

    @classmethod
    def from_json(cls, path: Path) -> Grid:
        d = json.loads(path.read_text())
        buses, branches, gens = d["buses"], d["branches"], d["gens"]
        return cls(
            name=d["name"],
            n_bus=len(buses),
            n_branch=len(branches),
            f=np.array([br["from"] for br in branches], dtype=np.int64),
            t=np.array([br["to"] for br in branches], dtype=np.int64),
            b=np.array([1.0 / br["x"] for br in branches]),
            rating=np.array([br["rating_mw"] for br in branches]),
            load=np.array([bus["load_mw"] for bus in buses]),
            gen_bus=np.array([g["bus"] for g in gens], dtype=np.int64),
            gen_pmax=np.array([g["pmax_mw"] for g in gens]),
            xy=np.array([[bus["x"], bus["y"]] for bus in buses]),
            is_trafo=np.array([br["trafo"] for br in branches], dtype=bool),
            base_mva=d.get("base_mva", 100.0),
            raw=d,
        )


@cache
def load_grid() -> Grid:
    return Grid.from_json(settings.artifacts_dir / "grid.json")

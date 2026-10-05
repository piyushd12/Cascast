"""Generate labelled cascade scenarios with the physics simulator.

Each scenario:
  1. scale nominal demand by a stress factor and add per-bus noise,
  2. apply 0-2 pre-existing outages and let the grid settle (a stable but degraded state),
  3. trip 1-3 additional branches (the contingency) and simulate the cascade.

Usage:  uv run python -m cascadia.ml.datagen --n 100000
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Tiny matrices: one BLAS thread per worker process beats oversubscribing the CPU.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from cascadia.config import settings  # noqa: E402
from cascadia.grid.cascade import settle, simulate_cascade  # noqa: E402
from cascadia.grid.dcpf import dc_power_flow  # noqa: E402
from cascadia.grid.model import load_grid  # noqa: E402

FIELDS = ("load", "served", "flow", "alive", "trip", "flow1", "served1", "failed", "stage",
          "shed")


def sample_load(rng: np.random.Generator, nominal: np.ndarray) -> np.ndarray:
    # Half the scenarios are "normal" operation, half are stressed (peaks, heatwaves).
    scale = rng.uniform(0.7, 1.1) if rng.random() < 0.5 else rng.uniform(1.0, 1.45)
    return nominal * scale * rng.lognormal(0.0, 0.08, nominal.size)


def scenario(rng: np.random.Generator) -> dict[str, np.ndarray]:
    grid = load_grid()
    load = sample_load(rng, grid.load)
    alive = np.ones(grid.n_branch, dtype=bool)
    n_pre = rng.choice(3, p=[0.6, 0.25, 0.15])
    alive[rng.choice(grid.n_branch, n_pre, replace=False)] = False
    alive, pre = settle(grid, alive, load)

    candidates = np.flatnonzero(alive)
    k = min(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]), candidates.size)
    trip = np.zeros(grid.n_branch, dtype=bool)
    trip[rng.choice(candidates, k, replace=False)] = True
    res = simulate_cascade(grid, load, trip, alive=alive, pre=pre)
    first = dc_power_flow(grid, alive & ~trip, load)  # stage-1 physics (a GNN input)
    return {
        "load": load.astype(np.float32),
        "served": pre.served.astype(np.float32),
        "flow": pre.flow.astype(np.float32),
        "alive": alive,
        "trip": trip,
        "flow1": first.flow.astype(np.float32),
        "served1": first.served.astype(np.float32),
        "failed": res.failed,
        "stage": res.stage.astype(np.int8),
        "shed": np.float32(res.load_shed_frac),
    }


def _chunk(args: tuple[int, int]) -> dict[str, np.ndarray]:
    seed, n = args
    rng = np.random.default_rng(seed)
    rows = [scenario(rng) for _ in range(n)]
    return {k: np.stack([r[k] for r in rows]) for k in FIELDS}


def generate(n: int, seed: int = 0, workers: int | None = None) -> dict[str, np.ndarray]:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    chunk = 500
    jobs = [(seed * 1_000_003 + i, min(chunk, n - i * chunk)) for i in range(-(-n // chunk))]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_chunk, jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(settings.data_dir / "scenarios.npz"))
    a = ap.parse_args()

    t0 = time.perf_counter()
    data = generate(a.n, a.seed)
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **data)

    cascades = data["failed"].any(1)
    print(f"{a.n} scenarios in {dt:.1f}s -> {a.out}")
    print(f"  with cascade: {cascades.mean():.1%}   positive branch labels: "
          f"{data['failed'].mean():.2%}   mean shed: {data['shed'].mean():.3f}   "
          f"max stages: {data['stage'].max()}")


if __name__ == "__main__":
    main()

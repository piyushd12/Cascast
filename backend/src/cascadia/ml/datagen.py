"""Generate labelled cascade scenarios with the physics simulator.

Each scenario:
  1. scale nominal demand by a stress factor and add per-bus noise,
  2. apply 0-2 pre-existing outages and let the grid settle (a stable but degraded state),
  3. trip 1-3 additional branches (the contingency) and simulate one random realisation of
     the cascade under stochastic protection (see grid/cascade.py).

Each scenario is simulated DRAWS times with independent randomness. The labels are the
empirical failure frequency of every branch and the mean load shed: low-variance targets
for the probabilities that physics can only estimate by Monte Carlo. One raw outcome
(`failed`) is kept for ranking metrics.

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

FIELDS = ("load", "served", "flow", "alive", "trip", "flow1", "served1", "failed", "fail_prob",
          "shed")
DRAWS = 8


def sample_load(rng: np.random.Generator, nominal: np.ndarray) -> np.ndarray:
    # Half the scenarios are "normal" operation, half are stressed (peaks, heatwaves).
    scale = rng.uniform(0.75, 1.05) if rng.random() < 0.5 else rng.uniform(1.0, 1.28)
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
    runs = [simulate_cascade(grid, load, trip, alive=alive, pre=pre, rng=rng)
            for _ in range(DRAWS)]
    first = dc_power_flow(grid, alive & ~trip, load)  # stage-1 physics (a GNN input)
    return {
        "load": load.astype(np.float32),
        "served": pre.served.astype(np.float32),
        "flow": pre.flow.astype(np.float32),
        "alive": alive,
        "trip": trip,
        "flow1": first.flow.astype(np.float32),
        "served1": first.served.astype(np.float32),
        "failed": runs[0].failed,
        "fail_prob": np.mean([r.failed for r in runs], axis=0).astype(np.float16),
        "shed": np.float32(np.mean([r.load_shed_frac for r in runs])),
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

    print(f"{a.n} scenarios x {DRAWS} draws in {dt:.1f}s -> {a.out}")
    print(f"  cascade in first draw: {data['failed'].any(1).mean():.1%}   mean branch failure "
          f"prob: {data['fail_prob'].astype(np.float32).mean():.3%}   "
          f"mean expected shed: {data['shed'].mean():.4f}")


if __name__ == "__main__":
    main()

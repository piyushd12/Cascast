from dataclasses import asdict

import numpy as np

from cascadia.grid.dcpf import dc_power_flow
from cascadia.ml.inference import get_model
from cascadia.services.risk_engine import LiveAccuracy, process


def test_audit_totals_round_trip_through_a_redis_hash():
    acc = LiveAccuracy(checked=3, prob_abs_err=1.25, prob_n=40, shed_abs_err=0.01)
    h = {k.encode(): str(v).encode() for k, v in asdict(acc).items()}  # what HGETALL returns
    assert LiveAccuracy.from_hash(h) == acc
    assert LiveAccuracy.from_hash({**h, b"retired": b"1"}) == acc  # rolling deploys
    assert LiveAccuracy.from_hash({}).as_dict()["checked"] == 0


def test_process_audits_only_when_asked(grid):
    load, alive = grid.load, np.ones(grid.n_branch, dtype=bool)
    res = dc_power_flow(grid, alive, load)
    state = {"tick": 7, "minute": 0.0, "load": load.tolist(), "alive": alive.tolist(),
             "flow": res.flow.tolist(), "served": res.served.tolist(), "n_islands": 1}
    rng = np.random.default_rng(0)
    out, audit = process(grid, get_model(), state, rng, audit=False)
    assert audit is None and out["tick"] == 7 and out["n_screened"] == grid.n_branch
    _, audit = process(grid, get_model(), state, rng, audit=True)
    assert audit.checked >= 1

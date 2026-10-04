"""Export the IEEE 118-bus test case to artifacts/grid.json.

Run once (needs the `train` dependency group):  uv run python -m cascadia.grid.case

* Topology, reactances, loads and generator capacities come from pandapower's case118.
* Our DC solver is cross-checked against pandapower's `rundcpp` on the original dispatch.
* The case has no meaningful thermal ratings, so ratings are derived to make the grid
  N-1 secure at nominal load with a small margin: rating = MARGIN * max(|f_base|, max_k |f_k|)
  over all single-branch contingencies k. Cascades then require stress (high load or
  multiple outages), mirroring how real grids are planned.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from cascadia.config import settings
from cascadia.grid.dcpf import dc_power_flow, islands, solve_flows
from cascadia.grid.model import Grid

MARGIN = 1.05
MIN_RATING_MW = 10.0
# MATPOWER column indices
PD, BUS_I, GEN_BUS, PG, PMAX, GEN_STATUS = 2, 0, 0, 1, 8, 7
F_BUS, T_BUS, BR_X, TAP, SHIFT, PF = 0, 1, 3, 8, 9, 13


def build() -> dict:
    net = pn.case118()
    pp.rundcpp(net)
    ppc = net._ppc
    bus, gen, branch = ppc["bus"], ppc["gen"], ppc["branch"]
    assert np.allclose(branch[:, SHIFT], 0), "phase shifters not modelled"

    ppc_to_pd = {int(v): int(k) for k, v in enumerate(net._pd2ppc_lookups["bus"]) if v >= 0}
    names = [str(net.bus.name.iloc[ppc_to_pd[i]]) for i in range(len(bus))]

    tap = np.where(branch[:, TAP].real == 0, 1.0, branch[:, TAP].real)
    x = branch[:, BR_X].real * tap  # DC model: b = 1 / (x * tap), as in MATPOWER's makeBdc
    f, t = branch[:, F_BUS].real.astype(int), branch[:, T_BUS].real.astype(int)
    n_line = len(net.line)

    on = gen[:, GEN_STATUS].real > 0
    gen_bus = gen[on, GEN_BUS].real.astype(int)
    pmax = gen[on, PMAX].real
    # The slack (ext_grid) has no sensible limit in the case; cap it at its dispatch x 2.
    pmax = np.where(pmax > 5000, 2 * np.abs(gen[on, PG].real), pmax)

    d = {
        "name": "IEEE 118-bus",
        "base_mva": float(ppc["baseMVA"]),
        "buses": [
            {"id": i, "name": names[i], "load_mw": float(bus[i, PD].real)} for i in range(len(bus))
        ],
        "branches": [
            {"id": e, "from": int(f[e]), "to": int(t[e]), "x": float(x[e]), "trafo": e >= n_line}
            for e in range(len(branch))
        ],
        "gens": [{"bus": int(b), "pmax_mw": float(p)} for b, p in zip(gen_bus, pmax, strict=True)],
    }

    # --- Cross-check our solver against pandapower on the original dispatch -------------
    tmp = _grid(d, rating=np.ones(len(branch)), xy=np.zeros((len(bus), 2)))
    p_bus = np.bincount(gen_bus, weights=gen[on, PG].real, minlength=len(bus)) - bus[:, PD].real
    alive = np.ones(len(branch), dtype=bool)
    n_comp, label = islands(tmp, alive)
    ours, _ = solve_flows(tmp, alive, p_bus, label, n_comp)
    err = float(np.abs(ours - branch[:, PF].real).max())
    assert err < 1e-6, f"DC flow mismatch vs pandapower: {err} MW"

    # --- Ratings: N-1 secure at nominal load ----------------------------------------
    base = dc_power_flow(tmp, alive, tmp.load).flow
    worst = np.abs(base)
    for k in range(len(branch)):
        a = alive.copy()
        a[k] = False
        worst = np.maximum(worst, np.abs(dc_power_flow(tmp, a, tmp.load).flow))
    rating = np.maximum(MARGIN * worst, MIN_RATING_MW).round(1)

    # --- Layout for the dashboard -----------------------------------------------------
    g = nx.Graph()
    g.add_nodes_from(range(len(bus)))
    g.add_edges_from(zip(f, t, strict=True))
    pos = nx.kamada_kawai_layout(g)
    xy = np.array([pos[i] for i in range(len(bus))])
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    for e, br in enumerate(d["branches"]):
        br["rating_mw"] = float(rating[e])
        br["base_flow_mw"] = round(float(base[e]), 3)
    for i, b in enumerate(d["buses"]):
        b["x"], b["y"] = round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)
    d["validation"] = {"max_flow_error_vs_pandapower_mw": err, "rating_margin": MARGIN}
    return d


def _grid(d: dict, rating: np.ndarray, xy: np.ndarray) -> Grid:
    br, gens = d["branches"], d["gens"]
    return Grid(
        name=d["name"],
        n_bus=len(d["buses"]),
        n_branch=len(br),
        f=np.array([b["from"] for b in br]),
        t=np.array([b["to"] for b in br]),
        b=np.array([1 / b["x"] for b in br]),
        rating=rating,
        load=np.array([b["load_mw"] for b in d["buses"]]),
        gen_bus=np.array([g["bus"] for g in gens]),
        gen_pmax=np.array([g["pmax_mw"] for g in gens]),
        xy=xy,
        is_trafo=np.array([b["trafo"] for b in br]),
    )


if __name__ == "__main__":
    out = settings.artifacts_dir / "grid.json"
    data = build()
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out}: {len(data['buses'])} buses, {len(data['branches'])} branches, "
          f"{len(data['gens'])} gens; validation {data['validation']}")

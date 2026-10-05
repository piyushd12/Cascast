"""Evaluate CascadeGNN against held-out data and against Monte Carlo physics, benchmark it,
and render the README figures (light + dark variants).

Two kinds of ground truth:
  * realised outcomes on the 10k-scenario test split (one random cascade each), used for
    ranking (PR-AUC, ROC-AUC) and calibration (Brier, ECE, reliability diagram);
  * Monte Carlo reference probabilities on a subset (256 cascades per scenario), used to ask
    "how many Monte Carlo runs would match the GNN's accuracy, and how long would they take?"

Usage:  uv run --group train python -m cascadia.ml.evaluate
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from cascadia.config import BACKEND_DIR, settings  # noqa: E402
from cascadia.grid.cascade import settle, simulate_cascade  # noqa: E402
from cascadia.grid.dcpf import dc_power_flow, n1_flows  # noqa: E402
from cascadia.grid.model import load_grid  # noqa: E402
from cascadia.ml.features import build_features  # noqa: E402
from cascadia.ml.train import CKPT, build_model, load_dataset, predict, split  # noqa: E402

FIG_DIR = BACKEND_DIR.parent / "docs" / "figures"
MC_SCENARIOS = 400
MC_RUNS = 256  # first half: estimates with k runs; second half: independent reference
MC_K = [1, 2, 4, 8, 16, 32, 64, 128]
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}


# --------------------------------------------------------------------------- Monte Carlo
def _mc_job(args) -> tuple[np.ndarray, np.ndarray]:
    seed, load, alive, trip = args
    grid, rng = load_grid(), np.random.default_rng(seed)
    pre = dc_power_flow(grid, alive, load)
    failed = np.zeros((MC_RUNS, grid.n_branch), dtype=bool)
    shed = np.zeros(MC_RUNS)
    for r in range(MC_RUNS):
        res = simulate_cascade(grid, load, trip, alive=alive, pre=pre, rng=rng)
        failed[r], shed[r] = res.failed, res.load_shed_frac
    return failed, shed


def monte_carlo_reference(data: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jobs = [(int(i), data["load"][i].astype(np.float64), data["alive"][i], data["trip"][i])
            for i in idx]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as ex:
        out = list(ex.map(_mc_job, jobs, chunksize=4))
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


# --------------------------------------------------------------------------- benchmark
def benchmark(session: ort.InferenceSession, repeats: int = 5) -> dict:
    """Full N-1 screen of a stressed operating point: GNN vs physics."""
    grid, rng = load_grid(), np.random.default_rng(1)
    load = grid.load * 1.15
    alive, pre = settle(grid, np.ones(grid.n_branch, dtype=bool), load)
    cont = np.flatnonzero(alive)

    def gnn():  # includes the exact first-stage physics (LODF) the GNN consumes
        c, flow1, served1 = n1_flows(grid, alive, load, pre)
        trip = np.eye(grid.n_branch, dtype=bool)[c]
        xv, xe = build_features(grid, load, pre.served, pre.flow, alive, trip, flow1, served1)
        session.run(None, {"node_x": xv, "edge_x": xe})

    def physics_pass():  # one random cascade per contingency
        for e in cont:
            simulate_cascade(grid, load, [e], alive=alive, pre=pre, rng=rng)

    def best_of(fn, n):
        fn()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times) * 1000

    gnn_ms = best_of(gnn, repeats)
    pass_ms = best_of(physics_pass, 3)
    return {"contingencies": int(cont.size), "gnn_ms": gnn_ms, "physics_pass_ms": pass_ms,
            "mc100_ms": 100 * pass_ms, "speedup_vs_mc100": 100 * pass_ms / gnn_ms}


# --------------------------------------------------------------------------- figures
def style(ax, th):
    ax.set_facecolor(th["surface"])
    ax.grid(True, color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=9)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])
    ax.title.set_color(th["ink"])


def save(fig, name: str, mode: str, th):
    fig.patch.set_facecolor(th["surface"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}-{mode}.png", dpi=160, facecolor=th["surface"])
    plt.close(fig)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, ece = [], [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            ece += m.mean() * abs(conf[-1] - acc[-1])
    return np.array(conf), np.array(acc), float(ece)


def figures(history, prec, rec, conf, acc, gnn_p, ref_p, gnn_s, ref_s, curve, gnn_err, bench):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for mode, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([h["epoch"] for h in history], [h["val_pr_auc"] for h in history],
                color=th["s1"], lw=2)
        ax.set(xlabel="Epoch", ylabel="Validation PR-AUC", title="Training progress")
        style(ax, th)
        save(fig, "training", mode, th)

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(rec, prec, color=th["s1"], lw=2)
        ax.set(xlabel="Recall", ylabel="Precision", title="Branch-failure ranking (test)",
               xlim=(0, 1), ylim=(0, 1.02))
        style(ax, th)
        save(fig, "pr_curve", mode, th)

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0, 1], [0, 1], color=th["axis"], lw=1, ls="--", label="Perfect calibration")
        ax.plot(conf, acc, color=th["s1"], lw=2, marker="o", ms=5, label="CascadeGNN")
        ax.set(xlabel="Predicted failure probability", ylabel="Observed failure rate",
               title="Calibration (test)", xlim=(0, 1), ylim=(0, 1))
        style(ax, th)
        ax.legend(frameon=False, labelcolor=th["ink2"], fontsize=9)
        save(fig, "calibration", mode, th)

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        m = (ref_p > 0.005) | (gnn_p > 0.005)
        ax.scatter(ref_p[m], gnn_p[m], s=5, color=th["s1"], alpha=0.3, linewidths=0)
        ax.plot([0, 1], [0, 1], color=th["axis"], lw=1, ls="--")
        ax.set(xlabel="Monte Carlo probability (128 runs)", ylabel="GNN probability",
               title="Per-branch failure probability", xlim=(0, 1), ylim=(0, 1))
        style(ax, th)
        save(fig, "mc_agreement", mode, th)

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(ref_s * 100, gnn_s * 100, s=9, color=th["s1"], alpha=0.5, linewidths=0)
        lim = max(2.0, float(np.percentile(ref_s, 99.5)) * 110)
        ax.plot([0, lim], [0, lim], color=th["axis"], lw=1, ls="--")
        ax.set(xlabel="Monte Carlo expected load shed (%)", ylabel="GNN expected load shed (%)",
               title="Expected blackout size", xlim=(0, lim), ylim=(0, lim))
        style(ax, th)
        save(fig, "load_shed", mode, th)

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        xs = [c["n1_seconds"] for c in curve]
        ax.plot(xs, [c["prob_mae"] for c in curve], color=th["s2"], lw=2, marker="o", ms=5,
                label="Monte Carlo (labels = runs)")
        for c in curve:
            ax.annotate(f"{c['runs']}", (c["n1_seconds"], c["prob_mae"]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8, color=th["muted"])
        gx = bench["gnn_ms"] / 1000
        ax.scatter([gx], [gnn_err], s=90, marker="*", color=th["s1"], zorder=3,
                   label="CascadeGNN")
        ax.annotate(f"GNN  {bench['gnn_ms']:.0f} ms", (gx, gnn_err), xytext=(8, -12),
                    textcoords="offset points", fontsize=9, color=th["ink2"])
        ax.set_xscale("log")
        ax.set(xlabel=f"Time to screen all {bench['contingencies']} contingencies (s, log)",
               ylabel="Error in failure probability\n(MAE vs reference)",
               title="Accuracy vs compute")
        style(ax, th)
        ax.legend(frameon=False, labelcolor=th["ink2"], fontsize=9, loc="upper right")
        save(fig, "speedup", mode, th)


# --------------------------------------------------------------------------- main
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = load_grid()
    data = load_dataset(settings.data_dir / "scenarios.npz")
    test = split(len(data["shed"]))["test"]
    meta = json.loads((settings.data_dir / "train_meta.json").read_text())
    a, b = meta["platt_scale"], meta["platt_bias"]

    model = build_model(grid, hidden=meta["hidden"], layers=meta["layers"]).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))

    # 1) Realised outcomes on the full test split
    logits, labels, shed_p, shed_t = predict(model, grid, data, test, device)
    p = torch.sigmoid(a * logits + b).numpy()
    y = labels.numpy().astype(int)
    conf, acc, ece = reliability(p, y)
    prec, rec, _ = precision_recall_curve(y, p)

    # 2) Monte Carlo reference on a subset
    sub = test[:MC_SCENARIOS]
    t0 = time.perf_counter()
    failed, shed = monte_carlo_reference(data, sub)
    print(f"Monte Carlo reference: {sub.size} scenarios x {MC_RUNS} runs "
          f"in {time.perf_counter() - t0:.0f}s")
    half = MC_RUNS // 2
    ref_p, ref_s = failed[:, half:].mean(1), shed[:, half:].mean(1)
    sl, _, gnn_s, _ = predict(model, grid, data, sub, device)
    valid = data["alive"][sub] & ~data["trip"][sub]
    gnn_full = np.zeros(valid.shape)
    gnn_full[valid] = torch.sigmoid(a * sl + b).numpy()
    gnn_s = gnn_s.numpy()
    active = valid & ((ref_p > 0.005) | (gnn_full > 0.005))
    gnn_err = float(np.abs(gnn_full - ref_p)[active].mean())

    session = ort.InferenceSession(str(settings.artifacts_dir / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    bench = benchmark(session)
    curve = []
    for k in MC_K:
        est_p, est_s = failed[:, :k].mean(1), shed[:, :k].mean(1)
        curve.append({"runs": k,
                      "prob_mae": float(np.abs(est_p - ref_p)[active].mean()),
                      "shed_mae_pct": float(np.abs(est_s - ref_s).mean() * 100),
                      "n1_seconds": k * bench["physics_pass_ms"] / 1000})
    # Smallest Monte Carlo budget at least as accurate as the GNN (interpolated in log-runs)
    errs = np.array([c["prob_mae"] for c in curve])
    if gnn_err >= errs[0]:
        equiv = 1.0
    elif gnn_err <= errs[-1]:
        equiv = float(MC_K[-1])
    else:
        j = int(np.argmax(errs <= gnn_err))
        lo, hi = np.log2(MC_K[j - 1]), np.log2(MC_K[j])
        w = (errs[j - 1] - gnn_err) / (errs[j - 1] - errs[j])
        equiv = float(2 ** (lo + w * (hi - lo)))
    bench["equivalent_mc_runs"] = equiv
    bench["equivalent_mc_ms"] = equiv * bench["physics_pass_ms"]
    bench["speedup_vs_equivalent"] = bench["equivalent_mc_ms"] / bench["gnn_ms"]

    metrics = {
        "test_scenarios": int(test.size),
        "branch": {
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "ece": ece,
            "positive_rate": float(y.mean()),
        },
        "monte_carlo": {
            "scenarios": int(sub.size),
            "reference_runs": half,
            "prob_mae": gnn_err,
            "prob_corr": float(np.corrcoef(gnn_full[active], ref_p[active])[0, 1]),
            "shed_mae_pct": float(np.abs(gnn_s - ref_s).mean() * 100),
            "shed_corr": float(np.corrcoef(gnn_s, ref_s)[0, 1]),
            "curve": curve,
        },
        "load_shed": {"mae_pct_vs_realised": float(np.abs(shed_p - shed_t).mean().item() * 100)},
        "benchmark": bench,
        "model": {k: meta[k] for k in ("params", "hidden", "layers", "platt_scale",
                                       "platt_bias", "epochs")},
        "history": [{k: h[k] for k in ("epoch", "train_loss", "val_pr_auc", "val_shed_mae")}
                    for h in meta["history"]],
    }
    (settings.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    figures(meta["history"], prec, rec, conf, acc, gnn_full[active], ref_p[active], gnn_s,
            ref_s, curve, gnn_err, bench)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()

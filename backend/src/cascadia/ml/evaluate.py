"""Evaluate the trained model on the held-out test split, benchmark it against the physics
simulator, and render README figures (light + dark variants).

Usage:  uv run --group train python -m cascadia.ml.evaluate
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cascadia.config import BACKEND_DIR, settings
from cascadia.grid.cascade import simulate_cascade
from cascadia.grid.dcpf import dc_power_flow, n1_flows
from cascadia.grid.model import load_grid
from cascadia.ml.features import build_features
from cascadia.ml.train import CKPT, build_model, load_dataset, predict, split

FIG_DIR = BACKEND_DIR.parent / "docs" / "figures"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, err = np.zeros(bins), np.zeros(bins), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf[b], acc[b] = p[m].mean(), y[m].mean()
            err += m.mean() * abs(conf[b] - acc[b])
    return float(err), conf, acc


def benchmark(session: ort.InferenceSession, repeats: int = 5) -> dict:
    """N-1 screening of every branch at nominal load: physics loop vs one ONNX batch."""
    grid = load_grid()
    load, alive = grid.load, np.ones(grid.n_branch, dtype=bool)
    pre = dc_power_flow(grid, alive, load)

    def physics():
        for e in range(grid.n_branch):
            simulate_cascade(grid, load, [e], alive=alive, pre=pre)

    def gnn():  # includes the exact first-stage physics (LODF) the GNN consumes
        cont, flow1, served1 = n1_flows(grid, alive, load, pre)
        trip = np.eye(grid.n_branch, dtype=bool)[cont]
        xv, xe = build_features(grid, load, pre.served, pre.flow, alive, trip, flow1, served1)
        session.run(None, {"node_x": xv, "edge_x": xe})

    def best_of(fn):
        fn()  # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times) * 1000

    phys_ms, gnn_ms = best_of(physics), best_of(gnn)
    return {"contingencies": grid.n_branch, "physics_ms": phys_ms, "gnn_onnx_ms": gnn_ms,
            "speedup": phys_ms / gnn_ms}


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


def figures(history, prec, rec, conf, acc, shed_p, shed_t, bench):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for mode, th in THEMES.items():
        # Training curve: validation PR-AUC per epoch
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ep = [h["epoch"] for h in history]
        ax.plot(ep, [h["val_pr_auc"] for h in history], color=th["s1"], lw=2)
        ax.set(xlabel="Epoch", ylabel="Validation PR-AUC", title="Training progress")
        style(ax, th)
        save(fig, "training", mode, th)

        # Precision-recall curve
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(rec, prec, color=th["s1"], lw=2)
        ax.set(xlabel="Recall", ylabel="Precision", title="Branch-failure prediction (test)",
               xlim=(0, 1), ylim=(0, 1.02))
        style(ax, th)
        save(fig, "pr_curve", mode, th)

        # Reliability diagram
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0, 1], [0, 1], color=th["axis"], lw=1, ls="--", label="Perfect calibration")
        m = conf > 0
        ax.plot(conf[m], acc[m], color=th["s1"], lw=2, marker="o", ms=5, label="CascadeGNN")
        ax.set(xlabel="Predicted probability", ylabel="Observed failure rate",
               title="Calibration (test)", xlim=(0, 1), ylim=(0, 1))
        style(ax, th)
        ax.legend(frameon=False, labelcolor=th["ink2"], fontsize=9)
        save(fig, "calibration", mode, th)

        # Load-shed: predicted vs simulated
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(shed_t * 100, shed_p * 100, s=6, color=th["s1"], alpha=0.35, linewidths=0)
        ax.plot([0, 100], [0, 100], color=th["axis"], lw=1, ls="--")
        lim = max(5.0, float(np.percentile(shed_t, 99.9)) * 110)
        ax.set(xlabel="Simulated load shed (%)", ylabel="Predicted load shed (%)",
               title="Blackout size (test)", xlim=(0, lim), ylim=(0, lim))
        style(ax, th)
        save(fig, "load_shed", mode, th)

        # Speedup bar
        fig, ax = plt.subplots(figsize=(6, 2.4))
        vals = [bench["physics_ms"], bench["gnn_onnx_ms"]]
        bars = ax.barh(["Physics simulator", "CascadeGNN (ONNX)"], vals,
                       color=[th["s2"], th["s1"]], height=0.5)
        ax.set_xscale("log")
        ax.set(xlabel="Time for full N-1 screen, ms (log scale)",
               title=f"Screening all {bench['contingencies']} contingencies "
                     f"— {bench['speedup']:.0f}× faster")
        for b, v in zip(bars, vals, strict=True):
            ax.text(v * 1.1, b.get_y() + b.get_height() / 2, f"{v:.1f} ms", va="center",
                    color=th["ink2"], fontsize=9)
        ax.set_xlim(min(vals) / 3, max(vals) * 6)
        style(ax, th)
        ax.tick_params(axis="y", colors=th["ink2"])
        save(fig, "speedup", mode, th)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = load_grid()
    data = load_dataset(settings.data_dir / "scenarios.npz")
    test = split(len(data["shed"]))["test"]
    meta = json.loads((settings.data_dir / "train_meta.json").read_text())

    model = build_model(grid, hidden=meta["hidden"], layers=meta["layers"]).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    logits, labels, shed_p, shed_t = predict(model, grid, data, test, device)
    p = torch.sigmoid(logits / meta["temperature"]).numpy()
    y = labels.numpy().astype(int)
    shed_p, shed_t = shed_p.numpy(), shed_t.numpy()
    pred = p >= 0.5

    # Scenario level: does the model flag a cascade (any branch >= 0.5) when one happens?
    valid = data["alive"][test] & ~data["trip"][test]
    per_scn_p = np.zeros(valid.shape)
    per_scn_p[valid] = p
    cascade_true = data["failed"][test].any(1)
    cascade_pred = (per_scn_p >= 0.5).any(1)

    err, conf, acc = ece(p, y)
    prec, rec, _ = precision_recall_curve(y, p)
    session = ort.InferenceSession(str(settings.artifacts_dir / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    bench = benchmark(session)

    metrics = {
        "test_scenarios": int(test.size),
        "branch": {
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "ece": err,
            "positive_rate": float(y.mean()),
        },
        "scenario": {
            "cascade_rate": float(cascade_true.mean()),
            "cascade_detection_accuracy": float((cascade_true == cascade_pred).mean()),
            "cascade_recall": float(cascade_pred[cascade_true].mean()),
            "false_alarm_rate": float(cascade_pred[~cascade_true].mean()),
        },
        "load_shed": {
            "mae_pct": float(np.abs(shed_p - shed_t).mean() * 100),
            "mae_pct_on_cascades": float(np.abs(shed_p - shed_t)[cascade_true].mean() * 100),
            "corr": float(np.corrcoef(shed_p, shed_t)[0, 1]),
        },
        "benchmark": bench,
        "model": {k: meta[k] for k in ("params", "hidden", "layers", "temperature", "epochs")},
        "history": [{k: h[k] for k in ("epoch", "train_loss", "val_pr_auc", "val_shed_mae")}
                    for h in meta["history"]],
    }
    Path(settings.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    figures(meta["history"], prec, rec, conf, acc, shed_p, shed_t, bench)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()

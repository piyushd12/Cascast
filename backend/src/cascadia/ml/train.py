"""Train CascadeGNN on simulator-generated scenarios, then calibrate it.

Usage:  uv run --group train python -m cascadia.ml.train --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from cascadia.config import settings
from cascadia.grid.model import Grid, load_grid
from cascadia.ml.features import EDGE_FEATURES, NODE_FEATURES, build_features
from cascadia.ml.model import CascadeGNN

CKPT = settings.data_dir / "model.pt"


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def split(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}


def batches(grid: Grid, data: dict, idx: np.ndarray, bs: int, device: str,
            shuffle: bool = False) -> Iterator[tuple[torch.Tensor, ...]]:
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, idx.size, bs):
        j = idx[i:i + bs]
        xv, xe = build_features(grid, data["load"][j], data["served"][j], data["flow"][j],
                                data["alive"][j], data["trip"][j], data["flow1"][j],
                                data["served1"][j])
        valid = data["alive"][j] & ~data["trip"][j]  # only in-service, untripped lines can fail
        yield tuple(torch.from_numpy(np.asarray(a)).to(device, non_blocking=True) for a in
                    (xv, xe, data["fail_prob"][j].astype(np.float32), valid, data["shed"][j],
                     data["failed"][j].astype(np.float32)))


def build_model(grid: Grid, **kw) -> CascadeGNN:
    return CascadeGNN(torch.from_numpy(grid.f), torch.from_numpy(grid.t), grid.n_bus,
                      len(NODE_FEATURES), len(EDGE_FEATURES), **kw)


def loss_fn(edge_logit, shed_logit, y, valid, shed):
    """Cross-entropy is a proper scoring rule: on random outcomes it learns probabilities."""
    bce = F.binary_cross_entropy_with_logits(edge_logit, y, reduction="none")
    edge_loss = (bce * valid).sum() / valid.sum()
    shed_loss = F.binary_cross_entropy_with_logits(shed_logit, shed)
    return edge_loss + 2.0 * shed_loss


@torch.no_grad()
def predict(model, grid, data, idx, device, bs=1024, soft=False):
    """Return (edge logits on valid edges, labels on valid edges, shed pred, shed true).

    Labels are the raw single-draw outcomes, or the multi-draw frequencies if `soft`.
    """
    model.eval()
    logits, labels, sp, st = [], [], [], []
    for xv, xe, prob, valid, shed, raw in batches(grid, data, idx, bs, device):
        el, sl = model(xv, xe)
        logits.append(el[valid].float().cpu())
        labels.append((prob if soft else raw)[valid].cpu())
        sp.append(torch.sigmoid(sl).float().cpu())
        st.append(shed.cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(sp), torch.cat(st)


def fit_platt(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Platt scaling: p = sigmoid(a * logit + b), fitted on held-out validation data."""
    a = torch.ones((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * logits + b, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return a.item(), b.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(settings.data_dir / "scenarios.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = load_grid()
    data = load_dataset(Path(a.data))
    parts = split(len(data["shed"]))

    tr = parts["train"]
    valid = data["alive"][tr] & ~data["trip"][tr]
    pos = data["fail_prob"][tr][valid].astype(np.float32).mean()
    print(f"device={device} train={tr.size} mean failure probability={pos:.4f}")

    model = build_model(grid, hidden=a.hidden, layers=a.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * -(-tr.size // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    history, best = [], float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, nb = time.perf_counter(), 0.0, 0
        for xv, xe, y, v, shed, _ in batches(grid, data, tr, a.bs, device, shuffle=True):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                el, sl = model(xv, xe)
            loss = loss_fn(el.float(), sl.float(), y, v, shed)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, nb = total + loss.item(), nb + 1

        lg, lb, sp, st = predict(model, grid, data, parts["val"], device, soft=True)
        # Select on validation cross-entropy over every valid edge: a proper scoring rule on a
        # fixed set. (A mask that depends on the model's own predictions can be gamed by an
        # underconfident model.) Probability MAE is reported on edges that can actually fail.
        val_loss = F.binary_cross_entropy_with_logits(lg, lb).item()
        prob_mae = (torch.sigmoid(lg) - lb).abs()[lb > 0].mean().item()
        _, raw, _, _ = predict(model, grid, data, parts["val"][:2000], device)
        ap_val = average_precision_score(raw.numpy(), lg[: raw.numel()].numpy())
        mae = (sp - st).abs().mean().item()
        history.append({"epoch": epoch, "train_loss": total / nb, "val_loss": val_loss,
                        "val_pr_auc": ap_val,
                        "val_prob_mae": prob_mae, "val_shed_mae": mae,
                        "seconds": time.perf_counter() - t0})
        print(f"epoch {epoch:3d}  loss {total / nb:.4f}  val loss {val_loss:.5f}  "
              f"val prob MAE {prob_mae:.4f}  PR-AUC {ap_val:.4f}  shed MAE {mae:.4f}  "
              f"({history[-1]['seconds']:.0f}s)")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), CKPT)

    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    lg, lb, _, _ = predict(model, grid, data, parts["val"], device, soft=True)
    scale, bias = fit_platt(lg, lb)
    meta = {"hidden": a.hidden, "layers": a.layers, "params": n_params,
            "platt_scale": scale, "platt_bias": bias, "epochs": a.epochs,
            "best_val_loss": best, "history": history}
    (settings.data_dir / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"best val loss {best:.5f}; Platt a={scale:.3f} b={bias:.3f}; "
          f"{n_params:,} params -> {CKPT}")


if __name__ == "__main__":
    main()

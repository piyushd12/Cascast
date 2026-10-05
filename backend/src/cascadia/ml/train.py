"""Train CascadeGNN on simulator-generated scenarios, then calibrate it.

Usage:  uv run --group train python -m cascadia.ml.train --epochs 30
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
                                data["alive"][j], data["trip"][j])
        valid = data["alive"][j] & ~data["trip"][j]  # only in-service, untripped lines can fail
        yield tuple(torch.from_numpy(np.asarray(a)).to(device, non_blocking=True) for a in
                    (xv, xe, data["failed"][j].astype(np.float32), valid, data["shed"][j]))


def build_model(grid: Grid, **kw) -> CascadeGNN:
    return CascadeGNN(torch.from_numpy(grid.f), torch.from_numpy(grid.t), grid.n_bus,
                      len(NODE_FEATURES), len(EDGE_FEATURES), **kw)


def loss_fn(edge_logit, shed_logit, y, valid, shed, pos_weight):
    bce = F.binary_cross_entropy_with_logits(edge_logit, y, pos_weight=pos_weight, reduction="none")
    edge_loss = (bce * valid).sum() / valid.sum()
    shed_loss = F.binary_cross_entropy_with_logits(shed_logit, shed)
    return edge_loss + 2.0 * shed_loss


@torch.no_grad()
def predict(model, grid, data, idx, device, bs=1024):
    """Return (edge logits on valid edges, labels on valid edges, shed pred, shed true)."""
    model.eval()
    logits, labels, sp, st = [], [], [], []
    for xv, xe, y, valid, shed in batches(grid, data, idx, bs, device):
        el, sl = model(xv, xe)
        logits.append(el[valid].float().cpu())
        labels.append(y[valid].cpu())
        sp.append(torch.sigmoid(sl).float().cpu())
        st.append(shed.cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(sp), torch.cat(st)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(settings.data_dir / "scenarios.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
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
    pos = data["failed"][tr][valid].mean()
    pos_weight = torch.tensor(min(20.0, (1 - pos) / pos), device=device)
    print(f"device={device} train={tr.size} positive rate={pos:.4f} pos_weight={pos_weight:.1f}")

    model = build_model(grid, hidden=a.hidden, layers=a.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * -(-tr.size // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    history, best = [], -1.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, nb = time.perf_counter(), 0.0, 0
        for xv, xe, y, v, shed in batches(grid, data, tr, a.bs, device, shuffle=True):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                el, sl = model(xv, xe)
            loss = loss_fn(el.float(), sl.float(), y, v, shed, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, nb = total + loss.item(), nb + 1

        lg, lb, sp, st = predict(model, grid, data, parts["val"], device)
        ap_val = average_precision_score(lb.numpy(), lg.numpy())
        mae = (sp - st).abs().mean().item()
        history.append({"epoch": epoch, "train_loss": total / nb, "val_pr_auc": ap_val,
                        "val_shed_mae": mae, "seconds": time.perf_counter() - t0})
        print(f"epoch {epoch:3d}  loss {total / nb:.4f}  val PR-AUC {ap_val:.4f}  "
              f"shed MAE {mae:.4f}  ({history[-1]['seconds']:.0f}s)")
        if ap_val > best:
            best = ap_val
            torch.save(model.state_dict(), CKPT)

    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    lg, lb, _, _ = predict(model, grid, data, parts["val"], device)
    temp = fit_temperature(lg, lb)
    model.temperature.fill_(temp)
    torch.save(model.state_dict(), CKPT)
    meta = {"hidden": a.hidden, "layers": a.layers, "params": n_params, "temperature": temp,
            "epochs": a.epochs, "best_val_pr_auc": best, "history": history,
            "pos_weight": float(pos_weight)}
    (settings.data_dir / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"best val PR-AUC {best:.4f}; temperature {temp:.3f}; {n_params:,} params -> {CKPT}")


if __name__ == "__main__":
    main()

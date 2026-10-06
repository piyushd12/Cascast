"""Export the calibrated model to ONNX and verify parity with PyTorch.

Usage:  uv run --group train python -m cascast.ml.export
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import torch

from cascast.config import settings
from cascast.grid.model import load_grid
from cascast.ml.features import EDGE_FEATURES, NODE_FEATURES
from cascast.ml.model import Exported
from cascast.ml.train import CKPT, build_model


def main() -> None:
    grid = load_grid()
    meta = json.loads((settings.data_dir / "train_meta.json").read_text())
    model = build_model(grid, hidden=meta["hidden"], layers=meta["layers"])
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    wrapped = Exported(model, meta["platt_scale"], meta["platt_bias"]).eval()

    xv = torch.randn(4, grid.n_bus, len(NODE_FEATURES))
    xe = torch.randn(4, grid.n_branch, len(EDGE_FEATURES))
    out = settings.artifacts_dir / "model.onnx"
    torch.onnx.export(
        wrapped, (xv, xe), out,
        input_names=["node_x", "edge_x"],
        output_names=["failure_prob", "load_shed"],
        dynamic_axes={"node_x": {0: "batch"}, "edge_x": {0: "batch"},
                      "failure_prob": {0: "batch"}, "load_shed": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    xv, xe = torch.randn(32, *xv.shape[1:]), torch.randn(32, *xe.shape[1:])
    got = sess.run(None, {"node_x": xv.numpy(), "edge_x": xe.numpy()})
    with torch.no_grad():
        ref = [t.numpy() for t in wrapped(xv, xe)]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(got, ref, strict=True))
    assert diff < 1e-4, f"ONNX parity failed: {diff}"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), max |onnx - torch| = {diff:.2e}")


if __name__ == "__main__":
    main()

"""CascadeGNN: an edge-aware message-passing network with a global context vector.

The topology is fixed (out-of-service branches are a feature, not removed edges), so the
graph structure is baked in as constant gather indices and scatter matrices. Every op is a
dense batched tensor op, which keeps inference fast and makes ONNX export trivial.

Per layer (all residual, pre-LayerNorm MLPs):
    edges  : h_e <- h_e + phi_e([h_e, h_from, h_to, g])
    nodes  : h_v <- h_v + phi_v([h_v, sum_out h_e, sum_in h_e, g])
    global : g   <- g   + phi_g([g, mean h_v, mean h_e])
Heads: per-branch failure logit, and graph-level load-shed fraction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class Block(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.edge = mlp(4 * h, h, 2 * h)
        self.node = mlp(4 * h, h, 2 * h)
        self.glob = mlp(3 * h, h, 2 * h)

    def forward(self, hv: Tensor, he: Tensor, g: Tensor, f: Tensor, t: Tensor,
                s_out: Tensor, s_in: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        E = he.shape[1]
        ge = g.unsqueeze(1).expand(-1, E, -1)
        he = he + self.edge(torch.cat([he, hv[:, f], hv[:, t], ge], -1))
        gv = g.unsqueeze(1).expand(-1, hv.shape[1], -1)
        hv = hv + self.node(torch.cat([hv, s_out @ he, s_in @ he, gv], -1))
        g = g + self.glob(torch.cat([g, hv.mean(1), he.mean(1)], -1))
        return hv, he, g


class CascadeGNN(nn.Module):
    def __init__(self, f: Tensor, t: Tensor, n_bus: int, node_dim: int, edge_dim: int,
                 hidden: int = 64, layers: int = 6):
        super().__init__()
        E = f.numel()
        s_out = torch.zeros(n_bus, E)
        s_in = torch.zeros(n_bus, E)
        s_out[f, torch.arange(E)] = 1.0
        s_in[t, torch.arange(E)] = 1.0
        self.register_buffer("f", f.long())
        self.register_buffer("t", t.long())
        self.register_buffer("s_out", s_out)
        self.register_buffer("s_in", s_in)
        self.register_buffer("temperature", torch.ones(()))  # set by calibration

        self.enc_v = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.enc_e = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(layers))
        self.head_e = mlp(hidden, 1, hidden)
        self.head_g = mlp(2 * hidden, 1, hidden)
        self.hidden = hidden

    def forward(self, xv: Tensor, xe: Tensor) -> tuple[Tensor, Tensor]:
        """xv (B, N, Fv), xe (B, E, Fe) -> failure logits (B, E), load-shed logit (B,)."""
        hv, he = self.enc_v(xv), self.enc_e(xe)
        g = hv.new_zeros(hv.shape[0], self.hidden)
        for blk in self.blocks:
            hv, he, g = blk(hv, he, g, self.f, self.t, self.s_out, self.s_in)
        edge_logit = self.head_e(he).squeeze(-1)
        shed_logit = self.head_g(torch.cat([g, he.max(1).values], -1)).squeeze(-1)
        return edge_logit, shed_logit


class Exported(nn.Module):
    """Inference wrapper: calibrated probabilities instead of logits."""

    def __init__(self, model: CascadeGNN):
        super().__init__()
        self.model = model

    def forward(self, xv: Tensor, xe: Tensor) -> tuple[Tensor, Tensor]:
        edge_logit, shed_logit = self.model(xv, xe)
        return torch.sigmoid(edge_logit / self.model.temperature), torch.sigmoid(shed_logit)

"""CascadeGNN: an edge-aware message-passing network with a global context vector.

The topology is fixed (out-of-service branches are a feature, not removed edges), so the
graph structure is baked in as constant gather indices and incidence matrices. Every op is a
dense batched tensor op, which keeps inference fast and makes ONNX export trivial.

Per block (pre-LayerNorm, residual):
    edges  : h_e <- h_e + W_o · GELU(W_e h_e + (W_f h_v)[from] + (W_t h_v)[to] + W_g g)
    nodes  : h_v <- h_v + W_o · GELU(W_v h_v + W_s Σ_out h_e + W_r Σ_in h_e + W_g g)
    global : g   <- g   + MLP([g, mean h_v, mean h_e])

This is an MLP over [edge, from-bus, to-bus, global] written so that bus terms are projected
once per bus and then gathered, instead of concatenating wide per-edge tensors. Same function
class, a fraction of the memory traffic (inference is memory-bound on CPU).
Heads: per-branch failure logit, and graph-level expected load-shed fraction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class Block(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.ln_v, self.ln_e, self.ln_e2, self.ln_g = (nn.LayerNorm(h) for _ in range(4))
        self.e_self = nn.Linear(h, h)
        self.e_from = nn.Linear(h, h, bias=False)
        self.e_to = nn.Linear(h, h, bias=False)
        self.e_glob = nn.Linear(h, h, bias=False)
        self.e_out = nn.Linear(h, h)
        self.v_self = nn.Linear(h, h)
        self.v_out = nn.Linear(h, h, bias=False)
        self.v_in = nn.Linear(h, h, bias=False)
        self.v_glob = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h)
        self.glob = mlp(3 * h, h, h)

    def forward(self, hv: Tensor, he: Tensor, g: Tensor, f: Tensor, t: Tensor,
                s_out: Tensor, s_in: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        nv, ng = self.ln_v(hv), self.ln_g(g)
        src = self.e_from(nv) + self.e_glob(ng).unsqueeze(1)  # fold global in at bus level
        x = self.e_self(self.ln_e(he)) + src[:, f] + self.e_to(nv)[:, t]
        he = he + self.e_out(F.gelu(x))
        ne = self.ln_e2(he)
        y = (self.v_self(nv) + self.v_out(s_out @ ne) + self.v_in(s_in @ ne)
             + self.v_glob(ng).unsqueeze(1))
        hv = hv + self.v_proj(F.gelu(y))
        g = g + self.glob(torch.cat([g, hv.mean(1), he.mean(1)], -1))
        return hv, he, g


class CascadeGNN(nn.Module):
    def __init__(self, f: Tensor, t: Tensor, n_bus: int, node_dim: int, edge_dim: int,
                 hidden: int = 48, layers: int = 6):
        super().__init__()
        self.register_buffer("f", f.long())
        self.register_buffer("t", t.long())
        # Bus-branch incidence (from / to ends): aggregation is one batched matmul each.
        E = f.numel()
        s_out = torch.zeros(n_bus, E)
        s_in = torch.zeros(n_bus, E)
        s_out[f, torch.arange(E)] = 1.0
        s_in[t, torch.arange(E)] = 1.0
        self.register_buffer("s_out", s_out)
        self.register_buffer("s_in", s_in)

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
    """Inference wrapper: Platt-calibrated probabilities instead of logits."""

    def __init__(self, model: CascadeGNN, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.model = model
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, xv: Tensor, xe: Tensor) -> tuple[Tensor, Tensor]:
        edge_logit, shed_logit = self.model(xv, xe)
        return torch.sigmoid(self.scale * edge_logit + self.bias), torch.sigmoid(shed_logit)

"""Four ways for a tiny model to learn while it is being prompted.

Every prompt is a SET of example tokens (one token per (x, y) pair) followed by query
tokens (x plus a "what is y?" marker). There are no position embeddings: a pair is one
token, so order carries no information and longer prompts need no new positions.

    kind="attn"  transformer. Arm A (frozen in-context), arm B (A + naive test-time SGD),
                 arm C (same net, meta-trained so that test-time SGD helps).
    kind="mem"   fast-weight memory instead of attention (arm D): each layer keeps a
                 matrix memory S per head and writes every example into it with the delta
                 rule  S <- (1 - alpha) S + beta (v - S k) k^T  (Schlag et al. 2021; the
                 TTT-Linear inner step of Sun et al. 2024; alpha = learned forgetting and
                 beta = learned write strength, a light version of the Titans gate).

Test-time training (arms B and C) updates only a small "fast" adapter: a low-rank
residual  h += (h A) B  in the last two blocks, with B starting at zero, so with no
update the model is exactly the frozen model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks import NX, P


@dataclass
class Config:
    kind: str = "attn"   # attn | mem
    d: int = 256
    layers: int = 6
    heads: int = 4
    ff: int = 1024
    rank: int = 8        # fast adapter rank
    n_fast: int = 2      # blocks with a fast adapter (the last n_fast)


SIZES = {
    "0.1M": dict(d=64, layers=2, heads=4, ff=256),
    "1M": dict(d=128, layers=5, heads=4, ff=512),
    "5M": dict(d=256, layers=6, heads=4, ff=1024),
}


class Block(nn.Module):
    def __init__(self, c: Config, fast: bool):
        super().__init__()
        self.c, self.fast = c, fast
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.w1, self.w2 = nn.Linear(c.d, c.ff), nn.Linear(c.ff, c.d)
        if c.kind == "mem":
            self.gate = nn.Linear(c.d, 2 * c.heads)  # beta (write) and alpha (forget)
            nn.init.zeros_(self.gate.weight)
            with torch.no_grad():
                self.gate.bias[: c.heads] = 0.0    # beta ~ 0.5
                self.gate.bias[c.heads:] = -4.0    # alpha ~ 0.02
        if fast:
            self.A = nn.Parameter(torch.randn(c.d, c.rank) / c.d ** 0.5)

    def heads(self, t):  # [B, T, d] -> [B, H, T, dh]
        B, T, _ = t.shape
        return t.view(B, T, self.c.heads, -1).transpose(1, 2)

    def mix_attn(self, h, mask):
        q, k, v = self.qkv(self.ln1(h)).chunk(3, -1)
        q, k, v = self.heads(q), self.heads(k), self.heads(v)
        # explicit softmax attention (not SDPA): arm C differentiates through the test-time
        # update, which needs a second derivative that fused attention kernels don't provide
        a = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
        a = a.masked_fill(~mask[:, None], float("-inf"))
        out = torch.softmax(a.float(), -1).to(v.dtype) @ v
        return self.o(out.transpose(1, 2).reshape(h.shape))

    def mix_mem(self, h, k_pairs, pvalid, prefix):
        """Delta-rule fast-weight memory. Pairs are written in order; a pair reads the
        memory before its own write. In prefix mode query i reads the state after pairs
        0..i-1; otherwise all queries read the final state."""
        c = self.c
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = self.heads(q), self.heads(k), self.heads(v)  # [B, H, T, dh]
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)
        g = torch.sigmoid(self.gate(x)).transpose(1, 2)  # [B, 2H, T]
        beta, alpha = g[:, : c.heads], g[:, c.heads:]
        B, H, T, dh = q.shape
        S = h.new_zeros(B, H, dh, dh)
        outs = torch.zeros_like(q)
        valid = pvalid.to(h.dtype)[:, None]  # [B, 1, k]
        for t in range(k_pairs):
            outs[:, :, t] = torch.einsum("bhij,bhj->bhi", S, q[:, :, t])
            if prefix:
                outs[:, :, k_pairs + t] = torch.einsum("bhij,bhj->bhi", S, q[:, :, k_pairs + t])
            kt, vt = k[:, :, t], v[:, :, t]
            err = vt - torch.einsum("bhij,bhj->bhi", S, kt)
            bt = (beta[:, :, t] * valid[:, :, t])[..., None, None]
            at = (alpha[:, :, t] * valid[:, :, t])[..., None, None]
            S = (1 - at) * S + bt * torch.einsum("bhi,bhj->bhij", err, kt)
        if not prefix:
            outs[:, :, k_pairs:] = torch.einsum("bhij,bhtj->bhti", S, q[:, :, k_pairs:])
        return self.o(outs.transpose(1, 2).reshape(h.shape))

    def forward(self, h, mask, k_pairs, pvalid, prefix, fastB=None):
        if self.c.kind == "attn":
            h = h + self.mix_attn(h, mask)
        else:
            h = h + self.mix_mem(h, k_pairs, pvalid, prefix)
        z = self.ln2(h)
        h = h + self.w2(F.gelu(self.w1(z)))
        if self.fast and fastB is not None:  # per-prompt adapter, fastB: [B, rank, d]
            h = h + torch.einsum("btr,brd->btd", z @ self.A, fastB)
        return h


class TinyLearner(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ex = nn.Embedding(NX, c.d)
        self.ey = nn.Embedding(P, c.d)
        self.eq = nn.Parameter(torch.randn(c.d) * 0.02)
        self.blocks = nn.ModuleList(Block(c, fast=i >= c.layers - c.n_fast) for i in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.out = nn.Linear(c.d, P)
        # meta-learned inner-loop step size per fast block (used by arm C; log scale)
        self.log_lr = nn.Parameter(torch.full((c.n_fast,), -1.0))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fast_init(self, batch: int, device=None):
        return [torch.zeros(batch, self.c.rank, self.c.d, device=device) for _ in range(self.c.n_fast)]

    @staticmethod
    def build_mask(pvalid, Q, prefix, drop=None):
        """Boolean attention mask (True = may attend) for [pairs, queries].
        pairs: causal over valid pairs. query i: valid pairs j < i (prefix) or all valid
        pairs; never other queries. Every row may attend to itself."""
        B, k = pvalid.shape
        dev = pvalid.device
        T = k + Q
        m = torch.zeros(B, T, T, dtype=torch.bool, device=dev)
        causal = torch.tril(torch.ones(k, k, dtype=torch.bool, device=dev))
        m[:, :k, :k] = causal[None] & pvalid[:, None, :]
        if prefix:
            assert Q == k
            strict = torch.tril(torch.ones(k, k, dtype=torch.bool, device=dev), -1)
            m[:, k:, :k] = strict[None] & pvalid[:, None, :]
        else:
            m[:, k:, :k] = pvalid[:, None, :].expand(B, Q, k)
        idx = torch.arange(T, device=dev)
        m[:, idx, idx] = True
        return m

    def forward(self, px, py, pvalid, qx, prefix=False, fast=None):
        """Returns logits [B, Q, P] for the queries."""
        k, Q = px.shape[1], qx.shape[1]
        pairs = self.ex(px) + self.ey(py)
        queries = self.ex(qx) + self.eq
        h = torch.cat([pairs, queries], 1)
        mask = self.build_mask(pvalid, Q, prefix) if self.c.kind == "attn" else None
        fi = 0
        for blk in self.blocks:
            fb = None
            if blk.fast and fast is not None:
                fb = fast[fi]
            if blk.fast:
                fi += 1
            h = blk(h, mask, k, pvalid, prefix, fb)
        return self.out(self.ln(h[:, k:]))


def config_dict(c: Config) -> dict:
    return asdict(c)

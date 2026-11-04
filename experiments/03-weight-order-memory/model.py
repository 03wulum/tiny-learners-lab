"""Tiny decoder-only transformer whose 'adaptive block' can be swapped per problem.

The adaptive block is the MLP input matrix (fc_in.weight, shape [4d, d]) of chosen layers.
`forward(idx, block=...)` accepts, per adaptive layer, either a shared [out, in] matrix
or one matrix per sequence [B, out, in], so many problems adapt in parallel.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc_in, self.fc_out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, w_in=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        hdn = self.ln2(x)
        if w_in is None:
            hdn = self.fc_in(hdn)
        elif w_in.dim() == 2:
            hdn = F.linear(hdn, w_in, self.fc_in.bias)
        else:
            hdn = torch.einsum("btd,bod->bto", hdn, w_in) + self.fc_in.bias
        return x + self.fc_out(F.gelu(hdn))


class TinyLM(nn.Module):
    def __init__(self, vocab, max_len, d=320, n_layer=6, n_head=5, adaptive=(4, 5)):
        super().__init__()
        self.cfg = dict(vocab=vocab, max_len=max_len, d=d, n_layer=n_layer, n_head=n_head,
                        adaptive=tuple(adaptive))
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.adaptive = tuple(adaptive)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def block_weights(self):
        """The adaptive block: list of fc_in.weight tensors (the originals, shared)."""
        return [self.blocks[i].fc_in.weight for i in self.adaptive]

    def forward(self, idx, block=None, return_hidden=False):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for i, b in enumerate(self.blocks):
            w = None
            if block is not None and i in self.adaptive:
                w = block[self.adaptive.index(i)]
            x = b(x, w)
        hdn = self.ln_f(x)
        logits = self.head(hdn)
        return (logits, hdn) if return_hidden else logits


def answer_loss(logits, idx, mask, reduce_per_seq=False):
    """Next-token CE on positions whose *target* token is marked in mask."""
    lg, tgt, m = logits[:, :-1], idx[:, 1:], mask[:, 1:].float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(m)
    per = (ce * m).sum(1) / m.sum(1).clamp_min(1)
    return per if reduce_per_seq else per.mean()


@torch.no_grad()
def exact_match(logits, idx, mask):
    pred = logits[:, :-1].argmax(-1)
    ok = (pred == idx[:, 1:]) | (mask[:, 1:] == 0)
    return ok.all(1).float()


class Controller(nn.Module):
    """P2: reads the support prompt (base model's pooled hidden state) and writes a
    low-rank 'shadow' direction per adaptive layer. The shadow is snapped to a pure
    reordering of the original weights (see adapt.project_rows)."""

    def __init__(self, d, out_dim, in_dim, n_layers, n_basis=64, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_layers * n_basis))
        self.U = nn.Parameter(torch.randn(n_layers, n_basis, out_dim) / math.sqrt(out_dim))
        self.V = nn.Parameter(torch.randn(n_layers, n_basis, in_dim) / math.sqrt(in_dim))
        self.scale = nn.Parameter(torch.full((n_layers,), 0.02))
        self.n_layers, self.n_basis = n_layers, n_basis
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled):
        g = self.mlp(pooled).view(-1, self.n_layers, self.n_basis)          # (B, L, nb)
        return [self.scale[l] * torch.einsum("bn,no,ni->boi", g[:, l], self.U[l], self.V[l])
                for l in range(self.n_layers)]


def n_params(m):
    return sum(p.numel() for p in m.parameters())

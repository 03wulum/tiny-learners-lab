"""Training prompts generated directly on the GPU (same distributions as tasks.py), so the
training loop does almost no CPU work and keeps nothing large in system RAM."""
from __future__ import annotations

import torch

from tasks import NX, P, TRAIN_RULES


class GpuTasks:
    def __init__(self, device, seed=0):
        self.dev = device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.rules = torch.as_tensor(TRAIN_RULES, device=device)

    def randint(self, high, shape):
        return torch.randint(high, shape, device=self.dev, generator=self.g)

    def rules_batch(self, n, k, q):
        ab = self.rules[self.randint(len(self.rules), (n,))]
        a, b = ab[:, :1], ab[:, 1:]
        px, qx = self.randint(P, (n, k)), self.randint(P, (n, q))
        return px, (a * px + b) % P, qx, (a * qx + b) % P

    def facts_batch(self, n, k, q, shown_frac=0.0):
        table = self.randint(P, (n, NX))
        px, qx = self.randint(NX, (n, k)), self.randint(NX, (n, q))
        if shown_frac > 0:  # most queries ask keys that were shown
            pick = torch.rand((n, q), device=self.dev, generator=self.g) < shown_frac
            qx = torch.where(pick, px.gather(1, self.randint(k, (n, q))), qx)
        return px, table.gather(1, px), qx, table.gather(1, qx)

    def mixed(self, n, k, q, prefix):
        """Half rule prompts, half fact prompts. prefix=True: query i is x_i (training format)."""
        h = n // 2
        r = self.rules_batch(h, k, q)
        f = self.facts_batch(n - h, k, q, shown_frac=0.0 if prefix else 0.75)
        px, py, qx, qy = (torch.cat([u, v], 0) for u, v in zip(r, f))
        if prefix:
            qx, qy = px, py
        pv = torch.ones_like(px, dtype=torch.bool)
        return px, py, pv, qx, qy

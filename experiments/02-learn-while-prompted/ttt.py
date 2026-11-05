"""Test-time training on the prompt's own examples (arms B and C).

Leave-one-out: for each shown pair j, hide it and ask the model for y_j from the other
pairs. The loss on those k guesses is the training signal, so the model trains only on
data inside the prompt. Only the per-prompt fast adapter (models.py) is updated; the
shared weights never change, so prompts don't interfere with each other.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def loo_loss(model, px, py, pvalid, fast):
    """Mean leave-one-out loss per prompt, summed over prompts (so each prompt's gradient
    only reaches its own fast weights)."""
    B, k = px.shape
    eye = torch.eye(k, dtype=torch.bool, device=px.device)
    rep = lambda t: t.repeat_interleave(k, 0)
    pv = rep(pvalid) & ~eye.repeat(B, 1)                 # [B*k, k]: pair j dropped in copy j
    qx = px.reshape(B * k, 1)
    qy = py.reshape(B * k)
    f = [rep(t) for t in fast]
    logits = model(rep(px), rep(py), pv, qx, prefix=False, fast=f)[:, 0]
    nll = F.cross_entropy(logits.float(), qy, reduction="none").view(B, k)
    w = pvalid.float()
    return ((nll * w).sum(1) / w.sum(1).clamp(min=1)).sum()


def adapt(model, px, py, pvalid, steps: int, lr=None, create_graph=False):
    """Returns fast weights after `steps` SGD steps on the leave-one-out loss.
    lr=None uses the model's meta-learned per-block step sizes (arm C)."""
    B = px.shape[0]
    fast = [t.requires_grad_(True) for t in model.fast_init(B, px.device)]
    if steps == 0 or px.shape[1] < 2:
        return fast
    for _ in range(steps):
        loss = loo_loss(model, px, py, pvalid, fast)
        grads = torch.autograd.grad(loss, fast, create_graph=create_graph)
        if lr is None:
            lrs = model.log_lr.exp()
            fast = [f - lrs[i] * g for i, (f, g) in enumerate(zip(fast, grads))]
        else:
            fast = [f - lr * g for f, g in zip(fast, grads)]
        if not create_graph:
            fast = [f.detach().requires_grad_(True) for f in fast]
    return fast

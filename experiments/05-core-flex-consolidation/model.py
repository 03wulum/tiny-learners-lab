"""Stacked tiny MLPs: many independent copies ("runs") trained in lockstep in ONE process.

One network reads [cue ; x] and answers y. Every tensor has a leading run dimension R, so
3 seeds x every arm share each GPU kernel (Fortran rule: one process, data on the GPU).

The same network plays two roles:
  * the CORE of the core+flex model (width 384), with a "slot port": a slot adds a code
    vector to layer 1 (where the input lands) and a rank-4 change U V^T to layer 2.
    That is how the outer, flexible weights inform the inner core.
  * the PLAIN network of the baselines, made wider so its parameter count equals the core
    plus all 18 active slots.
"""
import math, time
import torch
import torch.nn.functional as F

import skills

CORE_H, RANK = 384, 4
N_FLEX, N_CORE = 6, 12
IN = skills.CUE_DIM + skills.P


def slot_numel(h=CORE_H, rank=RANK):
    return h + 2 * h * rank + skills.CUE_DIM          # code + U + V + key


def mlp_numel(h):
    return (IN * h + h) + (h * h + h) + (h * skills.P + skills.P)


def matched_plain_h(n_slots=N_FLEX + N_CORE):
    """Width of the plain net whose parameter count matches core + n_slots slots."""
    target = mlp_numel(CORE_H) + n_slots * slot_numel()
    return max(h for h in range(16, 4096) if mlp_numel(h) <= target)


def init_stack(R, h, dev, gen, n_codes=0):
    def u(*shape, fan_in):
        b = 1 / math.sqrt(fan_in)
        return (torch.rand(*shape, generator=gen, device="cpu") * 2 - 1).mul_(b).to(dev)
    P = dict(W1=u(R, h, IN, fan_in=IN), b1=u(R, h, fan_in=IN), W2=u(R, h, h, fan_in=h), b2=u(R, h, fan_in=h),
             W3=u(R, skills.P, h, fan_in=h), b3=u(R, skills.P, fan_in=h))
    if n_codes:
        P["codes"] = (0.5 * torch.randn(R, n_codes, h, generator=gen)).to(dev)
    return P


def forward(P, cue, x, code=None, U=None, V=None):
    """cue (R, N, 64), x (R, N, X) ints -> logits (R, N, X, 17).
    The N cues each answer X inputs. code (R, N, h), U and V (R, N, h, r): per-cue slot."""
    W1c, W1x = P["W1"][:, :, :skills.CUE_DIM], P["W1"][:, :, skills.CUE_DIM:]
    R = cue.shape[0]
    h1 = torch.einsum("rnc,rhc->rnh", cue, W1c) + P["b1"][:, None]
    if code is not None:
        h1 = h1 + code
    xe = W1x.transpose(1, 2)[torch.arange(R, device=x.device)[:, None, None], x]        # (R, N, X, h)
    h1 = F.gelu(h1[:, :, None] + xe)
    h2 = torch.einsum("rnxh,rgh->rnxg", h1, P["W2"]) + P["b2"][:, None, None]
    if U is not None:
        h2 = h2 + torch.einsum("rnxk,rngk->rnxg", torch.einsum("rnxh,rnhk->rnxk", h1, V), U)
    return torch.einsum("rnxh,roh->rnxo", F.gelu(h2), P["W3"]) + P["b3"][:, None, None]


def ce(logits, y):
    """Per-element cross-entropy; logits (..., 17), y (...)."""
    return F.cross_entropy(logits.reshape(-1, skills.P), y.reshape(-1), reduction="none").view(y.shape)


class StackAdam:
    """Adam over stacked tensors with one learning rate per run (lr: (R,) tensor)."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.b, self.eps, self.t = params, lr, betas, eps, 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.b
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None:
                continue
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            lr = self.lr.view(-1, *([1] * (p.dim() - 1)))
            p.sub_(lr * (m / c1) / ((v / c2).sqrt() + self.eps))
            p.grad = None


def pretrain(stacks, rules, cues, seconds, dev, gen, batch=512, max_steps=None, ram_guard=None):
    """Teach the 208 known rules to every stacked net at once, in one time budget.
    stacks: list of (P, use_codes). cues: (R, 208, 64) on dev, one cue set per run.
    Cue path: noisy cue -> rule. Code path (core only): zero cue + learned code per rule,
    so the core learns what a slot code means."""
    tab = torch.as_tensor(skills.table(rules), device=dev)
    n = len(rules)
    params = [p for P, _ in stacks for p in P.values()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.01)
    t0, step, log = time.time(), 0, []
    while True:
        el = time.time() - t0
        if (max_steps is not None and step >= max_steps) or (max_steps is None and el >= seconds):
            break
        if ram_guard and step % 200 == 0:
            ram_guard()
        frac = el / seconds if max_steps is None else step / max_steps
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1.0, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, frac))))
        loss = 0.0
        for P, use_codes in stacks:
            R = P["W1"].shape[0]
            idx = torch.randint(n, (R, batch), device=dev)
            x = torch.randint(skills.P, (R, batch, 1), device=dev)
            c = cues[torch.arange(R, device=dev)[:, None], idx]
            c = F.normalize(c + skills.CUE_NOISE * torch.randn_like(c), dim=-1)
            y = tab[idx, x[..., 0]][..., None]
            l = ce(forward(P, c, x), y).mean() * R
            if use_codes:
                code = P["codes"][torch.arange(R, device=dev)[:, None], idx]
                code = code + 0.1 * torch.randn_like(code)          # keeps the code space smooth
                lc = ce(forward(P, torch.zeros_like(c), x, code=code), y).mean() * R
                l = 0.5 * l + 0.5 * lc
            loss = loss + l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log.append(dict(step=step, sec=round(el, 1), loss=round(loss.item(), 4)))
        step += 1
    for p in params:
        p.requires_grad_(False)
    return dict(steps=step, sec=round(time.time() - t0, 1), log=log)


@torch.no_grad()
def recall(P, cues, tables, code=None, U=None, V=None, chunk=64):
    """Accuracy (0-100) per (run, cue) answering all 17 inputs. cues (R, N, 64) on dev,
    tables (R, N, 17) numpy or tensor. Slot tensors optional, shaped as in forward."""
    R, N = cues.shape[:2]
    dev = cues.device
    x = torch.arange(skills.P, device=dev).expand(R, 1, skills.P)
    out = []
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        n = cues[:, sl].shape[1]
        kw = {} if code is None else dict(code=code[:, sl], U=U[:, sl], V=V[:, sl])
        out.append(forward(P, cues[:, sl], x.expand(R, n, skills.P), **kw).argmax(-1))
    pred = torch.cat(out, 1)
    tab = torch.as_tensor(tables, device=dev)
    return pred, 100.0 * (pred == tab).float().mean(-1)

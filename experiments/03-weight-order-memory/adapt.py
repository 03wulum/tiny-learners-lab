"""Test-time adaptation arms. All arms change only the adaptive block.

Every function works on a batch of B independent problems at once: the block is a list
(one entry per adaptive layer) of per-problem matrices [B, out, in].

Arms
  A   frozen               no change (in-context learning only)
  B   fine-tune            Adam steps on the support examples
  B'  bit-matched FT       B's change, squeezed to the bits a row reordering can carry
  P1  sort-reorder         B's result used only as a ranking: each row of the ORIGINAL
                           values is re-ordered to follow it (closest pure reordering, L2)
  P2  learned reorder      controller reads the prompt, proposes a shadow, snapped the same way
  M   mask ("pause")       zero the weights whose removal helps most
  R   random reorder       control: same fraction of positions moved as P1, at random
"""
import math
import numpy as np
import torch

from model import answer_loss, exact_match


def to_t(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def expand(W, B):
    return [w.detach().unsqueeze(0).expand(B, *w.shape).clone() for w in W]


# ---------------------------------------------------------------- bit accounting
def log2_factorial(n):
    return math.lgamma(n + 1) / math.log(2)


def perm_bits_per_weight(row_len):
    return log2_factorial(row_len) / row_len


def mask_bits(n, frac):
    k = max(1, int(round(n * frac)))
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


# ---------------------------------------------------------------- projections
def project_rows(W_orig, shadow):
    """Pure reordering of each row of W_orig whose ranks follow `shadow` (same shape).
    This is the L2-closest within-row permutation of the original values."""
    sv = torch.sort(W_orig, dim=-1).values
    order = torch.argsort(shadow, dim=-1)
    return torch.empty_like(shadow).scatter_(-1, order, sv.expand_as(shadow).contiguous())


def project_matrix(W_orig, shadow):
    """Same, but values may move anywhere in the matrix."""
    shp = shadow.shape
    flat = project_rows(W_orig.reshape(*W_orig.shape[:-2], -1), shadow.reshape(*shp[:-2], -1))
    return flat.reshape(shp)


def project_neuron_wiring(W_orig, shadow):
    """Whole rows (neurons' input wiring) move as units; outgoing weights do NOT follow,
    so this breaks the symmetry. Closest assignment via Hungarian matching (CPU)."""
    from scipy.optimize import linear_sum_assignment
    out = torch.empty_like(shadow)
    Wc = W_orig.detach().double().cpu()
    for b in range(shadow.shape[0]):
        S = shadow[b].detach().double().cpu()
        cost = (S * S).sum(1)[:, None] + (Wc * Wc).sum(1)[None, :] - 2 * S @ Wc.T
        r, c = linear_sum_assignment(cost.numpy())
        out[b, torch.as_tensor(r)] = W_orig[torch.as_tensor(c)].to(out.dtype)
    return out


PROJECT = {"row": project_rows, "matrix": project_matrix, "neuron": project_neuron_wiring}


def random_reorder(W_orig, frac, B, generator=None):
    """Control: in each row, a random `frac` of positions swap values at random."""
    out = []
    for w in W_orig:
        w = w.detach().unsqueeze(0).expand(B, *w.shape)
        L = w.shape[-1]
        u = torch.rand(w.shape, device=w.device, generator=generator)
        moved = u < frac
        pos = torch.arange(L, device=w.device, dtype=torch.float32).expand(w.shape)
        big = 2.0 * L
        key_src = torch.where(moved, torch.rand(w.shape, device=w.device, generator=generator) * L, big + pos)
        key_dst = torch.where(moved, pos, big + pos)
        src = torch.argsort(key_src, -1)
        dst = torch.argsort(key_dst, -1)
        out.append(torch.empty_like(w).scatter_(-1, dst, torch.gather(w, -1, src)))
    return out


def moved_fraction(W_orig, new):
    return float(np.mean([(n != w.unsqueeze(0)).float().mean().item() for w, n in zip(W_orig, new)]))


def quantize_delta(delta, bits):
    """Uniform per-row quantization of a weight change to 2**bits levels."""
    levels = max(2, int(2 ** bits))
    lo, hi = delta.amin(-1, keepdim=True), delta.amax(-1, keepdim=True)
    step = (hi - lo).clamp_min(1e-12) / (levels - 1)
    return torch.round((delta - lo) / step) * step + lo


# ---------------------------------------------------------------- core adaptation
def finetune(model, start, tb, lr, steps, use_full=False):
    """Adam on the support answers. start: list of [B,o,i]. Returns new list (detached)."""
    ws = [w.detach().clone().requires_grad_(True) for w in start]
    opt = torch.optim.Adam(ws, lr=lr)
    idx, m = (tb["full"], tb["ymask"]) if use_full else (tb["supp"], tb["smask"])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        answer_loss(model(idx, block=ws), idx, m, reduce_per_seq=True).sum().backward()
        opt.step()
    return [w.detach() for w in ws]


def apply_row_subset(W_orig, new, row_subset):
    """Keep rows outside the subset at their original values (block-size ablation)."""
    if row_subset is None:
        return new
    out = []
    for l, (w, n) in enumerate(zip(W_orig, new)):
        keep = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
        if row_subset[l] is not None:
            keep[row_subset[l]] = True
        out.append(torch.where(keep[None, :, None], n, w.unsqueeze(0).expand_as(n)))
    return out


def adapt(model, arm, tb, hp, W_orig, current=None, controller=None, granularity="row",
          row_subset=None):
    """Return the adapted block for a batch of problems.
    current: block to start from (cumulative streams); defaults to the original weights."""
    B = tb["supp"].shape[0]
    cur = current if current is not None else expand(W_orig, B)
    if arm == "A":
        return cur
    if arm in ("B", "Bq", "P1", "M", "oracle"):
        steps = hp.get("oracle_steps", 100) if arm == "oracle" else hp["steps"]
        lr = hp["lr"]["B" if arm == "oracle" else arm]
        shadow = finetune(model, cur, tb, lr, steps)
        if arm in ("B", "oracle"):
            new = shadow
        elif arm == "Bq":
            bits = perm_bits_per_weight(W_orig[0].shape[-1])
            new = [c + quantize_delta(s - c, bits) for c, s in zip(cur, shadow)]
        elif arm == "P1":
            proj = PROJECT[granularity]
            new = [proj(w, s) for w, s in zip(W_orig, shadow)]
        else:  # M: pause weights whose shrinking helped most
            new = []
            for c, s in zip(cur, shadow):
                score = (c * (c - s)).flatten(1)
                k = max(1, int(hp["mask_frac"] * score.shape[1]))
                thr = score.topk(k, dim=1).values[:, -1:]
                new.append(torch.where((score >= thr).view_as(c), torch.zeros_like(c), c))
        return apply_row_subset(W_orig, new, row_subset)
    if arm == "P2":
        with torch.no_grad():
            _, hdn = model(tb["supp"], block=cur, return_hidden=True)
            deltas = controller(hdn.mean(1))
        return [project_rows(w, c + d) for w, c, d in zip(W_orig, cur, deltas)]
    if arm == "R":
        return random_reorder(W_orig, hp["rand_frac"], B)
    raise ValueError(arm)


@torch.no_grad()
def query_acc(model, tb, block):
    return exact_match(model(tb["full"], block=block), tb["full"], tb["ymask"])

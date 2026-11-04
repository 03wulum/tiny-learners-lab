"""The slot memory of the core + flexible weights model: routing, rings, hardening, archive.

This file is bookkeeping only (which slot learns, which is replaced). The learning itself
is batched across runs in train.py.

Each slot is a key on the cue sphere plus a code and a rank-4 change that plug into the
frozen core. Two rings:
  flex ring (n_flex): a new skill lands here at once and learns at the full learning rate
  core ring (n_core): a skill moves here ("hardens") after harden_at visits; it then
                      learns 10x slower, so it stays stable
Routing is a nearest-key lookup (cosine >= tau): a pointer, the same work per query however
much is stored. Nothing grows: when a ring is full the weakest slot (strength fades 1% per
episode, +1 per use) moves to the archive (or is dropped if archive=False), and it comes
back the moment its cue shows up again.

flat='lru' / 'strength' merges the rings into one ring of n_flex + n_core slots that evicts
the least recently used / the weakest slot (ablations). n_flex=None means unlimited.
"""
import numpy as np
import torch

import skills
from model import RANK, slot_numel


class Slot:
    __slots__ = ("key", "code", "U", "V", "strength", "last_t", "uses", "tier", "skill")

    def __init__(self, key, code, U, V, skill):
        self.key, self.code, self.U, self.V, self.skill = key, code, U, V, skill
        self.strength, self.last_t, self.uses, self.tier = 0.0, 0, 0, "flex"

    def eff(self, t, decay):
        return self.strength * decay ** (t - self.last_t)


class SlotMemory:
    def __init__(self, h, mean_code, dev, gen, n_flex=6, n_core=12, harden_at=3, archive=True, flat=None,
                 decay=0.99, harden_lr=0.1, tau=0.7):
        self.h, self.mean_code, self.dev, self.gen = h, mean_code, dev, gen
        self.n_flex, self.n_core, self.harden_at = n_flex, n_core, harden_at
        self.archive_on, self.flat, self.decay, self.harden_lr, self.tau = archive, flat, decay, harden_lr, tau
        if flat is not None and n_flex is not None:
            self.n_flex, self.n_core = n_flex + n_core, 0
        self.active, self.archive = [], []
        self.route_errors = 0

    # ---- routing -------------------------------------------------------------------------
    def _best(self, which, cue):
        slots = self.active if which == "active" else self.archive
        if not slots:
            return None, -1.0
        cache = self.__dict__.setdefault("_keys", {})
        if cache.get(which) is None:                      # key matrix, rebuilt only after a change
            cache[which] = np.stack([s.key for s in slots])
        sims = cache[which] @ cue
        i = int(sims.argmax())
        return slots[i], float(sims[i])

    def _changed(self):
        self.__dict__["_keys"] = {}

    def lookup(self, cue, use_archive=True):
        s, sim = self._best("active", cue)
        if sim >= self.tau:
            return s, "active"
        if use_archive and self.archive_on:
            s, sim = self._best("archive", cue)
            if sim >= self.tau:
                return s, "archive"
        return None, "none"

    # ---- one episode: pick the slot that learns, then book-keep -----------------------------
    def prepare(self, t, cue, skill):
        """Return (slot to train, its learning-rate multiplier, where it was found)."""
        s, where = self.lookup(cue)
        if s is not None and s.skill != skill:
            self.route_errors += 1
        self._changed()
        if where == "archive":
            self.archive.remove(s)
            self._admit(s, t)                                       # restore
        elif s is None:
            V0 = torch.randn(self.h, RANK, generator=self.gen) / self.h ** 0.5
            s = Slot(cue.copy(), self.mean_code.clone(), torch.zeros(self.h, RANK, device=self.dev),
                     V0.to(self.dev), skill)
            self._admit(s, t)
        return s, (self.harden_lr if s.tier == "core" else 1.0), where

    def finish(self, t, s, cue):
        self._changed()
        s.strength = s.eff(t, self.decay) + 1.0
        s.last_t, s.uses = t, s.uses + 1
        if s.uses > 1:
            s.key = skills.unit(s.key * (s.uses - 1) + cue).astype(np.float32)
        if self.flat is None and s.tier == "flex" and s.uses >= self.harden_at and self.n_core:
            if self._ring_full("core"):
                self._evict(t, "core")
            s.tier = "core"                                         # harden

    def _evict(self, t, tier):
        pool = [s for s in self.active if s.tier == tier]
        if self.flat == "lru":
            victim = min(pool, key=lambda s: s.last_t)
        else:
            victim = min(pool, key=lambda s: s.eff(t, self.decay))
        self.active.remove(victim)
        if self.archive_on:
            self.archive.append(victim)                             # never lost: stored, restorable

    def _ring_full(self, tier):
        cap = self.n_flex if tier == "flex" else self.n_core
        return cap is not None and sum(s.tier == tier for s in self.active) >= cap

    def _admit(self, s, t):
        s.tier = "flex"
        if self._ring_full("flex"):
            self._evict(t, "flex")
        self.active.append(s)

    # ---- answering ------------------------------------------------------------------------
    def slot_tensors(self, cues, use_archive=True):
        """Per-cue slot weights (zeros where no slot matches) and the cue to feed the core
        (zeroed where a slot answers: the slot path replaces the cue path)."""
        n = len(cues)
        code = torch.zeros(n, self.h, device=self.dev)
        U = torch.zeros(n, self.h, RANK, device=self.dev)
        V = torch.zeros(n, self.h, RANK, device=self.dev)
        keep = np.ones(n, np.float32)
        for i in range(n):
            s, _ = self.lookup(cues[i], use_archive)
            if s is not None:
                code[i], U[i], V[i], keep[i] = s.code, s.U, s.V, 0.0
        return code, U, V, keep

    def bytes(self, which):
        return len(self.active if which == "active" else self.archive) * slot_numel(self.h) * 2   # fp16

    def state(self):
        pack = lambda s: dict(key=torch.as_tensor(s.key).half(), code=s.code.half().cpu(), U=s.U.half().cpu(),
                              V=s.V.half().cpu(), strength=s.strength, uses=s.uses, tier=s.tier)
        return dict(active=[pack(s) for s in self.active], archive=[pack(s) for s in self.archive])

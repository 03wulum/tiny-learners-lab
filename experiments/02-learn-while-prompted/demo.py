"""Watch a trained arm learn a rule it has never seen, one example at a time.

    python demo.py                      # arm C, a random held-out rule
    python demo.py --arm D --facts      # fact recall with the fast-weight memory
    python demo.py --teach "3:7 5:13 1:2"   # your own examples (x:y, x 0-16, y 0-16)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from arms import EVAL_ARMS, load
from run import predict
from tasks import HELDOUT_RULES, NX, P, apply_rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C", choices=list(EVAL_ARMS))
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--teach", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck, ttt = EVAL_ARMS[a.arm]
    model, _ = load(ck)
    rng = np.random.default_rng(a.seed)
    t = lambda x, dt=torch.long: torch.as_tensor(np.array(x), dtype=dt)

    if a.teach:
        pairs = [tuple(int(v) for v in p.split(":")) for p in a.teach.split()]
        px, py = [p[0] for p in pairs], [p[1] for p in pairs]
        qx = list(range(P))
        pred = predict(model, ttt, 1e-2, t([px]), t([py]), t([[True] * len(px)], torch.bool), t([qx]))[0]
        print(f"taught {len(px)} examples; the model now maps:")
        print("  " + "  ".join(f"{x}->{int(y)}" for x, y in zip(qx, pred)))
        return

    if a.facts:
        table = rng.integers(P, size=NX)
        xs = rng.permutation(NX)[:24]
        truth = lambda q: table[q]
        print("secret: a random table of 32 facts. Showing the model 1 fact at a time.")
    else:
        ab = HELDOUT_RULES[rng.integers(len(HELDOUT_RULES))]
        xs = rng.integers(P, size=24)
        truth = lambda q: apply_rule("affine", ab[None], np.array([q]))[0]
        print(f"secret rule (never seen in training): y = ({ab[0]}*x + {ab[1]}) mod {P}")
    for k in range(1, len(xs) + 1):
        px = list(xs[:k])
        py = [int(truth(int(x))) for x in px]
        qx = list(xs[:k]) if a.facts else list(range(P))
        pred = predict(model, ttt, 1e-2, t([px]), t([py]), t([[True] * k], torch.bool), t([qx]))[0].numpy()
        acc = float(np.mean([int(p) == int(truth(int(q))) for p, q in zip(pred, qx)]))
        bar = "#" * int(acc * 40)
        what = "facts recalled" if a.facts else "of all 17 inputs correct"
        print(f"after {k:2d} examples: {acc:6.1%} {what}  {bar}")


if __name__ == "__main__":
    main()

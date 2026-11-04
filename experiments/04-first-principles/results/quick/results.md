# Results · experiment 04

**Verdict: INCONCLUSIVE** · ✨ INTRIGUING: from a single example the composer got >= 90% on 8 unseen operations (add3, minus3, sub, half, mul, grandchild, grandmother, cousin)

| Criterion | Value | Threshold | Result |
|---|---|---|---|
| C1 budget | composer_mb 0.00, kept_mb 0.01, train_min 0.00, answer_ms 0.00, search_s 0.00, peak_ram_gb 1.59 | see criteria.py | PASS |
| C2 arithmetic, 5 unseen ops (composer mean, margin over best baseline) | composer 100.00, best_baseline 0.00, margin 100.00 | >= 90.0 and margin >= 30.0 | PASS |
| C3 going the opposite way: a - b | composer 100.00, training_facts 100 | >= 95.0, facts <= 100, 0 subtractions | PASS |
| C4 family, 6 unseen relations (composer mean, margin over best baseline) | composer 95.51, best_baseline 0.00, margin 95.51 | >= 90.0 and margin >= 30.0 | PASS |

Validity gate: G1 arith baseline own tasks FAILED, G2 arith oracle FAILED, G3 arith shuffled ok, G3 arith untrained ok, G3 arith no_undo ok, G3 arith random_op ok, G1 family baseline own tasks FAILED, G2 family oracle FAILED, G3 family shuffled ok, G3 family untrained ok, G3 family no_undo ok, G3 family random_relation ok

## Arithmetic (unseen operations), k = 4, mean of seeds

| op | composer | baseline small | baseline large | oracle baseline | composer, 10% facts missing | program found (seed 0) |
|---|---|---|---|---|---|---|
| add3 | 100.0 | 0.0 | 0.0 | 0.0 | 15.0 | `bind(iter(S),#3)` |
| minus3 | 100.0 | 0.0 | 0.0 | 0.0 | 15.0 | `inv(bind(iter(S),#3))` |
| sub | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | `iter(inv(S))` |
| half | 100.0 | 0.0 | 0.0 | 5.0 | 0.0 | `inv(diag(iter(S)))` |
| mul | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | `acc(iter(S))` |
| **mean** | **100.0** | 0.0 | 0.0 | 1.0 | 6.0 | |

Composer accuracy vs number of examples k: add3 90/100/100/100; minus3 100/100/100/100; sub 95/100/100/100; half 95/100/100/100; mul 100/100/100/100 (k = 1/2/4/8)

Negative controls (composer, should be ~0): shuffled 0.0, untrained 0.0, no_undo 0.0, random_op 0.0
Baselines on their own training tasks: small 1.4, large 1.4
Library: [415] distinct programs per seed, built in ≤ 0.18 s; slowest pick of a program for a new operation 0.1 ms; learned block fit 100.0%.

## Family (unseen relations), k = 4, mean of seeds

| relation | composer | baseline small | baseline large | oracle baseline | composer, 10% facts missing | program found (seed 0) |
|---|---|---|---|---|---|---|
| grandchild | 100.0 | 0.0 | 0.0 | 0.0 | 23.2 | `inv((P ; P))` |
| grandmother | 100.0 | 0.0 | 0.0 | 0.0 | 86.3 | `female((P ; P))` |
| uncle | 100.0 | 0.0 | 0.0 | 0.0 | 46.1 | `male((P ; nonself((P ; inv(P)))))` |
| aunt | 73.1 | 0.0 | 0.0 | 0.0 | 0.0 | `female((P ; nonself((P ; inv(P)))))` |
| nephew | 100.0 | 0.0 | 0.0 | 0.0 | 40.5 | `male(inv((P ; nonself((P ; inv(P))))))` |
| cousin | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | `(P ; inv((P ; nonself((P ; inv(P))))))` |
| **mean** | **95.5** | 0.0 | 0.0 | 0.0 | 32.7 | |

Composer accuracy vs number of examples k: grandchild 100/100/100/100; grandmother 100/100/100/100; uncle 53/100/100/100; aunt 50/74/73/100; nephew 29/100/100/100; cousin 100/100/100/100 (k = 1/2/4/8)

Negative controls (composer, should be ~0): shuffled 0.0, untrained 0.0, no_undo 0.0, random_relation 0.0
Baselines on their own training tasks: small 0.0, large 0.0
Library: [861] distinct programs per seed, built in ≤ 0.42 s; slowest pick of a program for a new operation 0.9 ms; learned block fit 100.0%.

No-snap ablation, family (programs run on soft probabilities): grandchild 100.0, grandmother 100.0, uncle 100.0, aunt 73.1, nephew 100.0, cousin 100.0

Budget: composer files 0.000 MB, kept 0.0 MB, training 0.00 min, answer 0.30 µs median, program pick ≤ 0.9 ms, peak RAM 1.59 GB.

# Explainer: core + flexible weights (tiny-learners 05)

_Readable in about 5 minutes. Verdict: **FAIL**, 4 of 5 criteria. It missed the headline by 0.6 points, and the most useful finding was one we didn't plan for._

## 1. What we tested and why

Your idea: a model with **core weights** that stay stable and a fixed amount of **flexible weights** that pick up whatever the current prompt needs. What keeps coming back **hardens** into reserved space in the core. What gets pushed out is **archived, never lost**, and the model never grows. We tested whether that design remembers more than the usual ways of updating a small model, at the same size, without being slower to learn.

## 2. How it works, with an analogy

Think of a workshop. The **core** is the trained craftsperson: their skills don't change from job to job. Around them is a **workbench** with 6 spots (the flexible ring). A new job's tools go on the bench straight away. A tool that is used for 3 jobs gets a hook on the **wall** (the core ring, 12 hooks). Wall tools are adjusted only gently ("hardened"). When the bench or the wall is full, the least-used tool goes to the **storeroom** (the archive). It is still there, and it comes back out the moment that job shows up again. The workshop never gets bigger.

In the model:

- **Core:** a small network (186K parameters) trained once on 208 known rules, then frozen.
- **Slot:** a *key* (where the skill sits on the cue sphere), a *code* that is added to the core's first layer, and a small rank-4 change to its second layer. So the outer ring talks to the inner network: the slot tells the core which rule to run. This is our version of your "inner sphere and outer sphere".
- **Pointer lookup:** to answer, the model compares the incoming cue with the slot keys and picks the closest one (cosine ≥ 0.7). That is one small matrix multiply, so it costs the same however much the model knows.
- **Strength:** +1 each time a skill is used, fading by 1% per episode. The weakest slot is the one that gets replaced.

## 3. The process, step by step (and the choices you might disagree with)

1. **Skills.** A skill is a cue bound to a rule `y = (a·x + b) mod 17`. The cue is a random point on a 64-dimensional sphere, and each time it arrives slightly blurred, the way a request is never phrased the same way twice. *Choice:* the skills are synthetic, so we know the right answer and can measure exactly what was kept.
2. **Pretraining (GPU, 60 s).** The core learns 208 rules. It also learns, through practice codes, what a slot code means. The baselines' plain network learns the same 208 rules and is made wider, so it has exactly as many parameters as the core plus all 18 slots (249K).
3. **The stream.** 224 episodes per seed, each showing 6 worked examples. 8 frequent skills come back 12 times, 24 occasional ones 4 times, and 32 one-offs arrive in 4 bursts of 8, which try to flush the important skills out. Every learner gets the same 20 gradient steps per episode.
4. **Opponents.** A frozen network. Plain fine-tuning, which is known to forget. EWC, which pulls important weights back toward their old values. Replay, which re-studies stored old examples, using as many bytes as our archive could ever reach. *Choice:* we gave replay the most generous fair budget, so beating it is meaningful.
5. **Ablations:** remove the archive, use one ring instead of two (with least-recently-used or strength eviction), harden every skill at once, unlimited slots, and half or double the slots.
6. **Controls that must fail:** shuffled cues, skills never shown, the frozen network on new skills, and shuffled labels. All must stay near chance (5.9%), or the verdict is INCONCLUSIVE.

## 4. What we found

The whole run took 1.9 minutes on Fortran's RTX 4080 SUPER, in one process, peaking at 1.6 GB of RAM.

| | Remembered at the end | Pretrained skills | New skills |
|---|---|---|---|
| Frozen | 52.5 | 100 | 4.9 (chance) |
| Fine-tune | 50.1 | 73.9 | 26.3 |
| EWC | 27.6 | 26.9 | 28.3 |
| Replay (same bytes) | 65.2 | 87.8 | 42.7 |
| **Core + flex (yours)** | **79.5** | **100** | **59.1** |

- **It remembered the most:** 79.5 vs 50.1 for fine-tuning and 65.2 for replay. The pre-registered bar was +30 over fine-tuning. It reached +29.4, which is a real miss: the 3 seeds agree within 0.15 points. So the verdict is FAIL. Against replay, the strongest baseline, it won by 14 points with less storage (340 KB of archive vs 400 KB of replay).
- **It learned as fast as fine-tuning** (35.4 vs 37.3 after a skill's first 6 examples), and it never damaged the core: pretrained skills stayed at 100%.
- **Archived skills came back perfectly:** 194 returns at 100% of their last recall, with no new examples. Without the archive, recall on return was 16%.
- **The surprise: the archive is what remembers.** Count only the weights that are loaded, and the model holds 23% of the new skills, less than replay's 43%. Take the archive away and the score falls from 79.5 to 61.6. A plain one-ring cache with an archive scored the same as your two-ring design (79.8), and so did 9, 36 or unlimited slots.
- **Hardening works, but the two rings aren't needed.** After each burst of one-offs, your model still had 77% of its frequent skills loaded, against 47% for a cache that drops the least recently used skill. A single ring that drops the *weakest* skill (strength = how often it's used, fading over time) kept 80%. The strength score does the protecting, not the second ring.
- **It memorised, it didn't find the rule.** On inputs a skill hadn't shown yet, every learner was at or below chance. So a skill seen once caps at 6 of 17 answers (35%), and only repetition fills in the rest. The core's code space did not let a slot jump to a new rule from 6 examples.

Caveats: EWC's tuning grid started at a learning rate of 1e-3 while fine-tuning's went down to 3e-4, so EWC is probably under-tuned. Skills are synthetic. Lookup was 3.3 µs per query on Fortran's CPU, and one model's update took 43 ms per episode on the GPU.

**Next steps this suggests:** (1) let hardened slots slowly teach the core ("consolidation" proper), so the active weights hold more and the archive less; (2) give slots structure (e.g. separate codes for a and b) so one example batch can find the rule, not just the pairs; (3) try it on a real on-device task, such as app intents a phone learns from you.

## 5. Why it matters

- **On-device assistants** (phones, laptops) can't retrain a big model every time you teach them something. A fixed-size model that learns per request and keeps what matters is the shape such an assistant needs.
- **No silent forgetting:** the archive makes "forgetting" a choice about what stays loaded, not lost information. That is easier to audit, and you can undo it.
- **Predictable cost:** memory and per-query work stay flat, which matters for battery and latency budgets.

## 6. Further reading

- McClelland, McNaughton & O'Reilly (1995), *Why there are complementary learning systems in the hippocampus and neocortex*: the brain's fast and slow learning split. Your design is a version of it.
- Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks* (EWC): https://arxiv.org/abs/1612.00796
- Ba, Hinton et al. (2016), *Using fast weights to attend to the recent past*: https://arxiv.org/abs/1610.06258
- Arani, Sarfraz & Zonooz (2022), *Learning fast, learning slow* (CLS-ER, a replay method with fast and slow weights): https://arxiv.org/abs/2201.12604
- Pritzel et al. (2017), *Neural episodic control* (a fixed-size memory with key lookup and least-recently-used replacement): https://arxiv.org/abs/1703.01988

## 7. Questions someone might ask you

- **Isn't the archive just a bigger model?** It's storage outside the running model, and we count its bytes. Replay gets the same number of bytes for raw examples, so the comparison is fair.
- **Why freeze the core?** Stability. Every new skill lives in slots, so learning one can't damage another. The cost is that the core never gets smarter. Letting hardened skills slowly teach the core is the natural next step.
- **How fast is the pointer lookup? Microseconds or nanoseconds?** We measure it (C1). The lookup itself is about 18 dot products of 64 numbers, which is tens of nanoseconds in compiled code. From Python, overhead puts it in the low microseconds, and a single GPU call costs about 5-10 µs just to launch. So the fastest place for a single lookup is the CPU, and the GPU wins only on big batches.

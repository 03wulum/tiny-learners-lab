# Explainer: core + flexible weights (tiny-learners 05)

_Readable in about 5 minutes. Results are filled in after the GPU run._

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
2. **Pretraining (GPU, about 75 s).** The core learns 208 rules. It also learns, through practice codes, what a slot code means. The baselines' plain network learns the same 208 rules and is made wider, so it has exactly as many parameters as the core plus all 18 slots (249K).
3. **The stream.** 224 episodes per seed, each showing 6 worked examples. 8 frequent skills come back 12 times, 24 occasional ones 4 times, and 32 one-offs arrive in 4 bursts of 8, which try to flush the important skills out. Every learner gets the same 20 gradient steps per episode.
4. **Opponents.** A frozen network. Plain fine-tuning, which is known to forget. EWC, which pulls important weights back toward their old values. Replay, which re-studies stored old examples, using as many bytes as our archive could ever reach. *Choice:* we gave replay the most generous fair budget, so beating it is meaningful.
5. **Ablations:** remove the archive, use one ring instead of two (with least-recently-used or strength eviction), harden every skill at once, unlimited slots, and half or double the slots.
6. **Controls that must fail:** shuffled cues, skills never shown, the frozen network on new skills, and shuffled labels. All must stay near chance (5.9%), or the verdict is INCONCLUSIVE.

## 4. What we found

_Pending the GPU run._

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

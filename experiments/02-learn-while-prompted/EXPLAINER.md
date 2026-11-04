# Explainer: can a tiny model learn while you're talking to it?

## The idea

Most AI models stop learning when training ends. After that, everything they "know" is frozen in their weights. We want a model of 20 MB or less (about the size of a few phone photos) that can still pick up something **new** from what you show it in the prompt, and get better the more you show it.

Think of teaching a friend a card game by playing a few hands. You don't hand them a rulebook. They watch some examples, figure out the rule, and then play hands they've never seen. That's what we test.

## How the test works

1. **A secret rule is made up for every prompt.** For example, "multiply by 3, add 5, and wrap around at 17". A quarter of all possible rules are never used in training, so the model can't have memorized them.
2. **The model sees a few examples** of the rule (like 4 → 0, 2 → 11) and is then asked about inputs it wasn't shown.
3. **We count how often it's right** after 0, 1, 2, ... 64 examples. A real learner gets better as examples pile up.
4. **Facts too.** In a second task, each prompt carries a random table of facts ("key 7 is 12"). The model must recall facts it was told, and it must not "know" facts it wasn't told. If it did, that would mean the answer leaked somewhere.

## Four ways to learn from a prompt

| Arm | Everyday analogy |
|---|---|
| A: in-context learning | Reading the examples and reasoning on the fly, without changing your mind permanently |
| B: plain test-time training | Cramming: drilling on the examples before answering, with no practice at cramming |
| C: meta-learned test-time training | Cramming, but you practiced cramming thousands of times during training, so you know how to do it well |
| D: fast-weight memory | Taking notes in a notebook as you read each example, then checking the notebook to answer |

In an early 4-minute test, plain cramming (B) made the model **worse**. The big question is whether practicing how to learn (C) or keeping a notebook (D) fixes that.

## The headline test

Training only ever showed the model up to 16 examples. At test time we give it up to **64**. If it keeps getting better past 16, it's genuinely learning from new data, not just repeating a pattern it practiced. That's criterion C4, and it decides PASS or FAIL.

## Why it matters

- **Personal AI on your phone**: an app that learns your shortcuts, names or habits on the device, with nothing sent to the cloud.
- **Fast adaptation**: a robot or a game character that adjusts to a new situation from a handful of examples.
- **Cheap**: training takes about 13 minutes on one gaming GPU, and the models fit in 31 MB together.

## Controls (how we know it isn't cheating)

- **Ideal learner**: a brute-force program that checks every possible rule. It shows the best score anyone could get.
- **Shuffled labels**: examples with random answers, so there's nothing to learn. Every model must stay at chance (1 in 17).
- **Untrained model**: must stay at chance too.

## Further reading

- [What can transformers learn in-context?](https://arxiv.org/abs/2208.01066) (Garg et al., 2022): tiny transformers learning whole families of functions from examples.
- [Learning to (learn at test time)](https://arxiv.org/abs/2407.04620) (Sun et al., 2024): layers whose memory is a tiny model trained as it reads.
- [Titans: learning to memorize at test time](https://arxiv.org/abs/2501.00663) (Behrouz et al., 2025): a memory that decides what's surprising enough to store.
- [The surprising effectiveness of test-time training for abstract reasoning](https://arxiv.org/abs/2411.07279) (Akyürek et al., 2024): cramming on puzzle examples works when done carefully.
- [MAML](https://arxiv.org/abs/1703.03400) (Finn et al., 2017): the classic "learning to learn" method that arm C uses.

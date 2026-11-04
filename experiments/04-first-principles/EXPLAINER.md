# Experiment 04, explained

## The question

Obi's picture: if all of human knowledge is a tree, it can be broken down into a few first principles. A model trained compactly on those principles should be able to assemble them, like blocks, into answers about the rest. If it knows numbers and that adding is "one more, again and again", it should be able to work out what going the opposite way means.

This experiment tests the smallest honest version of that idea, in two tiny worlds.

## What the model learns, and what it is given

**Learned (the first principles):**

- *Arithmetic:* 100 facts of the form "after 7 comes 8". Nothing else. No addition, no subtraction.
- *Family trees:* who is whose parent, and each person's gender. No grandparents, uncles or cousins.

Each fact type is learned by a very small neural network (about 7,000 numbers). After training, its answers are frozen into a lookup table: a dictionary where every question points straight to its answer. That is the "pointer" part of Obi's idea: a step costs one lookup, about a microsecond.

**Given (the ways to combine):** five moves that work on any table:

| Move | Plain meaning | Example |
|---|---|---|
| undo | go the opposite way | "next" undone is "previous" |
| repeat | do it again, b times | "next", 5 times, is "+5" |
| chain | do one thing, then another | parent's parent is grandparent |
| same input twice | feed one number in both slots | a + a is doubling |
| fix an input | hold one slot at a small number | a + 3 |

## How it answers something new

It gets 4 worked examples of an operation it has never seen, for example 9 − 4 = 5, 50 − 8 = 42 … It then searches all combinations of its blocks and moves, smallest first, and keeps the smallest program that gets all 4 examples right. For subtraction that is "undo next, b times". It then answers new questions with that program. If nothing fits, it says it doesn't know rather than guessing.

## Why the comparison is fair, and what would make it unfair

The baselines get far more training data and the same 4 examples. The arithmetic baseline is the kind of network Lake and Baroni (Nature, 2023) showed can learn to compose like people: it is trained on millions of few-shot puzzles from 14 operations, including some "backwards" ones (previous, minus 2, minus 5, minus 7). The family baseline reads the family graph directly and is trained on 300 families.

The honest weak spot: the five moves are handed to the composer, not learned. So the experiment does not show that a network discovers composition. It shows how far a few learned facts plus a few fixed moves go, and it checks the composer fails where it should (a random operation with no structure, scrambled examples, or the "undo" move removed).

## Results

_Filled in after the GPU run._

## Where this could be useful

- On-device assistants that must handle requests nobody trained them for, from a tiny model and a few examples.
- Rule discovery from small data: spreadsheets ("this column is the other column minus tax"), app automations, data cleaning.
- A fast "router" for a bigger model: the composer finds the right connections in microseconds and hands them to the model that writes the answer, which is Obi's part-3 idea.

## Further reading

- Lake and Baroni, *Generalization without systematicity* (SCAN), 2018: networks fail to reuse known parts in new combinations. https://arxiv.org/abs/1711.00350
- Lake and Baroni, *Human-like systematic generalization through a meta-learning neural network* (MLC), Nature 2023: the arithmetic baseline here. https://www.nature.com/articles/s41586-023-06668-3
- Ellis et al., *DreamCoder*, 2021: learning a library of building blocks and searching programs. The composer is a tiny DreamCoder with a fixed library. https://arxiv.org/abs/2006.08381
- Andreas et al., *Neural Module Networks*, 2016: assembling small learned modules per question. https://arxiv.org/abs/1511.02799
- Yang et al., *Neural LP*, 2017, and Zhu et al., *Neural Bellman-Ford Networks*, 2021: learning chains of relations in knowledge graphs. https://arxiv.org/abs/1702.08367 · https://arxiv.org/abs/2106.06935
- Dziri et al., *Faith and Fate*, 2023: transformers degrade on long compositions, which the no-snap ablation tests at small scale. https://arxiv.org/abs/2305.18654

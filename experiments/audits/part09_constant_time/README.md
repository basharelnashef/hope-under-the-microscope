# Part 9 constant-time and cache audit

This audit separates four costs that the paper discusses with inconsistent
scope:

1. scalar arithmetic for one already enumerated cached candidate;
2. enumeration of every live candidate for one greedy action;
3. repeated scans and cache updates over a complete compression trajectory;
4. cache storage and dependency invalidation after physical channel surgery.

The exact paper is [HOPE v1 on arXiv](https://arxiv.org/abs/2607.21366v1).
Its versioned PDF fingerprint is recorded in the
[paper manifest](../../../paper/manifest.json).
The relevant anchors are Sections 9 and 10 on PDF pages 18 through 20,
Appendix B.2 through B.4 on pages 38 through 40, and Algorithms 1 and 2 on
pages 68 and 69.

## What is tested

- A width-$m$ layer contains $m$ prune candidates and
  $\binom{m}{2}$ unordered merge candidates.
- A complete one-layer merge trajectory evaluates
  $\sum_{m=2}^{N}\binom{m}{2}=\binom{N+1}{3}$ cached merge scores.
- Pair initialization performs dimension-dependent geometry work, while the
  raw pair cache has quadratic entry count.
- A deterministic 1x1 Conv-BN-ReLU-Conv fixture removes one central channel
  and records changes to predecessor capacities, successor preactivation
  variances, and every successor pair correlation.
- Paper-local refresh remains stale on that fixture. A dependency-aware
  refresh exactly matches a full recomputation and changes the best pair.
- A fixed parameter-count fixture disproves a general uniform scaling between
  static and live action payoffs by reversing the action order.

The operation-count comparison is deterministic. It is not a wall-clock
benchmark and makes no hardware runtime claim. The cache byte count includes
only three float64 scalars per pair (`a`, `b`, and warped correlation); pair
indices and implementation overhead are intentionally excluded.

## Run

Write a fresh run outside the frozen reference package:

```bash
python3 experiments/audits/part09_constant_time/run.py \
  --result-dir artifacts/runs/part_09/manual
```

Authenticate the exact local paper as well:

```bash
python3 experiments/audits/part09_constant_time/run.py \
  --full \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir artifacts/runs/part_09/full
```

Run the focused tests with:

```bash
python3 -m unittest discover \
  -s experiments/audits/part09_constant_time \
  -p 'test_*.py' -v
```

The generator refuses to overwrite a nonempty result directory. An optional
`--manifest-path` hashes only this directory's README, runner, and focused
tests, plus the separately authenticated `paper/manifest.json` input. Every
result row carries an evidence label and claim ID, and the audit exits nonzero
if any declared invariant fails.

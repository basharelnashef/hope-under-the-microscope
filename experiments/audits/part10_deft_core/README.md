# Part 10 DEFT core-state audit

This deterministic NumPy audit makes four specification boundaries in HOPE
v1's Dispersed Elastic Fine-Tuning procedure executable:

1. the paper gives three different core and slack partition rules;
2. destination-gated gradients preserve the narrow slack-to-core routing
   barrier but can change a core neuron's outgoing weights into slack;
3. gradient masking does not freeze BatchNorm running buffers;
4. Algorithm 2 removes a merged child from the active layer while DEFT later
   describes released children as reusable plastic vessels, without specifying
   the retention, recreation, or reentry transition.

The exact paper is [HOPE v1 on arXiv](https://arxiv.org/abs/2607.21366v1).
Its versioned PDF fingerprint is recorded in the
[paper manifest](../../../paper/manifest.json).
The source anchors are Section 11.2.2 and Eqs. 24 through 29 on PDF pages 21
through 23, Appendix G.5 on page 62, Appendix H and Eqs. 102 through 118 on
pages 62 through 67, Algorithm 2 on page 69, and Algorithm 3 lines 1 through 9
and 27 through 30 on page 70.

## Fixed fixtures

The partition fixture uses two layers with capacities `A = [1, 2, 8]` and
`B = [3, 4, 5]`. It computes every recorded Eq. 24 cost as `J = N*c/E_b`:

```text
neuron  N  c  E_b  J
A0      3  1   10  0.3
A1      2  2    8  0.5
B0      3  3    9  1.0
B1      2  4    5  1.6
```

For each removal, `E_b` is checked against the sum of capacities remaining in
that layer. `A2` and `B2` are terminal, so their `N`, `E_b`, and cost are
unrecorded rather than invented.
The global trajectory threshold is `0.75`, Definition H.1 is instantiated with
declared per-layer thresholds `tau_A = 2` and `tau_B = 4`, and Algorithm 3's
global current-capacity threshold is `3.5`. The trajectory rule therefore
makes only a partial assignment: `A0` and `A1` are slack, `B0` and `B1` are
core, and terminal survivors `A2` and `B2` remain unassigned. Pairwise
comparisons involving this rule use only the common four-neuron domain; they
do not count an open terminal state as disagreement with an assigned state.

The routing fixture treats matrix rows as downstream neurons and columns as
upstream neurons. Core is index zero and slack is index one:

```text
source weights = [[2, 3], [5, 7]]
mask           = [[1, 0], [1, 1]]
raw gradient   = [[99, 99], [-2, -2]]
elasticity     = [0, 1]
learning rate  = 0.5
updated weight = [[2, 0], [6, 8]]
source h0      = [1, 1]
target h1      = [1, 4]
```

The slack-to-core entry stays zero. The core-to-slack entry changes from 5 to
6, so the complete outgoing vector belonging to the upstream core operator
changes from `[2, 5]` to `[2, 6]` even though that neuron's incoming and affine
parameters do not. Its operator drift norm is `1`, and its unit-activation
capacity changes from `sqrt(29)` to `sqrt(40)`, a relative increase of
`0.17444043902940712`. Applying the structural mask changes the declared source
core input from `5` to `2`, an initialization shock of `3`; after that
initialization, the destination gate keeps the same core input at `2` for both
`h0` and `h1`, so dynamic drift in the narrow routed core is zero.

The BatchNorm fixture freezes every trainable scalar and performs two
training-mode buffer updates with update rate `0.5`. The batches have
mean/variance pairs `(4, 4)` and `(6, 9)`. Running state moves from `(0, 1)`
to `(4, 5.75)`. On source probes `z = [-1, 0, 1]`, the audit evaluates the
vector BatchNorm output `(z - mu) / sqrt(var + 1e-5)` without ReLU. Its live
source RMSE is `1.7346960857596274`; restored-state and frozen-buffer controls
remain exactly at zero RMSE.

The released-slot fixture distinguishes three objects:

- the child removed or purged from Algorithm 2's active layer, with physical
  storage left unspecified;
- an all-zero retained ReLU slot, whose incoming and outgoing gradients are
  both zero at the declared point;
- a proposed fixed-width repair with `u = gamma = 1`, `beta = v = 0`, and
  `x = y = 1`. It preserves zero initial output. With half squared loss, the
  first outgoing gradient is `dv = -1`; a step of `0.25` sets `v = 0.25`.
  On the second pass, `du = dgamma = dbeta = -0.1875`, so the internal state
  becomes trainable after the outgoing edge opens.

The repair is not a paper-reported method.

## Outputs

The generator writes nine deterministic files:

```text
summary.json
source_map.csv
partition_assignments.csv
partition_disagreements.csv
routing_transitions.csv
bn_buffer_drift.csv
operator_drift.csv
released_slot_gradients.csv
source_state_controls.csv
```

Every CSV row has an evidence label, claim ID, and fail-closed `passed` field.
The summary enforces 26 checks.

## Run

Write a fresh run outside the frozen reference package:

```bash
python3 experiments/audits/part10_deft_core/run.py \
  --result-dir artifacts/runs/part_10/manual
```

Authenticate a local copy of the exact paper as well:

```bash
python3 experiments/audits/part10_deft_core/run.py \
  --full \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir artifacts/runs/part_10/full
```

The generator refuses to overwrite a nonempty result directory. Run the
portable focused suite directly:

```bash
python3 -m unittest discover \
  -s experiments/audits/part10_deft_core \
  -p 'test_*.py' -v
```

An optional `--manifest-path` hashes only this directory's README, runner, and
focused tests, plus the separately authenticated `paper/manifest.json` input.

## Boundary

This is an independent deterministic NumPy audit of DEFT's printed partition,
routing, state, and released-slot semantics. It does not reproduce the paper's
CIFAR-100 to SVHN training protocol, use a dataset or checkpoint, estimate
trained-network prevalence, or claim effects on trained-network final logits,
end-task loss, accuracy, wall-clock time, resource use, or optimizer
convergence. The fixed-width slot and frozen-buffer control are proposed
repairs, while Appendix G.5's source-state restoration is paper-reported.

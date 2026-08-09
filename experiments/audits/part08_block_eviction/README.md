# Part 8 block-eviction audit

This package tests two bounded questions in HOPE v1's block-eviction score:

1. Does Eq. 96 identify the second moment of the tensor that actually enters an
   interior identity shortcut in the post-activation ResNet V1 graph drawn in
   Figure 3, and what can be concluded without measuring that tensor?
2. Can the sum of local sequential-layer bounds in Eqs. 19-20 be interpreted
   as a projection cost determined only by the residual branch function?

For the first question, the topology and the distributional model must be kept
separate. Figure 3 makes the next interior identity input a post-ReLU tensor,
whereas Equation 96 uses

\[
\sqrt{\mathbb E[Y^2]}=\sqrt{\gamma^2+\beta^2},
\]

If the previous pre-activation coordinate is additionally modeled as
\(Y\sim\mathcal N(\beta,\gamma^2)\), the next block's identity input has the
conditional coordinate RMS

\[
\sqrt{\mathbb E[\operatorname{ReLU}(Y)^2]}.
\]

Inside that explicit synthetic model, at zero shift their ratio is exactly
\(\sqrt 2\). The paper-literal eviction cost is therefore \(1/\sqrt 2\) of the
conditional comparator when the numerator and payoff are held fixed. For
negative normalized shifts, the conditional ratio is unbounded; the frozen
finite sweep stops at \(\beta/|\gamma|=-4\). Figure 3 alone does not establish
this Gaussian law or identify its parameters with a visible BatchNorm, so these
numbers are not measured corrections for a trained ResNet.

The answer to the second question is also no in the narrow functional sense.
The audit composes ReLU with itself from one through four times. Every depth
realizes exactly the same scalar residual map, but the Eq. 20-style sum grows
linearly with the number of charged local layers. This does not show that the
proxy cannot be useful. It shows that it is an architecture-dependent heuristic
rather than a residual-function projection distance.

## Evidence boundary

- `paper_claim`: Eq. 20 sums local internal-layer terms, and Eq. 96 uses the
  unrectified Gaussian coordinate moment.
- `verified_fact`: Figure 3 places a terminal ReLU before the next interior V1
  identity block, and the analytic biased ReLU moment agrees with deterministic
  quadrature and seeded Monte Carlo on declared synthetic Gaussian cases.
- `independent_inference`: conditional on the declared Gaussian law, comparing
  Eq. 96 with the post-ReLU boundary gives the reported moment ratios.
- `counterexample`: one conditional equal-payoff fixture reverses order, and the
  ReLU-idempotence family separates residual function from summed local proxy.
- `proposed_repair`: use the actual boundary second moment; Eq. 79 is a closed
  form only where its Gaussian premise has been independently validated.

Projection and stride-changing shortcuts are classified as ineligible for the
coordinate-identity derivation in Eq. 95. No projection-shortcut repair is
proposed. The audit uses no checkpoint or dataset, does not identify a trained
block's post-add distribution, and makes no statement about trained-network
prevalence, logits, loss, accuracy, runtime, or compression quality.

## Run

Write an ordinary diagnostic run outside the frozen reference:

```bash
python3 experiments/audits/part08_block_eviction/run.py \
  --result-dir artifacts/runs/part_08/manual
```

Run the focused tests:

```bash
python3 -m unittest discover \
  -s experiments/audits/part08_block_eviction \
  -p 'test_*.py' -v
```

To write an optional provenance manifest, add for example
`--manifest-path artifacts/runs/part_08/manifest.json`. The manifest hashes
only this directory's README, runner, and focused tests, plus the separately
authenticated `paper/manifest.json` input. A local HOPE PDF is optional for a
normal run and mandatory only with `--full`.

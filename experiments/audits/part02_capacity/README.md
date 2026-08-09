# Part 2 closed-form capacity audit

Evidence labels: `paper_claim`, `verified_fact`, and
`independent_inference`, separated below.

This fail-closed NumPy audit independently evaluates the biased ReLU
self-kernel printed in HOPE Eq. 3 and Eq. 79:

```text
K(i,i) = (gamma^2 + beta^2) Phi(beta / abs(gamma))
         + beta abs(gamma) phi(beta / abs(gamma))
```

It then evaluates the capacity factorization from Eqs. 73-75:

```text
capacity = outgoing_l2 * sqrt(K(i,i)).
```

The paper identity is a `paper_claim`. Agreement with independent fixed-node
quadrature, the zero-shift identity, the deterministic `gamma = 0` case, and
the same-layer Eq. 6 ordering checks are `verified_fact` results of this
project. The simulated ranking comparison is an `independent_inference`; it
is not a paper experiment.

## Source identity and scope

The source is [HOPE v1 on arXiv](https://arxiv.org/abs/2607.21366v1).
Its exact PDF fingerprint and source URL are recorded in the
[paper manifest](../../../paper/manifest.json).

The exact anchors are PDF page 9, Eq. 3; PDF page 13, Eq. 6; PDF page 52,
Eqs. 73-75 and Eqs. 77-78; and PDF page 53, Eq. 79 and Theorem E.2. The audit
uses the negligible-epsilon marginal from Eq. 78. It does not present the
finite-epsilon variance in Eq. 77 as the same formula.

The audit does not evaluate cross-neuron kernels, global
positive-semidefiniteness, the maximum-entropy construction, cross-layer
selection, training, checkpoints, datasets, or any headline experiment from
the paper.

## Numerical methods

`relu_self_kernel(beta, gamma)` accepts broadcastable scalars or arrays. It
uses float64 and handles `gamma == 0` before forming a ratio:

```text
K(i,i) = max(0, beta)^2.
```

It also handles `beta == 0` directly as `gamma^2 / 2`. For a negative ratio,
write `x = -beta / abs(gamma)`. The two positive terms in the printed formula
nearly cancel as `x` grows. For `0 < x < 8`, the implementation factors out
`exp(-x^2 / 2)` and evaluates an `erfcx`-equivalent bracket with the standard
library `erfc`. At `x >= 8`, it evaluates the adaptively truncated expansion

```text
2 phi(x) / x^3 * (1 - 6/x^2 + 45/x^4 - 420/x^6 + ...)
```

in log space. This avoids a division-by-zero path and avoidable intermediate
overflow or underflow. Non-finite inputs, negative results, and true results
outside the float64 range raise instead of being silently accepted.

The primary numerical reference is deterministic Gauss-Legendre quadrature.
The declared default integration grid uses 256 fixed nodes over the active
part of `[-12, 12]`. The declared parameter grid is the Cartesian product of
ratios `[-8, -4, -1, 0, 1, 4, 8]` and signed gamma scales
`[-3, -0.25, 0.25, 3]`.

Seeded Monte Carlo is a secondary check, not a proof. The default is 200,000
samples with seed `20260801`, streamed in chunks of 50,000. The sample count,
seed, and chunk size are configurable and are recorded in the summary and
manifest.

The deterministic ranking fixture compares:

- incoming L1;
- absolute BN scale `abs(gamma)`;
- the zero-shift proxy `abs(gamma) * outgoing_l2`;
- full capacity `outgoing_l2 * sqrt(K(i,i))`.

All ranking inputs are simulated and are labeled that way in every output.
The audit also evaluates Eq. 6 within one fixed layer:

```text
J_prune(c) = N c / (E_a - c),
dJ_prune/dc = N E_a / (E_a - c)^2 > 0,
```

with fixed `N = 128`, fixed pre-action `E_a = 10`, and only
`0 <= c < E_a`. This is not a claim about cross-layer ordering.

## Fail-closed checks

Every check has an inclusive expected range in `manifest.json`. The process
fails for a missing check, a non-finite metric, a malformed range, a detailed
case failure, a config-hash mismatch, or a metric outside its range.

The main numerical tolerances are:

- quadrature: `atol = 5e-14`, `rtol = 5e-11`;
- zero shift: `rtol = 8e-15` with exact zero required at `gamma = 0`;
- complement identity: `atol = 1e-300`, `rtol = 2e-13`;
- near-degenerate limit: `gamma^2 + 2e-13 * abs(limit)`;
- ungated second-moment upper bound: one-sided relative numerical allowance
  `2e-13 * abs(beta^2 + gamma^2)`, with an exact zero bound at the origin;
- Monte Carlo: `5e-14 + 7 * estimated_standard_error`;
- Eq. 6 adjacent secants: `atol = 2e-13`, `rtol = 2e-13`.

Portable semantic revalidation uses the quadrature gate for primary numerical
values, without a global absolute floor; near-zero primary values therefore
remain protected by relative and ULP-scale checks. Because cancellation can
move derived error ratios by many relative units while their absolute values
remain tiny, the configuration also declares comparison-only guards of
`1e-4` for scaled errors, `1e-12` for relative errors, and `5e-13` for
absolute-error diagnostics, with a `1e-9` diagnostic relative term. These
guards compare two already passing reruns; they do not replace or loosen any
case's original pass threshold.

The public `validate_summary(payload)` function reapplies the same registry,
rederives every configured deterministic and seeded section, compares every
detailed value and aggregate under the documented cross-version numerical
allowances, verifies the paper and configuration hashes, and rejects any
non-passing or internally forged summary. It also recomputes row-level errors,
tolerances, pass flags, and section aggregates from the primary values before
the cross-version comparison. The focused test suite binds every CSV field to
the corresponding semantic rerun, so refreshed file hashes cannot conceal
changed ranking or diagnostic values.

## Run and test

Choose any fresh result directory:

```bash
python3 experiments/audits/part02_capacity/run.py \
  --result-dir /tmp/hope-part-02-audit
```

To write the manifest separately, select its path explicitly:

```bash
python3 experiments/audits/part02_capacity/run.py \
  --result-dir /tmp/hope-part-02-audit \
  --manifest-path /tmp/hope-part-02-manifest.json
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/audits/part02_capacity \
  -p 'test_*.py' -v
```

Useful configuration flags are `--seed`, `--mc-samples`,
`--mc-chunk-size`, `--quadrature-order`, `--quadrature-z-limit`, and
`--paper-pdf`.

## Outputs and provenance

The selected result directory receives:

- `metrics.csv`: long-form case metrics, methods, method-origin labels,
  evidence labels, references, errors, tolerances, ranks, and pass flags;
- `summary.json`: configuration, config hash, every detailed audit section,
  aggregate check registry, ranking orders, and Eq. 6 results;
- `manifest.json` by default, or the path selected with `--manifest-path`.

The complete run manifest records the paper checksum, local PDF verification
status, configuration and hash, source hashes, embedded-input hashes, output
paths and hashes, seeds, Python and NumPy versions, platform details, Git
state and commit when available, archive state otherwise, whether the hashed
audit sources match that commit, expected metric ranges, reproduction scope, and
method-origin labels.
By default no machine-local PDF path is embedded. Pass `--paper-pdf` only when
the run itself should verify and record a local copy. An explicitly supplied
path that is missing or is not a regular file fails closed.
The manifest excludes its own hash because a file cannot contain its final
cryptographic hash without self-reference.

A fresh rerun may differ in last-bit numerical values across supported NumPy
versions. Semantic validation requires the same configuration, sources,
inputs, schemas, case identities, evidence labels, and passing tolerance
contracts rather than byte identity across environments.

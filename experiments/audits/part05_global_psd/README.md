# Part 5 deterministic global PSD audit

This audit asks whether HOPE v1's separately constructed two-neuron Gaussian
surrogates necessarily assemble into one positive semidefinite Gram matrix.
It reconstructs the embedded fixed three-neuron witness below and fails closed
on every declared numerical and artifact contract.

The audit is an independent mathematical counterexample. It is not a paper
experiment, checkpoint study, random search, or estimate of how often trained
layers fail a PSD test.

## Fixed witness and formulas

The six-decimal values below are treated as exact inputs:

```text
R = [[ 1.000000, -0.173508, 0.590994],
     [-0.173508,  1.000000, 0.513747],
     [ 0.590994,  0.513747, 1.000000]]

r = [1.106838, 1.793881, 1.634949]
```

For each pair, the audit evaluates paper Eqs. 81 and 80 in that order:

```text
kappa_ij = R_ij / (1 - R_ij^2) * r_i * r_j
Q_ij = 2 * kappa_ij / (1 + sqrt(1 + 4 * kappa_ij^2))
```

It then evaluates the normalized zero-bias Eq. 85 map:

```text
phi(q) = (sqrt(1 - q^2) + (pi - acos(q)) * q) / pi
K_ij = phi(Q_ij)
```

The implementation rejects non-finite values, non-symmetric matrices,
non-unit correlation diagonals, raw pair correlations outside `(-1, 1)`, Eq.
85 inputs outside `[-1, 1]`, and non-positive scale ratios.

The pinned paper anchors are Section 5 and Eqs. 4-5 on physical PDF pages 7-9,
Appendix A Eq. 30 and Proposition A.1 on page 35, Eq. 73 on page 52,
Proposition E.3 and Eqs. 80-81 on page 54, and Eqs. 84-85 on page 56.

## What is proved

Using the exact decimal inputs, the audit verifies:

- `R` is positive definite, with minimum eigenvalue
  `0.12600540389160284`.
- Every isolated `2 x 2` warped-correlation pair and normalized Eq. 85 pair
  is positive definite.
- The assembled warped matrix `Q` is materially indefinite, with minimum
  eigenvalue `-0.25067976835991007`.
- The assembled normalized Eq. 85 matrix `K` is materially indefinite, with
  minimum eigenvalue `-0.04506976334652169`.
- Canonically signed unit eigenvectors give negative quadratic forms matching
  those minimum eigenvalues within the declared tolerance.
- Every pairwise squared distance derived separately from `Q` and `K` remains
  nonnegative. This pairwise fact does not certify global PSD.
- The square-root distances induced by `K` violate the triangle inequality by
  `0.01510131906285106`: the direct distance for pair 1-2 is larger than the
  route through neuron 3.
- Applying Eq. 85 directly to the one jointly valid input `R` gives a
  positive-definite shared-surrogate control, with minimum eigenvalue
  `0.19739822610147048`.

The negative direction for the Eq. 85 matrix is

```text
[-0.4656516215358381, -0.4945479775847531, 0.7338875017520566]
```

and its quadratic form is `-0.04506976334652168`. If the table were one
Hilbert Gram matrix, this value would be a squared norm and could not be
negative.

## Explicit repair comparison

Two independent, non-paper repair diagnostics are compared on both `Q` and
`K`:

1. `plain_eigenvalue_clipping` sets every negative eigenvalue to zero. It is
   PSD but generally changes the diagonal, so it is not a correlation matrix.
2. `psd_renormalized_correlation` performs the same clipping and then applies
   a positive diagonal congruence to restore a unit diagonal. It is PSD and
   has unit diagonal. No correlation-matrix optimality result is claimed.

The audit records minimum eigenvalue, Frobenius perturbation, maximum entry
change, diagonal deviation, and minimum pairwise squared distance for every
source/method combination. The method names above are the only repair labels
used in JSON and CSV artifacts.

## Run into a fresh output directory

From the repository root:

```bash
python3 experiments/audits/part05_global_psd/run.py \
  --result-dir /tmp/hope-part-05-audit
```

The result path is configurable and may be outside the repository. A separate
manifest destination is also supported:

```bash
python3 experiments/audits/part05_global_psd/run.py \
  --result-dir /tmp/hope-part-05-audit \
  --manifest-path /tmp/hope-part-05-manifest.json
```

The normal mode authenticates `paper/manifest.json` and does not require the
machine-local paper PDF. Full source authentication is optional and requires
an explicit pinned v1 PDF:

```bash
python3 experiments/audits/part05_global_psd/run.py \
  --result-dir /tmp/hope-part-05-full \
  --full \
  --paper-pdf /path/to/2607.21366v1.pdf
```

Full mode rejects a missing, symlinked, or checksum-mismatched PDF. Supplying
`--paper-pdf` without `--full` is also rejected so that the recorded mode is
unambiguous.

## Deterministic outputs

The result directory contains six canonical, directory-independent files:

| File | Rows | Role |
|---|---:|---|
| `summary.json` | n/a | Complete fixture, matrices, checks, certificates, distances, triangle diagnostic, repairs, and provenance |
| `witness_matrices.csv` | 72 | All entries of four witness/control matrices and four repaired matrices |
| `spectra.csv` | 24 | Three ordered eigenvalues for each of eight matrices |
| `pair_checks.csv` | 3 | Eqs. 80-81, Eq. 85, local eigenvalues, and squared distances for pairs 1-2, 1-3, and 2-3 |
| `negative_direction.csv` | 6 | Three vector components for each of the two indefinite assemblies |
| `projection_comparison.csv` | 4 | Two repair methods applied to each indefinite source matrix |

`manifest.json` is written beside them by default. It authenticates the six
outputs and records:

- the pinned paper identity and optional local PDF identity;
- exact configuration and configuration hash;
- an explicit no-random-sampling seed policy;
- commit state when available, archive state otherwise, and individually hashed sources;
- input and output hashes and byte sizes;
- Python, NumPy, platform, machine, byte order, and float64 environment;
- method origins and evidence labels;
- the `independent_mathematical_audit` reproduction scope; and
- all 22 expected metric ranges and observed passing checks.

Wall-clock generation time and absolute output paths are omitted so repeated
runs in different fresh directories produce identical bytes on the same
declared environment.

## The 22 fail-closed checks

The exact check registry is:

1. `witness_nonfinite_count`
2. `raw_symmetry_residual`
3. `raw_unit_diagonal_residual`
4. `raw_maximum_pair_absolute_correlation`
5. `minimum_scale_ratio`
6. `raw_minimum_eigenvalue`
7. `local_pair_count`
8. `minimum_local_pair_eigenvalue`
9. `maximum_eq80_equation_residual`
10. `warped_finite_symmetry_residual`
11. `warped_domain_diagonal_violation`
12. `warped_minimum_eigenvalue`
13. `warped_negative_direction_error`
14. `eq85_finite_symmetry_residual`
15. `eq85_domain_diagonal_violation`
16. `eq85_minimum_eigenvalue`
17. `eq85_negative_direction_error`
18. `minimum_unrepaired_pairwise_squared_distance`
19. `shared_control_minimum_eigenvalue`
20. `plain_clipping_maximum_psd_violation`
21. `renormalized_maximum_psd_or_diagonal_violation`
22. `eq85_maximum_triangle_violation`

Changing a check value, declared range, unit, pass flag, deterministic matrix,
certificate, triangle record, repair metric, or summary schema causes
`validate_summary` or `enforce_checks` to fail.

## Tests

Run the standalone suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/audits/part05_global_psd \
  -p 'test_*.py' -v
```

The tests cover formula domains and exact values, finite/symmetric/correlation
validation, local-versus-global spectra, canonical negative directions,
nonnegative pairwise distances, the triangle violation, shared control,
repair semantics, output byte determinism, CSV schemas and row order, source
and output hashes, PDF modes, summary tamper rejection, and ASCII dash safety.

## Claim and evidence registry

The summary and manifest bind these canonical Part 5 claim IDs in order:

1. `P05-COMMON-PX-KERNEL`
2. `P05-COMMON-GRAM-PSD`
3. `P05-PAIR-LOCAL-SURROGATE`
4. `P05-LOCAL-PAIR-VALIDITY`
5. `P05-GLOBAL-GRAM-COUNTEREXAMPLE`
6. `P05-SHARED-SURROGATE-CONTROL`
7. `P05-PSD-PROJECTION-REPAIR`

Paper formulas are labeled `paper_claim`; the fixed global witness is labeled
`counterexample`; the shared control is `verified_fact`; and both projection
branches are `proposed_repair`.

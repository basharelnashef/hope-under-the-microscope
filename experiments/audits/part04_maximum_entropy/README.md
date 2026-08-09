# Part 4 maximum-entropy covariance audit

This audit tests when the pure Gaussian maximum-entropy covariance problem
printed by HOPE v1 is bounded. It also compares that objective with the
identity-precision construction printed in Appendix E.3.1 and with an
explicitly labeled isotropic-reference repair.

The fixtures are deterministic mathematical examples. They are not a trained
checkpoint evaluation, a dataset experiment, or a reproduction of the
paper's compression results.

## Evidence boundary

The source is [HOPE v1 on arXiv](https://arxiv.org/abs/2607.21366v1).
Its exact PDF fingerprint and source URL are recorded in the
[paper manifest](../../../paper/manifest.json).

The exact anchors are Section 4 on physical PDF pages 6-7, Appendix E.3.1 and
Proposition E.3 on page 54, Eqs. 80-81 on page 54, and Eq. 82 on page 55.

The audit keeps four origins separate:

- `paper_claim`: the pure `log det Sigma` program in Section 4;
- `paper_claim`: the unit-identity precision and pair formula in Appendix
  E.3.1;
- `verified_fact` or `counterexample`: independent rank proofs and fixtures;
- `proposed_repair`: minimum KL divergence to `N(0, tau^2 I)` with declared
  `tau`.

The paper-literal implementation is in
`src/hope_repro/paper_literal/maximum_entropy.py`. The tunable reference
objective is isolated in
`src/hope_repro/proposed_repairs/relative_entropy.py`.

## Enforced fixtures

The audit fails unless all of these checks pass:

1. A feasible rank-two constraint set in dimension three keeps both
   projection variances fixed while a nullspace eigenvalue and `log det`
   escape.
2. Two duplicate constraints in dimension two show that constraint count
   equal to dimension does not imply full rank.
3. A full-spanning identity fixture has a finite trace bound and its unique
   fixture maximum at zero off-diagonal covariance.
4. Incompatible duplicate targets are classified as infeasible rather than
   unbounded.
5. Two unit directions separated by 60 degrees, with cosine one half, give
   zero projection correlation under pure local maximum entropy but
   correlation one half under the Appendix unit-identity construction. Both
   meet the same marginal variance constraints, while the pure solution has
   the larger `log det` objective.
6. A compatible collinear pair verifies that the two-direction Gram inverse,
   Eq. 80-81 formula, and pair repair all reject rank one explicitly even
   though both directions are nonzero.
7. A declared `tau` sweep verifies the relative-entropy stationarity equation,
   fixed marginal constraints, and prior-scale dependence of pair
   correlation.
8. Six seeded rank and nullity fixtures exercise both the escape construction
   and the full-spanning bound.

The audit uses float64 throughout, rejects non-finite values, checks positive
definiteness explicitly, and enforces declared residual tolerances. No result
depends on an optimizer declaring a problem unbounded.

## Run

Choose any fresh result directory:

```bash
python3 experiments/audits/part04_maximum_entropy/run.py \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir /tmp/hope-part-04-audit
```

The manifest can be directed separately:

```bash
python3 experiments/audits/part04_maximum_entropy/run.py \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir /tmp/hope-part-04-audit \
  --manifest-path /tmp/hope-part-04-manifest.json
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/audits/part04_maximum_entropy \
  -p 'test_*.py' -v
```

## Outputs

The result directory receives:

- `summary.json`: nested fixtures, checks, method origins, paper identity,
  configuration, and reproduction scope;
- `escape_path.csv`: the two explicit positive-definite nullspace paths;
- `bounded_control.csv`: the full-spanning off-diagonal sweep;
- `pair_comparison.csv`: pure local and Appendix identity-precision results;
- `tau_sweep.csv`: the declared prior-scale sensitivity;
- `rank_cases.csv`: seeded rank and nullity controls.

`manifest.json` is written in the result directory unless
`--manifest-path` is supplied. It authenticates every result, each source
file, the paper manifest, configuration, environment, seed, method origin,
and reproduction scope. The manifest omits wall-clock time and absolute
output paths so equivalent runs in different directories are byte-identical.

# Part 6 axioms audit

This directory contains a deterministic mathematical audit of the layer-capacity
and transition-cost derivation in HOPE v1. It does not reproduce a trained-model
experiment and does not evaluate compression accuracy.

The audit separates six questions:

1. Within the declared `L_p` family, equal partitioning preserves capacity only
   at `p = 1`. This is a conditional result, not a numerical proof that no
   functional outside the assumptions of Lemma C.1 can be partition invariant.
2. With a positive density `c` held constant along one transition, the
   capacity-drain integral is exactly `c log(E_a / E_b)`, independently of the
   chosen monotone parameterization of capacity.
3. Positivity and scale invariance of `c` do not by themselves establish an
   extinction barrier. On the path
   `Phi(E) = (E - E^2, E^2)`, the globally scale-invariant density
   `c(Phi) = Phi_2 / (Phi_1 + Phi_2)` equals `E`, so the integral remains finite
   as `E_b` tends to zero. A uniform positive lower bound on `c` along the
   extinction path is sufficient for logarithmic divergence.
4. Replacing capacity drain with configuration-space speed defines a different,
   path-dependent integral. The paper's endpoint expression is checked as an
   upper bound on the declared straight path only. A second monotone path with
   the same endpoints shows why the same endpoint expression is not a bound for
   arbitrary paths.
5. The family `J_p = N (D / E_b)^p`, `p > 0`, shares magnitude neutrality and,
   when `D` has a positive extinction limit, an extinction barrier. These facts
   do not make `J_p` path additive or a geometric-integral upper bound.
6. A fixed synthetic fixture shows that `p` can change cross-layer ordering when
   widths differ. A same-layer control proves that a fixed width preserves the
   ordering by `D / E_b` for every `p > 0`. No incidence, accuracy, or empirical
   utility claim is made.

The audit deliberately excludes parent-neuron generation, physical merge
execution, and the unknown correlation-threshold issue discussed elsewhere.

## Run

From the repository root:

```bash
python3 experiments/audits/part06_axioms/run.py \
  --result-dir artifacts/runs/part_06/manual
```

The runner resolves every destination before writing, refuses nonempty result
directories, refuses existing or symlink manifests, and rejects a manifest path
that aliases any of the six result files.

To bind a run to a locally available copy of the paper:

```bash
python3 experiments/audits/part06_axioms/run.py \
  --result-dir artifacts/runs/part_06/full \
  --paper-pdf paper/2607.21366v1.pdf \
  --full
```

The result directory contains exactly six audit outputs:

- `summary.json`
- `derivation_steps.csv`
- `partition_sweep.csv`
- `path_integrals.csv`
- `proxy_family.csv`
- `ranking_fixture.csv`

Supplying `--manifest-path` additionally writes the artifact manifest outside
that six-file result set. All fixtures are fixed constants. No random generator,
checkpoint, dataset, optimizer, or network execution is used.

## Evidence boundary

Paper-literal formulas are labeled `paper_claim`. Algebraic checks are labeled
`verified_fact`. The `J_p` family is labeled `counterexample` where it directly
witnesses endpoint-proxy non-uniqueness. Its ranking fixture is a scoped
`verified_fact`. The density path is a `counterexample` to the stronger
claim that positive scale-invariant `c` alone forces connectivity divergence;
it is not a counterexample to the theorem when a uniform positive lower bound
is added.

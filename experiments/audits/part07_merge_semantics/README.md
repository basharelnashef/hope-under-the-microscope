# Part 7 merge-semantics audit

Evidence classes: `paper_claim`, `verified_fact`, `counterexample`,
`independent_inference`, and `proposed_repair`.

This deterministic audit separates three objects that HOPE v1 brings together:

1. Eq. 6 prices the duplicated-coordinate transition
   `[f_i,f_j] -> [f_p,f_p]`.
2. Eqs. 15 through 18 realize one physical parent.
3. Eq. 86 and Algorithm 2 make the executed local transition
   `f_i+f_j -> f_p` after child `j` is purged.

For nonzero exact twins, Eq. 12 selects `f_p=f`. The duplicated-coordinate
distance is zero, while the executed local sum changes from `2f` to `f`. The
audit proves that result exactly and confirms it pointwise on two deterministic
dense-BN-ReLU-dense fixtures, including a nonzero-shift case.

The audit also includes bounded zero-bias diagnostics for input correlation,
output alignment, remaining layer capacity, and near-cancellation. These are
possibility fixtures, not estimates of trained-network frequency or accuracy.

## Repair boundary

Two fixed-direction alternatives are compared and kept outside the
paper-literal method:

- normalized physical-residual scale
  `s=(b E_rem+||f_i+f_j||^2)/(E_rem+b)`;
- unnormalized least-squares scale `s=b`.

Both select `2||f||` for exact twins. Neither is described as a universal
function-preserving merge when the child sum is not realizable in the selected
parent direction.

## Run

Normal runs write under the ignored run namespace and refuse to overwrite a
nonempty result directory:

```bash
python3 experiments/audits/part07_merge_semantics/run.py \
  --result-dir artifacts/runs/part_07/manual
```

Run the unit suite with:

```bash
python3 -m unittest discover \
  -s experiments/audits/part07_merge_semantics \
  -p 'test_*.py' -v
```

To write an optional provenance manifest alongside a disposable run:

```bash
python3 experiments/audits/part07_merge_semantics/run.py \
  --result-dir artifacts/runs/part_07/with-manifest \
  --manifest-path artifacts/runs/part_07/manifest.json
```

The manifest hashes this directory's README, runner, and focused tests. It also
authenticates `paper/manifest.json`; no repository-level verifier or contract
test is a runtime dependency. The optional full mode additionally hashes the
exact paper PDF:

```bash
python3 experiments/audits/part07_merge_semantics/run.py \
  --full \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir artifacts/runs/part_07/full
```

## Outputs

The frozen package contains seven deterministic outputs:

- `summary.json`
- `source_map.csv`
- `twin_cases.csv`
- `correlation_sweep.csv`
- `capacity_sweep.csv`
- `repair_comparison.csv`
- `physical_forward.csv`

The package uses no checkpoint, dataset, random sample, Torch, or SciPy. It
does not reproduce the paper's ResNet-50/ImageNet experiment, enforce the
existential merge-correlation threshold from Lemma C.3, exercise the adapted
compression fallback, or make an end-task performance claim.

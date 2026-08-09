# Part 1 positive-rescaling audit

Evidence label: `verified_fact`

This lightweight NumPy experiment checks the first article's central identity:
for a positive scale, multiplying the effective incoming affine map and
dividing every outgoing consumer by the same scale preserves the local PH-1
function.

It also separates:

- exact post-BN positive rescaling;
- exact raw-filter BatchNorm rescaling with finite-epsilon compensation;
- uncorrected raw-filter rescaling as a negative control;
- the paper's epsilon-free capacity approximation;
- an explicit finite-epsilon extension.

Run:

```bash
python3 experiments/audits/part01_positive_rescaling/run.py \
  --result-dir /tmp/hope-part-01-audit
```

The output directory receives `invariance_metrics.csv` and `summary.json`.

Test:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/audits/part01_positive_rescaling \
  -p 'test_*.py' -v
```

The script is a framework-independent algebraic reference. It is not a live
PyTorch or TensorFlow module test.

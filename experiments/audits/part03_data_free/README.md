# Part 3 data-free premise audit

This is a fail-closed diagnostic audit of what BatchNorm stores and what a
Gaussian surrogate adds. It instruments four points in the adapted VGG-8
architecture on a provenance-bound CIFAR-10 test subset:

| Audit layer | BN module | Role |
|---|---|---|
| `conv0` | `features.1` | early expansion layer |
| `conv2` | `features.9` | middle convolution |
| `conv5` | `features.19` | late convolution |
| `dense` | `bn_b` | bottleneck |

The checkpoint and dataset route is labeled `architecture_adaptation`; this
audit is its diagnostic analysis role.
It is not the paper's ResNet-50/ImageNet compression experiment and does not
reproduce a reported paper result.

## Evidence boundary

The paper source is [HOPE v1 on arXiv](https://arxiv.org/abs/2607.21366v1).
Its exact PDF fingerprint and source URL are recorded in the
[paper manifest](../../../paper/manifest.json).

The relevant anchors are Section 4 on PDF pages 6-7, Eqs. 76-78 and Footnote
11 on PDF page 52, and the exact biased pair moment in Eq. 83 on PDF page 55.

The paper statements are `paper_claim` inputs. Bounded measurements, formula
invariants, and the same-sample common-linear-map reconstruction are
`verified_fact` results of this project. Gaussian-fit errors are
`independent_inference` diagnostics. The shuffle construction is a
`counterexample` to identifying dependence from separate marginals.

This audit supports these claim records:

- `P03-DATA-FREE-SCOPE`
- `P03-BN-MARGINAL-CONSTRAINTS`
- `P03-EQ76-COLUMN-SPACE`
- `P03-BN-NOT-JOINT`
- `P03-GAUSSIAN-SURROGATE-BOUNDARY`
- `P03-CHECKPOINT-GAUSSIAN-DIAGNOSTIC`

One checkpoint does not establish behavior across seeds, architectures, or
training procedures. No accuracy, pruning, compression, fine-tuning, or
width-alone causal conclusion is produced.

## Deterministic bounded sample

The default selects 2,000 CIFAR-10 test examples by a seeded, near-equal
class-stratified sample without replacement. Indices are sorted before the
data loader runs. The data loader uses zero workers, CPU execution, one Torch
thread, and deterministic Torch algorithms.

Preprocessing is exactly `ToTensor`, with no normalization and no
augmentation. The VGG-8 checkpoint is deserialized through Torch's
weights-only loader and then loaded with strict state-dictionary matching;
that fact is recorded.

At each audited layer, at most 16 evenly spaced spatial positions per example
and 16 evenly spaced channels are retained. Eight deterministic channel pairs
are evaluated. Thus every layer has at most
`max_examples * spatial_positions` observations. The selected-index hash and
the hash of the selected raw images and labels are recorded.

The default subset, channel set, spatial positions, and pair set are
configuration, not claims of population representativeness. Their bounds and
selection rules appear in `summary.json` and the run manifest.

Dataset provenance binds the raw CIFAR-10 `test_batch` and `batches.meta`
files by SHA-256 and byte size, in addition to the selected-example hash. The
checkpoint record binds the complete checkpoint plus a deterministic digest
of ordered state keys, shapes, and dtypes. Historical training provenance is
explicitly labeled `incomplete`: the surviving checkpoint does not contain an
epoch history, optimizer state, training-data hashes, or its original command
and environment record.

## Measurements

For selected channels, both raw pre-BN and affine post-BN pre-ReLU values have:

- mean and population variance;
- skewness and excess kurtosis;
- fitted-normal Kolmogorov distance;
- nonpositive fraction and ReLU second moment.

The normality values have no pass threshold. A checkpoint is allowed to agree
or disagree with a Gaussian fit.

For every selected post-BN channel, the empirical ReLU self-kernel is
decomposed exactly as:

```text
empirical - Gaussian(BN state, negligible epsilon)
  = shape error
  + empirical-moment versus BN-state error
  + finite-epsilon error.
```

The decomposition identity is a quality gate. The size or sign of any
component is not.

For selected pairs, the audit uses the empirical means, standard deviations,
and correlation in the exact biased Gaussian ReLU pair kernel from Eq. 83.
It reports empirical-minus-Gaussian error, absolute error, and bounded
symmetric relative error. These fit errors are diagnostics, not pass gates.
The bivariate normal CDF is evaluated by deterministic fixed-node quadrature
of Plackett's correlation integral. Independent and zero-bias fixtures gate
the implementation itself.

## Marginals do not identify dependence

Within each layer, a seeded permutation is applied to one member of every
selected pair. The strongest deterministic witness is retained. The
permutation preserves its mean, variance, higher marginal moments, and ReLU
self-kernel, while its covariance and pair ReLU kernel can change. Marginal
preservation and a nontrivial dependence change are quality gates. This
witness does not claim that the shuffled joint is a realistic network state;
it establishes non-identification from marginals alone.

## Eq. 76 projection condition

For each audited raw weight matrix `W`, the audit measures

```text
||W W+ mu_BN - mu_BN|| / max(||mu_BN||, tiny(float64)).
```

That checkpoint residual is reported without a pass threshold. It tests the
condition used in the Eq. 76 derivation; it is not interpreted as caused by
layer width. Separately, the hooks construct a common mean input patch and
gate the same-sample bias-free identity

```text
mean(W x) = W mean(x).
```

The second identity protects the measurement plumbing and demonstrates why an
expansion layer does not fail merely because it is wide.

## Run

Install the Part 3 extra with a Torch build appropriate for the machine:

```bash
python -m pip install -e '.[part3]'
```

Run with explicit checkpoint, dataset, and paper inputs:

```bash
python3 experiments/audits/part03_data_free/run.py \
  --checkpoint /path/to/vgg8_cifar10.pt \
  --data-dir /path/to/cifar-root \
  --paper-pdf /path/to/2607.21366v1.pdf \
  --result-dir /tmp/hope-part-03-audit
```

The result and manifest destinations can be selected independently:

```bash
python3 experiments/audits/part03_data_free/run.py \
  --checkpoint /path/to/vgg8_cifar10.pt \
  --data-dir /path/to/cifar-root \
  --paper-pdf /path/to/2607.21366v1.pdf \
  --result-dir /tmp/hope-part-03-audit \
  --manifest-path /tmp/hope-part-03-manifest.json
```

Absolute runtime input paths are never serialized. External inputs are
recorded only by logical role, basename, byte size, split/count, and SHA-256.
Do not commit the checkpoint, CIFAR-10 files, caches, or raw activation arrays.

Focused tests do not require Torch:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/audits/part03_data_free \
  -p 'test_*.py' -v
```

The runner authenticates the checkpoint, selected CIFAR-10 data, and paper
PDF before using them. The focused suite checks the deterministic formulas,
schemas, provenance rules, and summary validation without requiring Torch or
the external inputs.

## Outputs

The result directory receives:

- `summary.json`: full nested measurements, selection rules, checks, and input
  identities;
- `metrics.csv`: long-form measurement and evidence-label rows;
- `layer_summary.csv`: compact per-layer normality, self-kernel, shuffle, and
  Eq. 76 summaries;
- `pair_summary.csv`: compact empirical versus exact biased Gaussian pair
  kernels;
- `histograms.csv`: fixed-bin standardized post-BN empirical and
  standard-normal densities for one explicitly selected channel per layer.

The selected manifest records paper identity, configuration hash, seeds,
source hashes, source snapshot state, dependency environment, checkpoint and
selected-data hashes, expected quality-gate ranges, and output hashes. It does
not hash itself because a file cannot contain its final cryptographic hash
without self-reference.

The histogram representative is the middle entry of each deterministic
evenly spaced channel selection. It is not selected for a large or small
Gaussian discrepancy. Forty bins cover standardized values from -5 to 5;
lower and upper tail fractions are recorded separately.

# Paper-literal module

This package is an independent translation of selected equations and
algorithms printed in HOPE v1. It is not author code.

## Included

- effective BatchNorm parameters;
- ReLU self and approximate cross kernels;
- warped pair correlation;
- the Section 4 pure log-determinant covariance objective and rank-domain
  helpers;
- pruning and merging costs;
- printed parent scale and parameter recovery;
- sequential VGG-8 model surgery;
- the Appendix G.2 VGG-8 network structure.

## Important scope limit

The current surgery code supports the local sequential VGG-8 diagnostic. It
does not support Keras ResNet-50, ImageNet evaluation, residual dependency
handling, block eviction, or DEFT. It therefore cannot reproduce either
reported paper experiment.

The model class in `vgg8.py` defines architecture only. Dataset selection,
training, checkpoint selection, and evaluation belong to an experiment and
must carry their own evidence label.

The independent sum-preserving parent-scale repair is implemented separately
in `hope_repro.proposed_repairs.sum_scale`. It is not enabled by the
paper-literal encoder.

The paper-literal maximum-entropy module records the Section 4 objective and
the Appendix E unit-identity precision formula as distinct printed
constructions. The tunable isotropic Gaussian reference-prior formulation
lives in `hope_repro.proposed_repairs.relative_entropy`; it is not silently
substituted for the Section 4 program.

## Install

From the repository root:

```bash
python -m pip install -e '.[paper]'
```

The paper extra installs Torch and SciPy. The core kernel path imports SciPy
for the normal cumulative distribution. The model surgery path requires
Torch. The separate `part3` extra adds torchvision for the data-backed Part 3
diagnostic.

## Data-free import check

```bash
python -c "from hope_repro.paper_literal.core import self_kernel; print(self_kernel(0.0, 1.0))"
python -c "from hope_repro.paper_literal.vgg8 import VGG8; print(type(VGG8()).__name__)"
```

The first command should print a value near `0.5`. These checks validate local
imports only. They do not validate a paper result.

Read [DEVIATIONS.md](DEVIATIONS.md) before using the implementation.

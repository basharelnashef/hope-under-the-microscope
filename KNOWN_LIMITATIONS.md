# Known limitations

*HOPE Under the Microscope* v1.0.0 is an independent audit and proposed
reference design by Bashar. It is not author code, an official implementation,
or an exact reproduction of the headline experiments in the HOPE paper.

## Evidence boundary

- Most results are analytical checks or deterministic local fixtures. They do
  not estimate how often a failure mode appears in trained networks.
- The release does not reproduce the paper's ResNet-50/ImageNet compression
  experiment or its complete CIFAR-100 to SVHN DEFT experiment.
- The Part 3 VGG-8/CIFAR-10 checkpoint diagnostic is an architecture
  adaptation. The checkpoint, dataset, and paper PDF are not redistributed,
  and the historical checkpoint provenance is incomplete.
- Part 11 audits published summaries and a provenance-bound digitization. It
  does not recover unpublished scenario records, identify the paper's
  uncertainty statistic, establish statistical significance, or reproduce a
  training run.

## Scope of the counterexamples

- The Part 4 entropy escape requires a non-spanning constraint set.
- The Part 5 result is a fixed counterexample, not a frequency estimate.
- The Part 7 merge result concerns a local additive contribution and does not
  by itself bound final logits, loss, or accuracy.
- Part 8 verifies the post-ReLU topology only for eligible post-activation
  ResNet V1 identity shortcuts. Its numerical ratios are conditional Gaussian
  fixtures: the release neither measures the actual post-add activation law nor
  proves that Eq. 96's BatchNorm parameters describe it.
- Part 9 reports exact counts and deterministic cache fixtures, not measured
  wall-clock time, memory, latency, or energy.
- Part 10 uses deterministic NumPy fixtures and does not measure transfer
  accuracy or H-score.

## Proposed-design boundary

HOPE-R in Part 12 is a proposed repair, not a method reported by the paper. Its
common geometry uses declared calibration rows and is therefore not strictly
data-free. The included fixtures do not establish optimizer convergence,
trained accuracy, source retention, latency, or universal compression
superiority. Unsupported graph and operator cases fail closed.

Numerical results can vary slightly across NumPy versions, linear-algebra
libraries, and hardware. Tests use explicit tolerances; the published result
files are separately authenticated by checksums.

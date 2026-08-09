# Scope and deviations

The current module is an independent partial translation for a sequential
VGG-8-style network. It is not a full reproduction of the paper.

## Covered

- effective parameters;
- ReLU self and cross kernels;
- pairwise geometry;
- printed parent construction;
- printed parent scale;
- BatchNorm recovery;
- sequential prune and merge operations;
- magnitude baselines used by the local diagnostic.
- the Section 4 covariance objective and Appendix E identity-precision shape,
  with explicit rank and feasibility diagnostics.

## Not covered

- Keras ResNet-50 wiring;
- ImageNet preprocessing and validation;
- residual dependency handling;
- block eviction;
- the complete DEFT procedure;
- all source and target scenarios from Section 11.2;
- a verified reconstruction of the paper's plotted compression curve.

## Paper-literal and repair separation

The independent sum-preserving scale lives in
`hope_repro.proposed_repairs.sum_scale`. The paper-literal parent constructor
always uses the scale printed in Eq. 12, and the default encoder never enables
the repair silently.

The adapted compression command selects the repair-specific encoder only for
the method labeled `extension_sum_scale`. It never presents that branch as
part of the printed method.

The proposed reference-prior covariance implementation lives in
`hope_repro.proposed_repairs.relative_entropy`. Its tunable scale and KL
objective are Part 4 repairs and interpretations, not equations printed in
Section 4. The paper-literal namespace preserves the pure `log det` program
and the Appendix E unit-identity expression separately.

## Degenerate-merge fallback defect

If parent construction returns `None`, the current encoder prunes the weaker
child instead of realizing the inactive-parent case described in Appendix
B.5.1. This behavior is not paper-literal.

The fallback also:

- logs the action as `prune*`;
- retains the rejected merge candidate's cost and distortion rate rather than
  recomputing the executed prune;
- is omitted from the reported merge count;
- is not preserved in a complete public action log.

Until this branch is removed or specified and recorded correctly, an adapted
run cannot establish from its summary whether the fallback fired. A fresh
publishable run must retain the complete action log and separate counts for
printed merges, ordinary prunes, and fallback actions.

## Experiment interpretation choices

The sequential diagnostic uses several choices that the Section 11.1 plot
does not specify:

- PyTorch VGG-8 and CIFAR-10 replace Keras ResNet-50 and ImageNet;
- baseline scores use one static global ranking;
- each sequential layer retains at least two live units;
- model density, evaluation spacing, and stopping density are command
  parameters;
- residual dependency handling and block eviction are absent.

These choices are recorded in each diagnostic JSON output. They must not be
described as author-reported settings.

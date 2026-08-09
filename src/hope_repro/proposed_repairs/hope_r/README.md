# HOPE-R reference design

This package is the framework-independent corrected design developed in Part
12. Every module is `proposed_repair`; none is paper-author code.

- `common.py` folds inference-time BatchNorm and evaluates every neuron under
  one declared weighted empirical calibration measure. Its activation Gram is
  positive semidefinite by construction. This mode is calibrated, not strictly
  data-free.
- `merge.py` fits one physical ReLU parent to the executed child sum. Exact
  twins are anchored; all other pairs receive a measured calibration residual,
  not a universal preservation claim.
- `graph.py` requires complete fan-out, independently enumerates the declared
  cache universe, computes prune/merge invalidation including solitary capacity
  and retained-parent records, applies abstract channel surgery, reports live
  resource counts, and fails closed outside its graph and block domain.
- `deft.py` preserves fixed physical widths and stable slot IDs while making
  release, reinitialization, route masks, optimizer state, BatchNorm buffers,
  and task-conditional evaluation state explicit.

The deterministic audit in
`experiments/audits/part12_corrected_hope/` establishes local fixture
properties only. It does not establish benchmark accuracy, retention,
convergence, latency, trained-network prevalence, or equivalence to an author
implementation.

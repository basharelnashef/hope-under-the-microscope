# Part 12 corrected HOPE audit

This audit exercises the framework-independent HOPE-R reference design. Every
HOPE-R branch is an independently proposed repair. Nothing in this directory is
author code or a result reported by HOPE v1.

## Evidence boundary

The audit uses deterministic float64 fixtures. It tests local algebra, common
Gram geometry, physical sum-map residuals, graph dependency closure, operation
counts, block eligibility, fixed-width transfer state, and the interaction of
those contracts. It does not train a model, access a benchmark dataset,
reproduce either paper experiment, or measure accuracy, loss, convergence,
latency, or trained-network prevalence.

The corrected geometry uses one declared weighted calibration measure for all
neurons in a layer. Its Gram matrix is positive semidefinite by construction,
but this choice consumes calibration inputs and is not strictly data-free. The
paper-literal pair approximation remains a separate comparison branch.

The graph fixture builds a complete cache universe independently of the
invalidation plan, applies an abstract prune or merge to a second graph, and
compares graph-derived pre/post structural cache snapshots. Current-layer
solitary capacities, incoming/outgoing records, BatchNorm assumptions, pair
geometry, successor capacities, and the explicitly retained merge-parent vessel
are all covered. This establishes the declared structural cache contract; it is
not a framework-runtime or trained numerical-capacity equivalence result.

The corrected merge is described as `sum-map optimized`. It measures the
physical residual between the child sum and one realizable parent selected from
a deterministic candidate set. Exact twins are included as anchored candidates
and are preserved. No general pair is promised an exact one-neuron
representation.

## Output contract

The generator writes exactly these files:

- `summary.json`
- `design_contract.csv`
- `common_gram_cases.csv`
- `merge_cases.csv`
- `invalidation_cases.csv`
- `resource_accounting.csv`
- `block_eligibility.csv`
- `deft_transitions.csv`
- `integration_ablations.csv`
- `source_map.csv`

Each CSV row carries a stable claim ID, evidence label, and fail-closed pass
flag. The manifest binds every output, every implementation source, the exact
paper manifest, configuration, environment, method origin, and reproduction
scope. It records the Git source snapshot when Git metadata is available while
retaining independent hashes for every shipped source. Deterministic outputs
omit wall-clock timestamps.

## Run

Write a disposable run outside the frozen reference:

```bash
python3 experiments/audits/part12_corrected_hope/run.py \
  --result-dir artifacts/runs/part_12/manual
```

Authenticate the exact local HOPE v1 paper as well:

```bash
python3 experiments/audits/part12_corrected_hope/run.py \
  --result-dir artifacts/runs/part_12/manual-full \
  --paper-pdf paper/2607.21366v1.pdf \
  --full
```

Run the focused audit tests directly:

```bash
python3 -m unittest discover \
  -s experiments/audits/part12_corrected_hope \
  -p 'test_*.py' -v
```

The generator refuses to overwrite a nonempty result directory or an existing
manifest. Its optional provenance manifest binds this directory's three files,
the public HOPE-R modules under `src/`, and `paper/manifest.json`; it does not
depend on repository-level build or verification machinery.

## Supported reference domain

- dense or ordinary ungrouped convolutional channel graphs;
- affine, BatchNorm, and ReLU functional neurons;
- complete and resolved outgoing-consumer maps;
- same-shape stride-one post-activation ResNet V1 identity shortcuts;
- fixed-width logical transfer slots with explicit state ownership.

Grouped and depthwise convolution, unresolved fan-out, shared weights,
projection shortcuts, stride changes, V2 block boundaries, no-skip blocks,
attention, LayerNorm, GroupNorm, and non-PH-1 activations fail closed or remain
outside the reference contract.

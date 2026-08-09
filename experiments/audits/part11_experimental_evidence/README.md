# Part 11 experimental-evidence audit

This package is a deterministic, fail-closed audit of what the two HOPE v1
experiments do and do not establish. It authenticates arXiv 2607.21366v1,
transcribes Table 2, recomputes Eq. 101, executes small synthetic fixtures, and
records a provenance-bound digitization of the compression raster.

It does not train a model, load a checkpoint, use CIFAR-100, SVHN, or ImageNet,
run a significance test, reconstruct missing per-scenario values, or reproduce
either paper experiment. Proposed experiments have no claimed outcomes.

## Exact Table 2 record and H-score arithmetic

The five displayed central-value triples are:

| Method | Target | Source | Reported H |
|---|---:|---:|---:|
| DEFT | 89.79 | 52.14 | 65.82 |
| Head-Only | 36.11 | 63.13 | 45.79 |
| Full FT | 94.09 | 7.52 | 13.88 |
| EWC | 93.94 | 6.74 | 12.54 |
| PEFT | 81.91 | 5.44 | 10.18 |

Applying Eq. 101 directly to the displayed target and source means gives,
respectively, 65.9713, 45.9416, 13.9269, 12.5776, and 10.2024. None rounds to
the displayed H value. The audit also gives every displayed central value a
conservative closed two-decimal bin of center +/- 0.005. For all five methods,
the harmonic interval implied by the target and source bins is disjoint from
the displayed H bin.

The arithmetic alone does not identify the cause of the discrepancy. The
caption says that all metrics are averaged across tasks and trials, so a mean
of per-scenario H-scores is a plausible explanation. The per-scenario values
and explicit operation order are not supplied, so the printed record cannot
distinguish that explanation from a transcription, reporting, or table error.
The audit therefore records the aggregation identity as an open question.

The exact two-task fixture makes the distinction executable. Both fixtures
have mean target accuracy 50 and mean source accuracy 50:

- Balanced tasks (50, 50) and (50, 50) have mean per-task H equal to 50.
- Crossed tasks (90, 10) and (10, 90) have mean per-task H equal to 18.
- The harmonic mean of the aggregate accuracies is 50 in both fixtures.

Aggregate target and source means therefore do not determine the mean of
per-task harmonic scores.

## Undefined uncertainty

Table 2 prints `+/-` values but does not define their statistic or sampling
unit. The output enumerates several compatible possibilities, including a
standard deviation over 20 scenarios, a standard deviation over four trial
means, standard errors, and a confidence-interval half-width. Every row is
marked possible but unverified.

The paper also reports no inferential test. The audit does not treat `+/-` as
an interval, calculate a p-value, or conclude that any difference is or is not
statistically significant. It records only that a formal significance claim is
unsupported by the undefined fields alone.

## Absolute accuracy, normalized retention, and evaluation state

The Table 2 source column is an absolute post-transfer CIFAR test accuracy.
The paper does not report the matching pre-transfer source accuracy or a
normalization formula. The retention fixture holds DEFT's displayed 52.14
fixed and uses hypothetical pre-transfer accuracies of 60 and 90. Under the
explicit illustrative rule `100 * post / pre`, those denominators yield 86.9
and 57.9333 percent. The hypothetical denominators are not paper results. They
show only why the normalized quantity is not recoverable.

Appendix G.5 evaluates the source side by:

1. grafting the original source head back onto the fine-tuned backbone;
2. zeroing source-head connections from slack origins; and
3. restoring source BatchNorm moving means and variances.

The reported source operand is therefore evaluated under a restored,
source-specific state rather than the unchanged live target-adapted state.
The live-state source accuracy is not separately reported. No direction or
magnitude of the difference between live and restored accuracy is inferred.

## Checkpoint-rule fixture

Appendix G.5 and the Table 2 caption select the final epoch by peak target
validation accuracy. Algorithm 3 selects it by peak validation H-score using
both source and target validation. In the independent two-epoch fixture,
epoch 1 has target/source accuracies 80/80 and H equal to 80, while epoch 2 has
90/10 and H equal to 18. Peak target selects epoch 2; peak H selects epoch 1.

This proves only that the printed selectors are non-equivalent. It does not
identify which checkpoint generated Table 2 or claim that the published method
ranking would change.

## Density and parameter-budget fixture

Section 11.1 plots active-neuron density, while Eq. 21 constrains parameters
released and final parameter footprint. The independent weight-only dense
chain has widths `100 -> 2 -> 4 -> 10` and 248 weights. Pruning either hidden
layer removes one of six active hidden neurons, so both actions reduce neuron
density by 1/6. Pruning a first-hidden-layer neuron releases 104 weights;
pruning a second-hidden-layer neuron releases 12.

Equal neuron density therefore does not fix the Eq. 21 parameter budget. This
fixture is not ResNet-50 and makes no claim that a parameter-matched comparison
would reverse the plotted ranking.

## Compression digitization provenance

The long-form digitization contains 32 rows: four methods at densities 0.3
through 1.0. Every row is labeled `digitized_approximation` and
`author_reported_numeric=false`. It is bound to:

- the exact paper file recorded in the
  [paper manifest](../../../paper/manifest.json);
- physical PDF page 20, image object 1088, and decoded RGB raster size 2929
  by 1724;
- exact paper, RGB-pixel, and soft-mask fingerprints retained in the
  [frozen summary](../../../artifacts/part_11/results/summary.json) and every
  digitization row;
- explicit x and y grid calibration and raw pixel windows.

All rows retain the observed line-width interval. The L1-Joint trace near
density 0.50 and BN-Scale trace near density 0.60 have wide vertical support;
their `accuracy_estimate` is `not_single_valued` and their full y interval is
preserved. A plotting convention must not silently replace either interval by
a midpoint.

## Proposed missing-experiment matrix

The final CSV lists fifteen matched controls for compression and DEFT. It
includes component actions, score attribution, merge semantics, biased-kernel
and empirical-oracle controls, payoff and refresh policies, matched partition
controls, stagewise source damage, live versus restored evaluation, checkpoint
selection, source training recipe, sensitivity, and an extended EWC grid.

Every row is `proposed_not_run`, `paper_reported=false`, and
`outcome_known=false`. The matrix asks what should be measured; it supplies no
answer.

## Outputs and checks

The runner writes eleven deterministic outputs:

1. `summary.json`
2. `source_map.csv`
3. `table2_recalculation.csv`
4. `aggregation_fixture.csv`
5. `uncertainty_interpretations.csv`
6. `retention_fixture.csv`
7. `evaluation_states.csv`
8. `checkpoint_fixture.csv`
9. `density_budget_fixture.csv`
10. `compression_digitization.csv`
11. `missing_experiments.csv`

The summary contains 36 ordered checks. Any non-finite number, failed row,
evidence-class mismatch, changed Table 2 value, rounding-bin overlap, changed
fixture result, unqualified digitization, missing raster identity, or invented
missing-experiment outcome makes the run fail.

Run an offline audit into a new directory:

```bash
python3 experiments/audits/part11_experimental_evidence/run.py \
  --result-dir artifacts/runs/part_11/manual
```

Authenticate an explicit paper copy as well:

```bash
python3 experiments/audits/part11_experimental_evidence/run.py \
  --full \
  --paper-pdf paper/2607.21366v1.pdf \
  --result-dir artifacts/runs/part_11/full
```

The runner refuses to overwrite a nonempty result directory or an existing
manifest.

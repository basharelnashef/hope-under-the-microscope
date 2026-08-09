# HOPE Under the Microscope

An independent, executable 12-part audit of *Hilbert Operator for Progressive
Encoding (HOPE)*, with bounded diagnostics and a proposed corrected reference
design.

- [Read the 12-part site](https://basharelnashef.github.io/hope-under-the-microscope/)
- [Download the compiled report](downloads/hope-under-the-microscope.pdf)
- [Review the evidence limits](KNOWN_LIMITATIONS.md)

This is not author code and does not claim an exact reproduction of the
paper's ResNet-50/ImageNet or complete DEFT experiments.

## The 12 parts

| Part | Detailed breakdown | Reproduction package |
|---:|---|---|
| 1 | [A neuron is not a filter](https://basharelnashef.github.io/hope-under-the-microscope/parts/01-neuron-not-filter/) | [Positive-rescaling audit](experiments/audits/part01_positive_rescaling) |
| 2 | [The closed-form capacity inside HOPE](https://basharelnashef.github.io/hope-under-the-microscope/parts/02-closed-form-capacity/) | [Capacity audit](experiments/audits/part02_capacity) |
| 3 | [What data-free actually means](https://basharelnashef.github.io/hope-under-the-microscope/parts/03-data-free-meaning/) | [Data-free audit](experiments/audits/part03_data_free) |
| 4 | [When pure maximum entropy is unbounded](https://basharelnashef.github.io/hope-under-the-microscope/parts/04-maximum-entropy/) | [Maximum-entropy audit](experiments/audits/part04_maximum_entropy) |
| 5 | [When valid pairs do not form a Hilbert space](https://basharelnashef.github.io/hope-under-the-microscope/parts/05-one-hilbert-space/) | [Global-PSD audit](experiments/audits/part05_global_psd) |
| 6 | [What the axioms derive, and what they do not](https://basharelnashef.github.io/hope-under-the-microscope/parts/06-axioms/) | [Axiom audit](experiments/audits/part06_axioms) |
| 7 | [The zero-cost merge that removes one copy](https://basharelnashef.github.io/hope-under-the-microscope/parts/07-zero-cost-merge/) | [Merge-semantics audit](experiments/audits/part07_merge_semantics) |
| 8 | [Residual-block eviction: topology and a conditional Gaussian stress test](https://basharelnashef.github.io/hope-under-the-microscope/parts/08-residual-eviction/) | [Block-eviction audit](experiments/audits/part08_block_eviction) |
| 9 | [Constant time per candidate is not a constant-time optimizer](https://basharelnashef.github.io/hope-under-the-microscope/parts/09-constant-time/) | [Complexity audit](experiments/audits/part09_constant_time) |
| 10 | [DEFT has multiple definitions of core](https://basharelnashef.github.io/hope-under-the-microscope/parts/10-deft-core/) | [DEFT-core audit](experiments/audits/part10_deft_core) |
| 11 | [What the experiments establish](https://basharelnashef.github.io/hope-under-the-microscope/parts/11-experimental-evidence/) | [Evidence audit](experiments/audits/part11_experimental_evidence) |
| 12 | [Building a corrected HOPE](https://basharelnashef.github.io/hope-under-the-microscope/parts/12-corrected-hope/) | [HOPE-R reference design](experiments/audits/part12_corrected_hope) |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make test
make verify-results
```

Each part contains a short README, a standalone `run.py`, and focused unit
tests. Published CSV and JSON outputs live under `artifacts/`. See
[`REPRODUCING.md`](REPRODUCING.md) for fresh-run commands and external-input
boundaries.

## Questions, corrections, and contributions

Email [Bashar Elnashef](mailto:elnashefb@gmail.com) at
[`elnashefb@gmail.com`](mailto:elnashefb@gmail.com) with questions,
corrections, reproducibility reports, or proposals. Public problems can be
reported through the
[issue tracker](https://github.com/basharelnashef/hope-under-the-microscope/issues),
and code or documentation contributions are covered by
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If this project is useful in your work, please cite
[Bashar Elnashef](https://scholar.google.com/citations?user=tHNUc5wAAAAJ) and
the original HOPE paper. GitHub's **Cite this repository** menu uses
[`CITATION.cff`](CITATION.cff); a ready-to-copy BibTeX record is available in
[`CITATION.bib`](CITATION.bib).

```bibtex
@software{elnashef2026hope,
  author  = {Elnashef, Bashar},
  title   = {{HOPE} Under the Microscope: Independent Reproduction, Audit, and Extensions},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/basharelnashef/hope-under-the-microscope},
  note    = {Independent executable 12-part audit of HOPE v1}
}
```

## Release contents

```text
artifacts/           Selected final CSV and JSON results
downloads/           Compiled report
experiments/audits/  Twelve standalone reproduction packages
paper/               Paper identity and equation map, not the paper PDF
site/                Publication-ready Quarto articles and web figures
src/                 Shared paper-literal functions and proposed repairs
tests/               Focused cross-part tests
```

Original software is Apache-2.0. Original writing and figures are CC BY 4.0.

# Reproducing the results

## Environment

Python 3.10 or newer is required. A clean environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run all focused tests and authenticate the published result files:

```bash
make test
make verify-results
```

The tests validate formulas, boundary behavior, schemas, and deterministic
fixtures. The checksum target authenticates the 80 selected CSV/JSON files and
all 15 static editorial PNGs exactly as published; it does not
require regenerated floating-point result files to be byte-identical across
platforms.

## Running an individual part

Every directory under `experiments/audits/` contains its own scope statement,
command-line options, and focused tests. For example:

```bash
python experiments/audits/part05_global_psd/run.py \
  --result-dir runs/part_05

python -m unittest discover \
  -s experiments/audits/part05_global_psd \
  -p 'test_*.py' -v
```

Use a new output directory for each run. The programs refuse unsafe
overwrites where the audit contract requires a fresh destination.

## Part 3 external inputs

Part 3 is the one dataset-backed diagnostic. Its complete run requires the
declared VGG-8 checkpoint, CIFAR-10 test data, PyTorch, and the pinned paper
PDF. Those external files are not redistributed. Install the optional runtime
and follow the exact path and checksum instructions in
[`experiments/audits/part03_data_free/README.md`](experiments/audits/part03_data_free/README.md):

```bash
python -m pip install -e '.[part3]'
```

The other parts use analytical or deterministic local fixtures and do not
download a dataset or checkpoint.

## Rendering the site

With Quarto 1.9.38 installed:

```bash
quarto render site
```

The generated `_site` directory is intentionally not versioned.

## Questions and reproducibility reports

For a public, reproducible problem, open a
[GitHub issue](https://github.com/basharelnashef/hope-under-the-microscope/issues)
with the command, environment, observed output, and expected result. Questions
or reports containing private context can be sent to
[Bashar Elnashef](mailto:elnashefb@gmail.com). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete reporting and contribution
guidance.

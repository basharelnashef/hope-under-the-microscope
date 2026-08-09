# Published results

Each `part_NN/results/` directory contains the selected CSV and JSON outputs
used by the corresponding article. These are final result files, not raw
datasets, caches, checkpoints, or build intermediates.

`SHA256SUMS` authenticates the 80 published CSV/JSON result files and all 15
static editorial PNG figures used by Parts 1, 2, 6, and 7. Authenticate all 95
files from the repository root with:

```bash
sha256sum --check artifacts/SHA256SUMS
```

Fresh result runs may differ in insignificant floating-point digits across
NumPy and linear-algebra environments. The focused tests define the semantic
tolerances; `SHA256SUMS` authenticates the files exactly as published. The
15 PNGs are editorial visualizations of nearby authenticated tables and
derivations, not generator outputs; their hashes authenticate the displayed
bytes but do not claim plot-code reproducibility.

If a published checksum fails, do not overwrite the result. Record the file,
command, environment, expected digest, and observed digest in a
[GitHub issue](https://github.com/basharelnashef/hope-under-the-microscope/issues)
or follow the private-contact route in the
[contribution guide](../CONTRIBUTING.md).

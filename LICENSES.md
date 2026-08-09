# Licensing

The repository uses file-level licensing; it is not offering every file under
either license interchangeably.

## Apache License 2.0

The following original software, tests, and build/automation files are
licensed under Apache License 2.0; see [`LICENSE`](LICENSE):

- `src/**/*.py`
- `tests/**/*.py`
- `experiments/**/*.py`
- `Makefile` and `pyproject.toml`
- `.github/**/*.yml`
- `.gitattributes`, `.gitignore`, and `site/.gitignore`
- `site/_quarto.yml` and `site/styles.css`

## Creative Commons Attribution 4.0 International

The following original prose, metadata, published result data, diagrams, and
compiled content are licensed under Creative Commons Attribution 4.0
International; see [`LICENSE-CONTENT`](LICENSE-CONTENT):

- `*.md`, `experiments/**/*.md`, `src/**/*.md`, and `artifacts/**/*.md`
- `.github/**/*.md`
- `site/**/*.qmd`, `site/**/*.svg`, and `site/assets/**/*.png`
- `artifacts/**/*.csv`, `artifacts/**/*.json`, and `artifacts/SHA256SUMS`
- `paper/*.json`
- `CITATION.bib` and `CITATION.cff`
- `downloads/hope-under-the-microscope.pdf` and `downloads/SHA256SUMS`

Preferred attribution: Bashar Elnashef, *HOPE Under the Microscope*.

`LICENSE` and `LICENSE-CONTENT` reproduce their respective license texts and
remain governed by the notices in those texts.

These licenses do not cover the HOPE paper, datasets, checkpoints, third-party
libraries, or other external material. Each external work retains its own
terms. The paper PDF is identified by checksum but is not redistributed.

The root `CITATION.cff` describes the software component and therefore carries
the single value `Apache-2.0`. CFF 1.2 defines a list of licenses as an OR
relationship, which would incorrectly imply that every repository file may be
used under either license. Its `abstract` points consumers to this file-level
mapping instead.

# Contributing

Questions, corrections, reproducibility reports, proposals, and contributions
are welcome.

## Choose the right channel

- **Questions or private context:** email
  [Bashar Elnashef](mailto:elnashefb@gmail.com) at `elnashefb@gmail.com`.
- **Corrections and reproducible problems:** open a
  [focused correction report](https://github.com/basharelnashef/hope-under-the-microscope/issues/new?template=correction.yml)
  so the evidence and resolution remain visible.
- **Proposals and extensions:** use the
  [proposal form](https://github.com/basharelnashef/hope-under-the-microscope/issues/new?template=proposal.yml)
  or email before investing in a large implementation.
- **Code, tests, documentation, or figures:** open a pull request against
  `main`.
- **Sensitive reports:** follow [`SECURITY.md`](SECURITY.md) and use email
  rather than a public issue.

## What makes a useful report

Please include:

1. the affected part, file, equation, claim, result, or URL;
2. what you observed and what you expected;
3. the smallest reproducible example or supporting source;
4. your operating system, Python version, and relevant dependency versions;
5. whether the problem affects the paper-literal audit, a proposed repair, or
   only the presentation.

## Pull requests

Keep changes focused and preserve the project's evidence boundaries. Before
submitting:

```bash
python -m pip install -e .
make test
make verify-results
```

Add or update tests for behavioral changes. Do not overwrite authenticated
results without explaining how they were regenerated and updating the
corresponding checksum index. Keep paper claims, verified facts, independent
inferences, counterexamples, proposed repairs, and open questions clearly
separated.

Contributions are accepted under the license that applies to the files being
changed: Apache-2.0 for software and CC BY 4.0 for original writing and
figures.

There is no guaranteed response time. Focused reports with concrete evidence
are the easiest to evaluate and preserve in the public record.

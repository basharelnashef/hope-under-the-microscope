PYTHON ?= python3
QUARTO ?= quarto

.PHONY: test verify-results site

test:
	@set -e; \
	for audit in experiments/audits/part*; do \
		$(PYTHON) -m unittest discover -s "$$audit" -p 'test_*.py' -v; \
	done
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

verify-results:
	sha256sum --check artifacts/SHA256SUMS

site:
	$(QUARTO) render site

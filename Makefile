PY ?= python

.PHONY: help setup test verify-paper reproduce-figures check-release

help:
	@echo "Targets:"
	@echo "  setup              install package and development dependencies"
	@echo "  test               run the no-GPU unit and artifact tests"
	@echo "  verify-paper       verify published headline claims from bundled data"
	@echo "  reproduce-figures  regenerate figures from bundled result CSVs"
	@echo "  check-release      scan the public artifact for unsafe local material"

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

verify-paper:
	$(PY) scripts/verify_paper_claims.py

reproduce-figures:
	$(PY) code/plotting/make_figures.py --out reproduced/figures --tables-out reproduced/tables

check-release:
	$(PY) scripts/check_release.py .

PYTHON ?= python

.PHONY: verify figures full
verify:
	$(PYTHON) verify.py all
	$(PYTHON) theory.py --self-check

figures:
	$(PYTHON) report.py all

full:
	$(PYTHON) experiments.py full --workers 2
	$(PYTHON) report.py all --data reproduction/full

PYTEST := .venv/bin/python -m pytest
PYTHON := .venv/bin/python

up:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

test:
	$(PYTEST) tests/unit/ tests/integration/ tests/functional/

test-unit:
	$(PYTEST) tests/unit/

test-integration:
	$(PYTEST) tests/integration/

test-bench:
	$(PYTEST) tests/bench/

test-functional:
	$(PYTEST) tests/functional/

experiment01-recall-under-noise:
	$(PYTEST) tests/experiments/test_exp01_recall_under_noise.py -v -s

results01-recall-under-noise:
	$(PYTHON) scripts/show_results.py

results01-float:
	$(PYTHON) scripts/show_results.py $(shell ls -t test_results/recall_under_noise_float_*.json | head -1)

results01-binary:
	$(PYTHON) scripts/show_results.py $(shell ls -t test_results/recall_under_noise_binary_*.json | head -1)

experiment02-dedup-recall:
	$(PYTEST) tests/experiments/test_exp02_dedup_recall.py -v -s

results02-dedup-recall:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dedup_recall_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment03-weights:
	$(PYTEST) tests/experiments/test_exp03_field_weighting.py -v -s

results03-weights:
	@for mode in binary float; do \
		file=$$(ls -t test_results/field_weighting_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment04-scalability:
	$(PYTEST) tests/experiments/test_exp04_scalability.py -v -s

results04-scalability:
	@for mode in binary float; do \
		file=$$(ls -t test_results/scalability_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment05-ranking:
	$(PYTEST) tests/experiments/test_exp05_ranking_metrics.py -v -s

results05-ranking:
	@for mode in binary float; do \
		file=$$(ls -t test_results/ranking_metrics_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment06-per-field-noise:
	$(PYTEST) tests/experiments/test_exp06_per_field_noise.py -v -s

results06-per-field-noise:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_noise_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment07-per-field-sweep:
	$(PYTEST) tests/experiments/test_exp07_per_field_noise_sweep.py -v -s

results07-per-field-sweep:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_sweep_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment08-dimensionality:
	$(PYTEST) tests/experiments/test_exp08_dimensionality.py -v -s

results08-dimensionality:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dimensionality_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment09-date-encoding:
	$(PYTEST) tests/experiments/test_exp09_date_encoding.py -v -s

results09-date-encoding:
	@for mode in binary float; do \
		file=$$(ls -t test_results/date_encoding_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done


experiment10-scalability-noisy-dupes:
	$(PYTEST) tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

results10-scalability-noisy-dupes:
	@for mode in binary float; do \
		file=$$(ls -t test_results/exp10_scalability_noisy_dupes/exp10_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment11-nk-sweep:
	$(PYTEST) tests/experiments/test_exp11_recall_nk_sweep.py -v -s

results11-nk-sweep:
	@file=$$(ls -t test_results/recall_nk_sweep_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; \
	else echo "No recall_nk_sweep results found in test_results/"; fi

experiment12-recall-n-sweep:
	$(PYTEST) tests/experiments/test_exp12_recall_n_sweep.py -v -s

results12-recall-n-sweep:
	@file=$$(ls -t test_results/exp12_recall_n_sweep_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; \
	else echo "No exp12 results found in test_results/"; fi

experiment13-separability:
	$(PYTHON) tests/experiments/test_exp13_separability_analysis.py

results13-separability:
	@file=$$(ls -t results/exp13_separability_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; \
	else echo "No exp13_separability results found in results/"; fi

PYTEST := .venv311/bin/python -m pytest
PYTHON := .venv311/bin/python

# nprobe default — override with NPROBE=128 for exhaustive mode B
# Experiment scripts route their own output based on HDC_NPROBE: nprobe=8 runs
# write to test_results/, nprobe=128 runs write to test_results_128/ with an
# "_exhaustive" filename marker (exp13 always runs exhaustive, unconditionally).
# The resultsNN-* targets below look in both directories.
NPROBE ?= 8

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

# ---------------------------------------------------------------------------
# Run ALL experiments — choose mode A (ANN, nprobe=8) or B (exhaustive, nprobe=128)
#
#   make experiments-ann          # mode A: true approximate search
#   make experiments-exhaustive   # mode B: exhaustive reference
# ---------------------------------------------------------------------------

experiments-ann:
	$(MAKE) _run-all-experiments NPROBE=8

experiments-exhaustive:
	$(MAKE) _run-all-experiments NPROBE=128

_run-all-experiments:
	@echo "=== Running ALL experiments with nprobe=$(NPROBE) ==="
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp01_recall_under_noise.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp02_dedup_recall.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp03_field_weighting.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp04_scalability.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp05_ranking_metrics.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp06_per_field_noise.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp07_per_field_noise_sweep.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp08_dimensionality.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp09_date_encoding.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp11_recall_nk_sweep.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTEST) tests/experiments/test_exp12_recall_n_sweep.py -v -s
	HDC_NPROBE=$(NPROBE) $(PYTHON) tests/experiments/test_exp13_separability_analysis.py
	@echo "=== All experiments completed (nprobe=$(NPROBE)) ==="

# ---------------------------------------------------------------------------

experiment01-recall-under-noise-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp01_recall_under_noise.py -v -s

experiment01-recall-under-noise-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp01_recall_under_noise.py -v -s

results01-recall-under-noise:
	$(PYTHON) scripts/show_results.py

results01-float:
	@ann=$$(ls -t test_results/recall_under_noise_float_*.json 2>/dev/null | head -1); \
	exh=$$(ls -t test_results_128/recall_under_noise_float_exhaustive_*.json 2>/dev/null | head -1); \
	if [ -z "$$ann" ] && [ -z "$$exh" ]; then echo "No recall_under_noise_float results found."; fi; \
	if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
	if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi

results01-binary:
	@ann=$$(ls -t test_results/recall_under_noise_binary_*.json 2>/dev/null | head -1); \
	exh=$$(ls -t test_results_128/recall_under_noise_binary_exhaustive_*.json 2>/dev/null | head -1); \
	if [ -z "$$ann" ] && [ -z "$$exh" ]; then echo "No recall_under_noise_binary results found."; fi; \
	if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
	if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi

experiment02-dedup-recall-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp02_dedup_recall.py -v -s

experiment02-dedup-recall-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp02_dedup_recall.py -v -s

results02-dedup-recall:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/dedup_recall_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/dedup_recall_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment03-weights:
	$(PYTEST) tests/experiments/test_exp03_field_weighting.py -v -s

results03-weights:
	@for mode in binary float; do \
		file=$$(ls -t test_results/field_weighting_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment04-scalability-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp04_scalability.py -v -s

experiment04-scalability-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp04_scalability.py -v -s

results04-scalability:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/scalability_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/scalability_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment05-ranking-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp05_ranking_metrics.py -v -s

experiment05-ranking-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp05_ranking_metrics.py -v -s

results05-ranking:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/ranking_metrics_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/ranking_metrics_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment06-per-field-noise:
	$(PYTEST) tests/experiments/test_exp06_per_field_noise.py -v -s

results06-per-field-noise:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_noise_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done

experiment07-per-field-sweep-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp07_per_field_noise_sweep.py -v -s

experiment07-per-field-sweep-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp07_per_field_noise_sweep.py -v -s

results07-per-field-sweep:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/per_field_sweep_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/per_field_sweep_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment08-dimensionality-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp08_dimensionality.py -v -s

experiment08-dimensionality-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp08_dimensionality.py -v -s

results08-dimensionality:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/dimensionality_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/dimensionality_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment09-date-encoding:
	$(PYTEST) tests/experiments/test_exp09_date_encoding.py -v -s

results09-date-encoding:
	@for mode in binary float; do \
		file=$$(ls -t test_results/date_encoding_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; fi \
	done


experiment10-scalability-noisy-dupes-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

experiment10-scalability-noisy-dupes-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

results10-scalability-noisy-dupes:
	@for mode in binary float; do \
		ann=$$(ls -t test_results/exp10_scalability_noisy_dupes/exp10_$${mode}_*.json 2>/dev/null | head -1); \
		exh=$$(ls -t test_results_128/exp10_scalability_noisy_dupes/exp10_$${mode}_exhaustive_*.json 2>/dev/null | head -1); \
		if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
		if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi; \
	done

experiment11-nk-sweep-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp11_recall_nk_sweep.py -v -s

experiment11-nk-sweep-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp11_recall_nk_sweep.py -v -s

results11-nk-sweep:
	@ann=$$(ls -t test_results/recall_nk_sweep_*.json 2>/dev/null | head -1); \
	exh=$$(ls -t test_results_128/recall_nk_sweep_exhaustive_*.json 2>/dev/null | head -1); \
	if [ -z "$$ann" ] && [ -z "$$exh" ]; then echo "No recall_nk_sweep results found."; fi; \
	if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
	if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi

experiment12-recall-n-sweep-ann:
	HDC_NPROBE=8 $(PYTEST) tests/experiments/test_exp12_recall_n_sweep.py -v -s

experiment12-recall-n-sweep-exhaustive:
	HDC_NPROBE=128 $(PYTEST) tests/experiments/test_exp12_recall_n_sweep.py -v -s

results12-recall-n-sweep:
	@ann=$$(ls -t test_results/exp12_recall_n_sweep_*.json 2>/dev/null | head -1); \
	exh=$$(ls -t test_results_128/exp12_recall_n_sweep_exhaustive_*.json 2>/dev/null | head -1); \
	if [ -z "$$ann" ] && [ -z "$$exh" ]; then echo "No exp12 results found."; fi; \
	if [ -n "$$ann" ]; then $(PYTHON) scripts/show_results.py $$ann; fi; \
	if [ -n "$$exh" ]; then $(PYTHON) scripts/show_results.py $$exh; fi

experiment13-separability:
	$(PYTHON) tests/experiments/test_exp13_separability_analysis.py

results13-separability:
	@file=$$(ls -t test_results_128/exp13_separability_exhaustive_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then $(PYTHON) scripts/show_results.py $$file; \
	else echo "No exp13_separability results found in test_results_128/ (exp13 always runs exhaustive)"; fi


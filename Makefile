up:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

test:
	pytest tests/unit/ tests/integration/ tests/functional/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-bench:
	pytest tests/bench/

test-functional:
	pytest tests/functional/

experiment:
	pytest tests/experiments/test_recall_under_noise.py -v -s

results:
	python scripts/show_results.py

results-float:
	python scripts/show_results.py $(shell ls -t test_results/recall_under_noise_float_*.json | head -1)

results-binary:
	python scripts/show_results.py $(shell ls -t test_results/recall_under_noise_binary_*.json | head -1)

experiment-dedup:
	pytest tests/experiments/test_dedup_recall.py -v -s

results-dedup:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dedup_recall_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-weights:
	pytest tests/experiments/test_field_weighting.py -v -s

results-weights:
	@for mode in binary float; do \
		file=$$(ls -t test_results/field_weighting_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-scalability:
	pytest tests/experiments/test_scalability.py -v -s

results-scalability:
	@for mode in binary float; do \
		file=$$(ls -t test_results/scalability_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-ranking:
	pytest tests/experiments/test_ranking_metrics.py -v -s

results-ranking:
	@for mode in binary float; do \
		file=$$(ls -t test_results/ranking_metrics_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-per-field-noise:
	pytest tests/experiments/test_per_field_noise.py -v -s

results-per-field-noise:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_noise_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-per-field-sweep:
	pytest tests/experiments/test_per_field_noise_sweep.py -v -s

results-per-field-sweep:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_sweep_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-dimensionality:
	pytest tests/experiments/test_dimensionality.py -v -s

results-dimensionality:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dimensionality_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-date-encoding:
	pytest tests/experiments/test_date_encoding.py -v -s

results-date-encoding:
	@for mode in binary float; do \
		file=$$(ls -t test_results/date_encoding_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment-nk-sweep:
	pytest tests/experiments/test_recall_nk_sweep.py -v -s

results-nk-sweep:
	@file=$$(ls -t test_results/recall_nk_sweep_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then python scripts/show_results.py $$file; \
	else echo "No recall_nk_sweep results found in test_results/"; fi


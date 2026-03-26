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

experiment01-recall-under-noise:
	pytest tests/experiments/test_exp01_recall_under_noise.py -v -s

results01-recall-under-noise:
	python scripts/show_results.py

results01-float:
	python scripts/show_results.py $(shell ls -t test_results/recall_under_noise_float_*.json | head -1)

results01-binary:
	python scripts/show_results.py $(shell ls -t test_results/recall_under_noise_binary_*.json | head -1)

experiment02-dedup-recall:
	pytest tests/experiments/test_exp02_dedup_recall.py -v -s

results02-dedup-recall:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dedup_recall_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment03-weights:
	pytest tests/experiments/test_exp03_field_weighting.py -v -s

results03-weights:
	@for mode in binary float; do \
		file=$$(ls -t test_results/field_weighting_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment04-scalability:
	pytest tests/experiments/test_exp04_scalability.py -v -s

results04-scalability:
	@for mode in binary float; do \
		file=$$(ls -t test_results/scalability_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment05-ranking:
	pytest tests/experiments/test_exp05_ranking_metrics.py -v -s

results05-ranking:
	@for mode in binary float; do \
		file=$$(ls -t test_results/ranking_metrics_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment06-per-field-noise:
	pytest tests/experiments/test_exp06_per_field_noise.py -v -s

results06-per-field-noise:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_noise_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment07-per-field-sweep:
	pytest tests/experiments/test_exp07_per_field_noise_sweep.py -v -s

results07-per-field-sweep:
	@for mode in binary float; do \
		file=$$(ls -t test_results/per_field_sweep_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment08-dimensionality:
	pytest tests/experiments/test_exp08_dimensionality.py -v -s

results08-dimensionality:
	@for mode in binary float; do \
		file=$$(ls -t test_results/dimensionality_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment09-date-encoding:
	pytest tests/experiments/test_exp09_date_encoding.py -v -s

results09-date-encoding:
	@for mode in binary float; do \
		file=$$(ls -t test_results/date_encoding_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done


experiment10-scalability-noisy-dupes:
	pytest tests/experiments/test_exp10_scalability_noisy_dupes.py -v -s

results10-scalability-noisy-dupes:
	@for mode in binary float; do \
		file=$$(ls -t test_results/exp10_scalability_noisy_dupes/exp10_$${mode}_*.json 2>/dev/null | head -1); \
		if [ -n "$$file" ]; then python scripts/show_results.py $$file; fi \
	done

experiment11-nk-sweep:
	pytest tests/experiments/test_recall_nk_sweep.py -v -s

results11-nk-sweep:
	@file=$$(ls -t test_results/recall_nk_sweep_*.json 2>/dev/null | head -1); \
	if [ -n "$$file" ]; then python scripts/show_results.py $$file; \
	else echo "No recall_nk_sweep results found in test_results/"; fi

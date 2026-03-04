up:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

test:
	pytest tests/unit/ tests/integration/ tests/bench/ tests/functional/

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

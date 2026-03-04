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

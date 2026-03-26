.PHONY: install train test lint serve docker-build gradio audit

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python src/training/train.py --config config/config.yaml

test:
	pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

lint:
	black src/ tests/ && flake8 src/ tests/ && isort src/ tests/ && mypy src/ && bandit -r src/ -ll -ii

serve:
	uvicorn src.api.app:app --reload --port 8000

docker-build:
	docker build -t b1-hf-fastapi .

gradio:
	python src/api/gradio_demo.py

audit:
	pip-audit -r requirements.txt --ignore-vuln CVE-2026-4539

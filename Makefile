.PHONY: help install data data-from-drive features train simulate pipeline predict-today dashboard test lint docker-build docker-run report run_all

help:
	@echo "Available targets:"
	@echo "  make install          install dependencies"
	@echo "  make data             download source data (yf, fred, dgap)"
	@echo "  make data-from-drive  download pre-built parquet from Drive"
	@echo "  make features         build unified dataset + features"
	@echo "  make train            train & tune all models"
	@echo "  make simulate         run all trading simulations"
	@echo "  make report           consolidate reports + equity-curve plot"
	@echo "  make run_all          full pipeline (data + features + train + simulate + report)"
	@echo "  make predict-today    incremental daily prediction"
	@echo "  make dashboard        launch Streamlit dashboard"
	@echo "  make docker-build     build Docker image"
	@echo "  make docker-run       run pipeline inside Docker"
	@echo "  make test             run pytest"
	@echo "  make lint             ruff check"

install:
	pip install -r requirements.txt

data:
	python3 scripts/run_pipeline.py --download-only

data-from-drive:
	python3 scripts/run_pipeline.py --from-drive

features:
	python3 scripts/run_pipeline.py --features-only

train:
	python3 scripts/run_pipeline.py --train-only

simulate:
	python3 scripts/run_pipeline.py --simulate-only

pipeline:
	python3 scripts/run_pipeline.py --full

run_all:
	python3 scripts/run_pipeline.py
	python3 scripts/make_report_plots.py

report:
	python3 scripts/make_report_plots.py

predict-today:
	python3 scripts/daily_predict.py

dashboard:
	streamlit run src/monitoring/dashboard.py --server.port=8501

docker-build:
	docker build -t eu-midcap-echo:latest -f docker/Dockerfile .

docker-run:
	docker run --rm -v $(PWD)/storage:/app/storage -v $(PWD)/models:/app/models \
	           -v $(PWD)/reports:/app/reports eu-midcap-echo:latest

test:
	pytest tests/ -q

lint:
	ruff check src/ scripts/ tests/

PY ?= python

.PHONY: install data train index pipeline test regress baseline bench serve docker

install:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m ml.data_generation

train:
	$(PY) -m ml.train

index:
	$(PY) -m ml.vector_store

pipeline: data train index   ## regenerate data, retrain, rebuild vector index

test:
	$(PY) -m pytest -q

regress:                      ## fail if quality/latency/routing regressed vs baseline
	$(PY) -m ml.regression_check

baseline:                     ## accept current numbers as the new baseline
	$(PY) -m ml.regression_check --update-baseline

bench:
	$(PY) -m ml.benchmark_parallel

serve:
	$(PY) -m uvicorn app.main:app --reload --port 8000

docker:
	docker compose up --build

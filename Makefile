PY ?= python

.PHONY: install data train index pipeline test lint regress baseline bench serve docker mlflow-ui models

install:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m ml.data_generation

train:                        ## train, log to MLflow, register; promote to @champion if not worse
	$(PY) -m ml.train

index:
	$(PY) -m ml.vector_store

pipeline: data train index   ## regenerate data, retrain, rebuild vector index

test:
	$(PY) -m pytest -q

lint:                         ## pyright (strict) + ruff
	$(PY) -m pyright
	$(PY) -m ruff check app agent ml tests

regress:                      ## fail if quality/latency/routing regressed vs baseline
	$(PY) -m ml.regression_check

baseline:                     ## accept current numbers as the new baseline
	$(PY) -m ml.regression_check --update-baseline

bench:
	$(PY) -m ml.benchmark_parallel

models:                       ## registry versions per model; * marks @champion
	$(PY) -m ml.promote

mlflow-ui:                    ## runs, registered models and agent traces at http://127.0.0.1:5050
	$(PY) -m mlflow ui --backend-store-uri sqlite:///ml/artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5050

serve:
	$(PY) -m uvicorn app.main:app --reload --port 8000

docker:
	docker compose up --build

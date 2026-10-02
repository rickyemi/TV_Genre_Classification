# TV genre classification: automation commands
# Usage: `make help`

PYTHON ?= python
export MPLBACKEND = Agg
IMAGE  ?= tv-genre-classification:latest

.DEFAULT_GOAL := help
.PHONY: help install data eda features train evaluate stress predict regression all fast \
        notebooks test lint clean clean-all docker-build docker-run docker-test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

data:  ## Audit + clean the raw feeds, 80/20 split of the labelled shows
	$(PYTHON) -m src.data.make_dataset

eda: data  ## Exploratory analysis and data-quality figures (01-11)
	$(PYTHON) -m src.analysis.eda

features: data  ## Preprocessing + feature selection (figures 12-16)
	$(PYTHON) -m src.features.build_features

train: features  ## Optimisation steps + RandomizedSearchCV: SSGMM, XGBoost, SVM-RBF (+ screened DeepMLP) (~3 min)
	$(PYTHON) -m src.models.train_model

evaluate: data  ## Holdout + repeated-CV Train/Test comparison, >= 0.89 acceptance check, figures 19-25
	$(PYTHON) -m src.models.evaluate_model

stress: data  ## 11 stress tests + scorecard (figures 26-27, ~1 min)
	$(PYTHON) -m src.models.stress_test

predict: data  ## Genre + probabilities for the 400 unlabelled shows (run after evaluate)
	$(PYTHON) -m src.models.predict_model

regression: data  ## J from C + two uncorrelated supporting features (figures 29-32)
	$(PYTHON) -m src.models.regression_j

all:  ## Run the full pipeline end to end
	$(PYTHON) -m src.pipeline

fast:  ## Full pipeline without the stress tests
	$(PYTHON) -m src.pipeline --skip-stress

notebooks:  ## Execute every notebook in place (outputs saved)
	cd notebooks && for nb in 0*.ipynb; do \
	  $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=1800 $$nb || exit 1; done

test:  ## Run the unit and integration tests
	$(PYTHON) -m pytest -q

lint:  ## Syntax-check every module
	$(PYTHON) -m compileall -q src tests

clean:  ## Remove generated interim/processed data and caches
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find data/interim data/processed -type f ! -name ".gitkeep" -delete

clean-all: clean  ## Also remove models, figures, tables and predictions
	rm -f models/*.joblib models/*.json
	rm -f reports/figures/*.png reports/tables/* reports/*.csv

docker-build:  ## Build the Docker image
	docker build -t $(IMAGE) .

docker-run:  ## Run the full pipeline inside Docker
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) make all

docker-test:  ## Run the tests inside Docker
	docker run --rm $(IMAGE) make test

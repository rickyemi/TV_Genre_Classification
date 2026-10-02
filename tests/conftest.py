"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402


@pytest.fixture(scope="session")
def data():
    """Cleaned splits, rebuilt from the raw CSVs if needed."""
    from src.data import make_dataset
    if not cfg.TRAIN_FILE.exists():
        make_dataset.main()
    return {n: make_dataset.load(n) for n in ["train", "holdout", "unlabelled", "labelled"]}


@pytest.fixture(scope="session")
def models():
    if not all(cfg.model_file(m, "production").exists() for m in cfg.MODEL_NAMES):
        pytest.skip("models not trained yet - run `make train`")
    from src.models.train_model import load_models
    return {"evaluation": load_models("evaluation"), "production": load_models("production")}

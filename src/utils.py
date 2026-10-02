"""Small shared helpers: logging, seeding and JSON I/O."""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single, consistently formatted handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the non-Transformer stages
        pass


def save_json(obj, path: Path) -> None:
    """Write an object to JSON (NumPy scalars are converted to Python types)."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)

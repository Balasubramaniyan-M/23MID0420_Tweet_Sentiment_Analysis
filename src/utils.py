"""Central configuration, constants, and utility functions.

This module defines all project-wide constants, paths, and helper
functions used across the MDI3003 Lab 05 pipeline.
"""

from pathlib import Path
import json
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ── Project Configuration ──────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # of total; carved from train
N_FOLDS = 5
PRIMARY_METRIC = "f1_macro"
MIN_ENTITY_SUPPORT = 30

# ── Student Info ───────────────────────────────────────────────────────
STUDENT_NAME = "Balasubramaniyan M"
REG_NUMBER = "23MID0420"
COURSE = "MDI3003 – Advanced Predictive Analytics"
EXPERIMENT = "Experiment 05 – Product and Brand Sentiment Prediction from Tweet Data"
SEMESTER = "Fall Semester 2026–2027"

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

# ── Dataset Configuration ──────────────────────────────────────────────
DATASET_NAME = "Twitter US Airline Sentiment"
DATASET_URL = "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment"
DATASET_FILENAME = "Tweets.csv"
TEXT_COLUMN = "text"
TARGET_COLUMN = "airline_sentiment"
ENTITY_COLUMN = "airline"
TWEET_ID_COLUMN = "tweet_id"
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# Columns to EXCLUDE from modeling (leakage/privacy)
LEAKAGE_COLUMNS = [
    "airline_sentiment_confidence",
    "negativereason",
    "negativereason_confidence",
    "airline_sentiment_gold",
    "negativereason_gold",
]
PRIVACY_COLUMNS = [
    "name",  # Twitter username
    "tweet_id",
    "retweet_count",
    "tweet_coord",
    "tweet_location",
    "user_timezone",
]
EXCLUDED_COLUMNS = LEAKAGE_COLUMNS + PRIVACY_COLUMNS

# ── Target Definitions ─────────────────────────────────────────────────
SENTIMENT_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def ensure_directories() -> None:
    """Create all output directories if they do not exist."""
    for d in [DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
              REPORTS_DIR, SUBMISSION_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across numpy and stdlib."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_environment_info() -> Dict[str, str]:
    """Collect runtime environment metadata for reproducibility."""
    import sklearn
    import matplotlib
    import seaborn
    import nltk
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "seaborn_version": seaborn.__version__,
        "nltk_version": nltk.__version__,
        "seed": str(SEED),
        "timestamp": datetime.now().isoformat(),
    }
    return info


def save_json(data: Any, path: Path) -> None:
    """Save a dictionary or list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {path}")


def load_json(path: Path) -> Any:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱ {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_section(title: str, level: int = 1) -> None:
    """Print a formatted section header."""
    if level == 1:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    elif level == 2:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
    else:
        print(f"\n  ▸ {title}")

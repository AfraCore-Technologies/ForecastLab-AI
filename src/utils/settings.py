from __future__ import annotations

from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main folders
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Input files
TRAIN_DATA_PATH = INPUT_DIR / "train.csv"
TEST_DATA_PATH = INPUT_DIR / "test.csv"
HYPERPARAMETERS_PATH = INPUT_DIR / "hyperparameters.json"

# Output files
MODEL_OUTPUT_PATH = OUTPUT_DIR / "model.pkl"
PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"

# Optional alternative formats
TRAIN_PARQUET_PATH = INPUT_DIR / "train.parquet"
TEST_PARQUET_PATH = INPUT_DIR / "test.parquet"
PREDICTIONS_PARQUET_PATH = OUTPUT_DIR / "predictions.parquet"


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
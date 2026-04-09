from __future__ import annotations

from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main folders
INPUT_DIR = PROJECT_ROOT / "input"
CONFIG_DIR = INPUT_DIR / "config"
DATA_DIR = INPUT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Input files
TRAIN_DATA_PATH = TRAIN_DIR / "train.csv"
TEST_DATA_PATH = TEST_DIR / "test.csv"
HYPERPARAMETERS_PATH = CONFIG_DIR / "hyperparameters.json"

# Output files
MODEL_OUTPUT_PATH = OUTPUT_DIR / "model.pkl"
PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"

# Optional alternative formats
TRAIN_PARQUET_PATH = TRAIN_DIR / "train.parquet"
TEST_PARQUET_PATH = TEST_DIR / "test.parquet"
PREDICTIONS_PARQUET_PATH = OUTPUT_DIR / "predictions.parquet"


def ensure_directories() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
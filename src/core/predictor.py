from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .prepare import Prepare
from ..utils.settings import (
    MODEL_OUTPUT_PATH,
    PREDICTIONS_OUTPUT_PATH,
    TEST_DATA_PATH,
    ensure_directories,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact: Any, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or LOGGER
        self.artifact = artifact

        if isinstance(artifact, dict):
            self.algorithm = str(artifact.get("algorithm", "")).strip().lower()
            self.frequency = str(artifact.get("frequency", "daily")).strip().lower()
            predictions = artifact.get("predictions")
        elif isinstance(artifact, pd.DataFrame):
            self.algorithm = ""
            self.frequency = "daily"
            predictions = artifact
        else:
            raise TypeError("Unsupported artifact type for predictor.")

        if not isinstance(predictions, pd.DataFrame):
            raise ValueError("Artifact must contain a pandas DataFrame under 'predictions'.")

        self.predictions = predictions.copy()
        self.predictions.columns = [str(column).strip() for column in self.predictions.columns]
        if "ds" in self.predictions.columns:
            self.predictions["ds"] = pd.to_datetime(self.predictions["ds"], errors="coerce")

    @staticmethod
    def _normalize_prediction_input(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data must not be empty")

        prepared = data.copy()
        prepared.columns = [str(column).strip() for column in prepared.columns]
        if "ds" not in prepared.columns:
            raise ValueError("Missing required columns: ['ds']")

        prepared["ds"] = pd.to_datetime(prepared["ds"], errors="coerce")
        if prepared["ds"].isna().any():
            raise ValueError("Column 'ds' contains invalid datetime values.")

        return prepared

    @classmethod
    def from_file(
        cls,
        artifact_path: str | Path = MODEL_OUTPUT_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> "Predictor":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            artifact = Prepare.read_dataframe(path)
        elif suffix in {".parquet", ".pq"}:
            artifact = Prepare.read_dataframe(path)
        else:
            with open(path, "rb") as file:
                artifact = pickle.load(file)

        return cls(artifact=artifact, logger=logger)

    def _read_data(self, data_path: str | Path) -> pd.DataFrame:
        return Prepare.load_prediction_data(data_path)


    def _match_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        requested = self._normalize_prediction_input(data)
        key_columns = [column for column in ["TSId", "ds"] if column in requested.columns and column in self.predictions.columns]

        if not key_columns:
            if len(requested) != len(self.predictions):
                raise ValueError("Prediction input must share 'ds' or 'TSId' columns with the artifact predictions.")
            return self.predictions.reset_index(drop=True).copy()

        lookup = requested[key_columns].copy()
        lookup["_request_order"] = range(len(lookup))
        matched = lookup.merge(self.predictions, on=key_columns, how="left", sort=False)
        matched = matched.sort_values("_request_order").drop(columns=["_request_order"]).reset_index(drop=True)
        return matched

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        periods: Optional[int] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        if periods is not None:
            self.logger.warning("Ignoring 'periods'; predictions are loaded from the trained artifact.")
        if include_history:
            self.logger.warning("Ignoring 'include_history'; predictions are loaded from the trained artifact.")

        if data is None:
            return self.predictions.copy()

        return self._match_predictions(data)

    def predict_from_file(
        self,
        data_path: Optional[str | Path] = TEST_DATA_PATH,
        periods: Optional[int] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        data = self._read_data(data_path) if data_path else None
        return self.predict(data=data, periods=periods, include_history=include_history)

    def save_predictions(self, predictions: pd.DataFrame, output_path: str | Path = PREDICTIONS_OUTPUT_PATH) -> None:
        ensure_directories()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".csv":
            predictions.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in [".parquet", ".pq"]:
            predictions.to_parquet(output_path, index=False)
        else:
            with open(output_path, "wb") as file:
                pickle.dump(predictions, file)

        metadata = {
            "algorithm": self.algorithm,
            "frequency": self.frequency,
            "rows": int(len(predictions)),
            "columns": list(predictions.columns),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.logger.info("Saved predictions to %s", output_path)

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run predictions from a trained artifact")
        parser.add_argument(
            "--artifact",
            default=str(MODEL_OUTPUT_PATH),
            type=str,
            help="Path to trained artifact (.pkl)",
        )
        parser.add_argument(
            "--data",
            default=str(TEST_DATA_PATH),
            type=str,
            help="Path to prediction input (.csv or .parquet)",
        )
        parser.add_argument("--periods", type=int, help="Future periods for Prophet prediction")
        parser.add_argument(
            "--include-history",
            action="store_true",
            help="Include training history for Prophet predictions",
        )
        parser.add_argument(
            "--output",
            default=str(PREDICTIONS_OUTPUT_PATH),
            type=str,
            help="Output predictions path",
        )
        return parser


def main() -> None:
    parser = Predictor.build_arg_parser()
    args = parser.parse_args()

    predictor = Predictor.from_file(args.artifact)
    predictions = predictor.predict_from_file(
        data_path=args.data,
        periods=args.periods,
        include_history=args.include_history,
    )
    predictor.save_predictions(predictions, args.output)


if __name__ == "__main__":
    main()
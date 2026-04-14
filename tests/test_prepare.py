from __future__ import annotations

import pandas as pd
import pytest

from src.core.prepare import Prepare


def test_prepare_training_data_sorts_and_normalizes_columns() -> None:
    raw = pd.DataFrame(
        {
            " TSId ": ["B", "A"],
            " ds ": ["2024-02-01", "2024-01-01"],
            " y ": ["12", "10"],
        }
    )

    prepared = Prepare.prepare_training_data(raw)

    assert list(prepared.columns) == ["TSId", "ds", "y"]
    assert prepared["TSId"].tolist() == ["A", "B"]
    assert pd.api.types.is_datetime64_any_dtype(prepared["ds"])
    assert prepared["y"].tolist() == [10, 12]


def test_prepare_prediction_data_allows_missing_target() -> None:
    raw = pd.DataFrame({"ds": ["2025-01-01", "2025-02-01"]})

    prepared = Prepare.prepare_prediction_data(raw)

    assert list(prepared.columns) == ["ds"]
    assert pd.api.types.is_datetime64_any_dtype(prepared["ds"])


def test_prepare_training_data_rejects_invalid_target_values() -> None:
    raw = pd.DataFrame({"ds": ["2024-01-01"], "y": ["not-a-number"]})

    with pytest.raises(ValueError, match="Column 'y' contains invalid numeric values"):
        Prepare.prepare_training_data(raw)


def test_read_dataframe_rejects_empty_csv(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("ds,y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Input data file is empty"):
        Prepare.read_dataframe(path)
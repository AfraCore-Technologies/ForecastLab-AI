from __future__ import annotations

import json

from src.utils.hyperparameters import Hyperparameters


def test_parse_merges_defaults_with_partial_json(tmp_path) -> None:
    path = tmp_path / "hyperparameters.json"
    path.write_text(
        json.dumps(
            {
                "frequency": "monthly",
                "xgboost": {"model_parameters": {"max_depth": 4}},
                "probabilistic_forecast": {"quantiles": [0.2, 0.8]},
            }
        ),
        encoding="utf-8",
    )

    parsed = Hyperparameters.parse(str(path))

    assert parsed.frequency == "monthly"
    assert parsed.xgboost["model_parameters"]["max_depth"] == 4
    assert parsed.xgboost["model_parameters"]["eta"] == 0.3
    assert parsed.probabilistic_forecast["quantiles"] == [0.2, 0.8]
    assert parsed.algorithm["name"] == "auto"
from __future__ import annotations

from src.core.trainer import Trainer


def make_hyperparameters() -> dict:
    return {
        "frequency": "monthly",
        "seasonality": "auto",
        "algorithm": {"name": "xgboost", "models": {"smooth": "prophet"}, "cutoffs": {}},
        "prophet": {},
        "xgboost": {
            "model_parameters": {"eta": 0.3, "max_depth": 6, "objective": "reg:quantileerror"},
            "level_method": "mean",
            "exogenous": {"numerical": [], "categorical": []},
        },
        "probabilistic_forecast": {"quantiles": []},
    }


def test_get_xgboost_kwargs_falls_back_to_square_error_without_quantiles() -> None:
    trainer = Trainer(make_hyperparameters())

    kwargs = trainer._get_xgboost_kwargs()

    assert kwargs["learning_rate"] == 0.3
    assert "eta" not in kwargs
    assert kwargs["objective"] == "reg:squarederror"
    assert "quantile_alpha" not in kwargs


def test_get_xgboost_kwargs_defaults_quantile_alpha_for_median() -> None:
    hyperparameters = make_hyperparameters()
    hyperparameters["xgboost"]["level_method"] = "median"
    trainer = Trainer(hyperparameters)

    kwargs = trainer._get_xgboost_kwargs()

    assert kwargs["objective"] == "reg:quantileerror"
    assert kwargs["quantile_alpha"] == 0.5


def test_get_xgboost_kwargs_uses_configured_quantiles() -> None:
    hyperparameters = make_hyperparameters()
    hyperparameters["probabilistic_forecast"]["quantiles"] = [0.1, 0.5, 0.9]
    trainer = Trainer(hyperparameters)

    kwargs = trainer._get_xgboost_kwargs()

    assert kwargs["objective"] == "reg:quantileerror"
    assert kwargs["quantile_alpha"] == [0.1, 0.5, 0.9]


def test_resolve_algorithm_uses_auto_mapping() -> None:
    hyperparameters = make_hyperparameters()
    hyperparameters["algorithm"] = {
        "name": "auto",
        "models": {"smooth": "prophet"},
        "cutoffs": {},
    }
    trainer = Trainer(hyperparameters)

    resolved = trainer._resolve_algorithm(data=None)

    assert resolved == "prophet"
import copy
import json
from ast import AST, literal_eval
from dataclasses import MISSING, dataclass, field, fields
from typing import Any


@dataclass
class Hyperparameters:
    """Dataclass holding the definition of the hyperparameters.

    Modify the attributes of the class and specify their data types
    to be checked while parsing the JSON file.
    """

    frequency: str = "daily"
    seasonality: str = "auto"
    prophet: dict = field(
        default_factory=lambda: {
            "uncertainty_samples": 1000,
            "interval_width": 0.8,
            "growth": "auto",
            "changepoint_prior_scale": 0.05,
            "changepoint_range": 0.8,
            "seasonality_mode": "additive",
            "seasonality_prior_scale": 10.0,
            "exogenous": "",
            "holidays_prior_scale": 10.0,
        }
    )
    xgboost: dict = field(
        default_factory=lambda: {
            "model_parameters": {
                "eta": 0.3,
                "max_depth": 6,
                "sampling_method": "uniform",
                "objective": "reg:quantileerror",
            },
            "level_method": "median",
            "exogenous": {"numerical": [], "categorical": []},
        }
    )
    probabilistic_forecast: dict = field(
        default_factory=lambda: {"quantiles": [0.1, 0.5, 0.9]}
    )
    algorithm: dict = field(
        default_factory=lambda: {
            "name": "auto",
            "models": {
                "intermittent": "multivariate",
                "smooth": "prophet",
                "erratic": "multivariate",
                "lumpy": "multivariate",
                "new": "multivariate",
            },
            "cutoffs": {"nzd": 1 / 1.32, "cv2": 0.49, "min_obs": 6},
        }
    )

    @staticmethod
    def parse(path_to_hyperparameters_file: str) -> "Hyperparameters":
        """Parses the JSON hyperparameters file.

        Args:
            path_to_hyperparameters_file (str): path to JSON file

        Returns:
            Hyperparameters: class with attributes mapping to the hyperparameters # noqa
        """

        def deep_update(default: dict, update: dict) -> dict:
            """Recursively update nested dictionaries."""
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(default.get(k), dict):
                    default[k] = deep_update(default[k], v)
                else:
                    default[k] = v
            return default

        def infer_types(**kwargs: str | AST) -> dict[str, Any]:
            """Infers the types of the dict values.

            Returns:
                Dict[str, Any]: a dictionary with the inferred values
            """
            inferred_params: dict[str, Any] = {}
            for key, value in kwargs.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    inferred_value = value  # Handle empty strings
                elif isinstance(value, str):
                    v = value.strip()
                    if v.lower() in ("true", "false"):
                        inferred_value = v.lower() == "true"
                    else:
                        try:
                            inferred_value = literal_eval(v)
                        except (ValueError, SyntaxError):
                            inferred_value = value
                else:
                    inferred_value = value  # already a proper type
                inferred_params[key] = inferred_value
            return inferred_params

        with open(path_to_hyperparameters_file) as file:
            params = json.load(file)
        inferred_params = infer_types(**params)
        # Handle nested dict defaults
        default_instance = Hyperparameters()
        merged_xgb = deep_update(
            copy.deepcopy(default_instance.xgboost),
            inferred_params.get("xgboost", {}),
        )
        inferred_params["xgboost"] = merged_xgb
        merged_probabilistic = deep_update(
            copy.deepcopy(default_instance.probabilistic_forecast),
            inferred_params.get("probabilistic_forecast", {}),
        )
        inferred_params["probabilistic_forecast"] = merged_probabilistic
        merged_algorithm = deep_update(
            copy.deepcopy(default_instance.algorithm),
            inferred_params.get("algorithm", {}),
        )
        inferred_params["algorithm"] = merged_algorithm

        merged_prophet = deep_update(
            copy.deepcopy(default_instance.prophet),
            inferred_params.get("prophet", {}),
        )
        inferred_params["prophet"] = merged_prophet
        return Hyperparameters(**inferred_params)

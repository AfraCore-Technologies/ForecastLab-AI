from typing import Any, Dict, Optional
import pickle
from pathlib import Path

import pandas as pd
from prophet import Prophet


class ProphetModel:
    """
    Thin wrapper around `prophet.Prophet` to standardize fit / predict / save / load.
    Expects training data as a DataFrame with columns ['ds', 'y'].
    """

    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        model_kwargs = model_kwargs or {}
        if seed is not None:
            model_kwargs.setdefault("random_state", seed)
        self._model = Prophet(**model_kwargs)
        self._is_fitted = False

    @property
    def model(self) -> Prophet:
        return self._model

    def fit(self, df: pd.DataFrame, **fit_kwargs) -> "ProphetModel":
        """
        Fit the internal Prophet model.
        df must contain 'ds' (datetime) and 'y' (target) columns.
        """
        if not {"ds", "y"}.issubset(df.columns):
            raise ValueError("Training DataFrame must contain 'ds' and 'y' columns.")
        # Ensure correct dtypes
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        self._model.fit(df, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(
        self,
        periods: Optional[int] = None,
        freq: str = "D",
        future_df: Optional[pd.DataFrame] = None,
        include_history: bool = True,
    ) -> pd.DataFrame:
        """
        Make forecasts.
        - If future_df is provided, it will be used directly (must contain 'ds').
        - Otherwise, a future DataFrame is created using periods and freq.
        Returns Prophet's forecast DataFrame (contains 'ds', 'yhat', 'yhat_lower', 'yhat_upper', ...).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        if future_df is None:
            if periods is None:
                raise ValueError("Either future_df or periods must be provided.")
            future = self._model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        else:
            future = future_df.copy()
            if "ds" not in future.columns:
                raise ValueError("future_df must contain 'ds' column.")
            future["ds"] = pd.to_datetime(future["ds"])

        forecast = self._model.predict(future)
        return forecast

    def save(self, path: str | Path) -> None:
        """
        Persist wrapper + internal model to disk using pickle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ProphetModel":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ProphetModel):
            raise TypeError("Loaded object is not a ProphetModel instance.")
        return obj

    def set_seed(self, seed: int) -> None:
        """
        Update the model's random seed (requires re-instantiation of Prophet).
        Existing fitted state is not preserved when changing seed.
        """
        kwargs = self._model.__dict__.get("params", {}) or {}
        # Recreate Prophet with same init kwargs where possible
        # Note: Prophet does not expose all init args; this is best effort.
        self._model = Prophet(**{**getattr(self._model, "_init_kwargs", {}), "random_state": seed})  # type: ignore
        self._is_fitted = False
"""Baseline models for bike availability forecasting.

Provides two simple models that establish a performance floor:

- :class:`NaiveBaseline` — predicts the last known value (``bikes_lag_1``).
- :class:`LinearRegressionModel` — ordinary least-squares on all features.

Both follow the same duck-typed interface: ``fit(X, y) -> self`` and
``predict(X) -> ndarray``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.dataset.features import FEATURE_COLS

logger = logging.getLogger(__name__)


class NaiveBaseline:
    """Predict the last known bike count (``bikes_lag_1``).

    This is a stateless model — ``fit`` is a no-op. It serves as the
    simplest possible baseline: "the number of bikes in 15 minutes will
    be the same as right now."
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveBaseline:
        """No-op. Returns self for API compatibility."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``bikes_lag_1`` as the prediction."""
        return X["bikes_lag_1"].values


class LinearRegressionModel:
    """Ordinary least-squares regression on all features.

    Wraps :class:`sklearn.linear_model.LinearRegression` and automatically
    subsets the input to :data:`~src.dataset.features.FEATURE_COLS`.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearRegressionModel:
        """Train on ``FEATURE_COLS``.

        Args:
            X: Feature DataFrame (may contain extra columns).
            y: Target Series.

        Returns:
            self
        """
        self._model.fit(X[FEATURE_COLS], y)
        logger.info(
            "LinearRegression trained on %d samples, %d features",
            len(X),
            len(FEATURE_COLS),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions from the trained model.

        Args:
            X: Feature DataFrame (may contain extra columns).

        Returns:
            Predictions as a 1-D numpy array.
        """
        return self._model.predict(X[FEATURE_COLS])

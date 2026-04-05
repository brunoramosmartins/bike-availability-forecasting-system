"""Advanced models for bike availability forecasting.

Provides gradient boosting models that capture non-linear patterns:

- :class:`LightGBMModel` — LightGBM regressor with optional Optuna tuning.
- :class:`XGBoostModel` — XGBoost regressor for comparison.

Both follow the same duck-typed interface as baseline models:
``fit(X, y) -> self`` and ``predict(X) -> ndarray``.

The :func:`tune_lightgbm` function runs Optuna hyperparameter search
using a held-out validation set (temporal split).
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from src.dataset.features import FEATURE_COLS

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMModel:
    """LightGBM gradient boosting regressor.

    Wraps :class:`lightgbm.LGBMRegressor` and automatically subsets
    the input to :data:`~src.dataset.features.FEATURE_COLS`.

    Parameters
    ----------
    **params
        Keyword arguments forwarded to ``LGBMRegressor``. Sensible
        defaults are applied for keys not provided.
    """

    _DEFAULTS: dict[str, object] = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
    }

    def __init__(self, **params: object) -> None:
        merged = {**self._DEFAULTS, **params}
        self._model = lgb.LGBMRegressor(**merged)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LightGBMModel:
        """Train on ``FEATURE_COLS``.

        When *X_val* and *y_val* are provided, early stopping is used
        (patience = 50 rounds).
        """
        fit_params: dict[str, object] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val[FEATURE_COLS], y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]

        self._model.fit(X[FEATURE_COLS], y, **fit_params)
        logger.info(
            "LightGBM trained — n_estimators=%d, best_iteration=%s",
            self._model.n_estimators,
            getattr(self._model, "best_iteration_", "N/A"),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions from the trained model."""
        return self._model.predict(X[FEATURE_COLS])

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame.

        Parameters
        ----------
        importance_type
            ``"gain"`` (default) or ``"split"``.
        """
        importances = self._model.booster_.feature_importance(
            importance_type=importance_type
        )
        return (
            pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class XGBoostModel:
    """XGBoost gradient boosting regressor.

    Wraps :class:`xgboost.XGBRegressor` with sensible defaults.
    Included for comparison — not separately tuned.
    """

    _DEFAULTS: dict[str, object] = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": 42,
    }

    def __init__(self, **params: object) -> None:
        merged = {**self._DEFAULTS, **params}
        self._model = xgb.XGBRegressor(**merged)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> XGBoostModel:
        """Train on ``FEATURE_COLS``."""
        fit_params: dict[str, object] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val[FEATURE_COLS], y_val)]
            fit_params["verbose"] = False

        self._model.fit(X[FEATURE_COLS], y, **fit_params)
        logger.info(
            "XGBoost trained — n_estimators=%d, best_iteration=%s",
            self._model.n_estimators,
            getattr(self._model, "best_iteration", "N/A"),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions from the trained model."""
        return self._model.predict(X[FEATURE_COLS])


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict[str, object]:
    """Run Optuna hyperparameter search for LightGBM.

    Uses *X_val* / *y_val* (temporal validation split) as the objective
    instead of cross-validation, since the splits are already time-aware.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """
    from sklearn.metrics import mean_absolute_error

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_estimators": 1000,
            "random_state": random_state,
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train[FEATURE_COLS],
            y_train,
            eval_set=[(X_val[FEATURE_COLS], y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        y_pred = model.predict(X_val[FEATURE_COLS])
        return float(mean_absolute_error(y_val, y_pred))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna finished — best MAE=%.4f in %d trials",
        study.best_value,
        n_trials,
    )
    logger.info("Best params: %s", study.best_params)

    return study.best_params

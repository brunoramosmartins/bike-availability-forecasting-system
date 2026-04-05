"""Tests for src.model.advanced — LightGBM, XGBoost, and Optuna tuning."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.dataset.features import FEATURE_COLS, TARGET_COL
from src.model.advanced import LightGBMModel, XGBoostModel, tune_lightgbm


def _make_ml_df(
    n_rows: int = 100,
    n_stations: int = 2,
) -> pd.DataFrame:
    """Create a synthetic ML-ready DataFrame for testing."""
    rng = np.random.default_rng(42)

    base = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    rows_per_station = n_rows // n_stations
    frames = []

    for sid in range(1, n_stations + 1):
        timestamps = pd.date_range(base, periods=rows_per_station, freq="15min")
        bikes = rng.integers(0, 30, size=rows_per_station).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": str(sid),
                    "timestamp": timestamps,
                    "num_bikes_available": bikes,
                    "num_docks_available": 60 - bikes,
                    "bikes_lag_1": np.roll(bikes, 1),
                    "bikes_lag_2": np.roll(bikes, 2),
                    "bikes_lag_3": np.roll(bikes, 3),
                    "bikes_lag_4": np.roll(bikes, 4),
                    "bikes_rolling_mean_1h": bikes + rng.normal(0, 1, rows_per_station),
                    "bikes_rolling_std_1h": np.abs(rng.normal(2, 1, rows_per_station)),
                    "hour": timestamps.hour,
                    "weekday": timestamps.weekday,
                    "is_weekend": (timestamps.weekday >= 5).astype(int),
                    "month": timestamps.month,
                    "capacity": 60,
                    "lat": -23.57,
                    "lon": -46.69,
                    TARGET_COL: np.roll(bikes, -1),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# LightGBMModel
# ---------------------------------------------------------------------------


class TestLightGBMModel:
    """Tests for LightGBMModel."""

    def test_fit_returns_self(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10)
        result = model.fit(df[FEATURE_COLS], df[TARGET_COL])
        assert result is model

    def test_predict_shape(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert len(preds) == len(df)

    def test_predict_returns_ndarray(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert isinstance(preds, np.ndarray)

    def test_predict_not_constant(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert np.std(preds) > 0

    def test_fit_with_validation(self) -> None:
        df = _make_ml_df(n_rows=200)
        train = df.iloc[:150]
        val = df.iloc[150:]
        model = LightGBMModel(n_estimators=100)
        model.fit(
            train[FEATURE_COLS],
            train[TARGET_COL],
            X_val=val[FEATURE_COLS],
            y_val=val[TARGET_COL],
        )
        preds = model.predict(val[FEATURE_COLS])
        assert len(preds) == len(val)

    def test_feature_importance_returns_dataframe(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        imp = model.feature_importance("gain")
        assert isinstance(imp, pd.DataFrame)
        assert set(imp.columns) == {"feature", "importance"}

    def test_feature_importance_has_all_features(self) -> None:
        df = _make_ml_df()
        model = LightGBMModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        imp = model.feature_importance("gain")
        assert set(imp["feature"]) == set(FEATURE_COLS)


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_fit_returns_self(self) -> None:
        df = _make_ml_df()
        model = XGBoostModel(n_estimators=10)
        result = model.fit(df[FEATURE_COLS], df[TARGET_COL])
        assert result is model

    def test_predict_shape(self) -> None:
        df = _make_ml_df()
        model = XGBoostModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert len(preds) == len(df)

    def test_predict_returns_ndarray(self) -> None:
        df = _make_ml_df()
        model = XGBoostModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert isinstance(preds, np.ndarray)

    def test_predict_not_constant(self) -> None:
        df = _make_ml_df()
        model = XGBoostModel(n_estimators=10).fit(df[FEATURE_COLS], df[TARGET_COL])
        preds = model.predict(df[FEATURE_COLS])
        assert np.std(preds) > 0

    def test_fit_with_validation(self) -> None:
        df = _make_ml_df(n_rows=200)
        train = df.iloc[:150]
        val = df.iloc[150:]
        model = XGBoostModel(n_estimators=100)
        model.fit(
            train[FEATURE_COLS],
            train[TARGET_COL],
            X_val=val[FEATURE_COLS],
            y_val=val[TARGET_COL],
        )
        preds = model.predict(val[FEATURE_COLS])
        assert len(preds) == len(val)


# ---------------------------------------------------------------------------
# tune_lightgbm
# ---------------------------------------------------------------------------


class TestTuneLightGBM:
    """Tests for tune_lightgbm()."""

    def test_returns_dict(self) -> None:
        df = _make_ml_df(n_rows=100)
        train = df.iloc[:70]
        val = df.iloc[70:]
        result = tune_lightgbm(
            train[FEATURE_COLS],
            train[TARGET_COL],
            val[FEATURE_COLS],
            val[TARGET_COL],
            n_trials=3,
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self) -> None:
        df = _make_ml_df(n_rows=100)
        train = df.iloc[:70]
        val = df.iloc[70:]
        result = tune_lightgbm(
            train[FEATURE_COLS],
            train[TARGET_COL],
            val[FEATURE_COLS],
            val[TARGET_COL],
            n_trials=3,
        )
        expected_keys = {
            "num_leaves",
            "learning_rate",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_best_params_can_train_model(self) -> None:
        df = _make_ml_df(n_rows=100)
        train = df.iloc[:70]
        val = df.iloc[70:]
        params = tune_lightgbm(
            train[FEATURE_COLS],
            train[TARGET_COL],
            val[FEATURE_COLS],
            val[TARGET_COL],
            n_trials=3,
        )
        model = LightGBMModel(**params)
        model.fit(train[FEATURE_COLS], train[TARGET_COL])
        preds = model.predict(val[FEATURE_COLS])
        assert len(preds) == len(val)

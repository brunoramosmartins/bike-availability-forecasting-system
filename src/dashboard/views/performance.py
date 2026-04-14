"""Page 4: Model Performance Dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data import generate_predictions, load_feature_importance
from src.dataset.features import TARGET_COL
from src.model.evaluate import compute_metrics, per_hour_metrics


def render(
    metrics: dict[str, dict[str, float]],
    test_df: pd.DataFrame,
    models_dir: Path,
) -> None:
    """Render the model performance page."""
    st.header("Model Performance")
    st.markdown("Compare all trained models on the held-out test set.")

    if not metrics:
        st.warning("No metrics found. Run `python -m src.model` to train models first.")
        return

    # Metrics comparison table
    st.subheader("Metrics Comparison")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Model"
    metrics_df.columns = [c.upper() for c in metrics_df.columns]

    st.dataframe(
        metrics_df.style.highlight_min(axis=0, subset=["MAE", "RMSE"])
        .highlight_max(axis=0, subset=["R2"])
        .format("{:.4f}"),
        use_container_width=True,
    )

    # Model selector for deep dive
    model_names = list(metrics.keys())
    selected_model = st.selectbox(
        "Select model for detailed analysis",
        model_names,
        index=model_names.index("lgbm") if "lgbm" in model_names else 0,
    )

    model_path = models_dir / f"{selected_model}.joblib"
    if not model_path.exists():
        st.warning(f"Model file not found: {model_path}")
        return

    # Generate predictions
    y_pred = generate_predictions(test_df, model_path)
    y_true = test_df[TARGET_COL].values

    model_metrics = compute_metrics(y_true, y_pred)

    # KPI row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{model_metrics['mae']:.4f}")
    with col2:
        st.metric("RMSE", f"{model_metrics['rmse']:.4f}")
    with col3:
        st.metric("R2", f"{model_metrics['r2']:.4f}")

    # Actual vs Predicted scatter
    st.subheader("Actual vs Predicted")
    sample_idx = np.random.default_rng(42).choice(
        len(y_true), size=min(5000, len(y_true)), replace=False
    )

    scatter_df = pd.DataFrame(
        {"Actual": y_true[sample_idx], "Predicted": y_pred[sample_idx]}
    )

    fig_scatter = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        opacity=0.2,
        title=f"Actual vs Predicted ({selected_model.upper()})",
    )
    max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect",
        )
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Error distribution and per-hour MAE side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Error Distribution")
        errors = y_true - y_pred
        fig_hist = px.histogram(
            x=errors,
            nbins=60,
            title="Residuals (Actual - Predicted)",
            labels={"x": "Error", "y": "Count"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("MAE by Hour of Day")
        pred_col = "__pred__"
        eval_df = test_df.copy()
        eval_df[pred_col] = y_pred

        hour_mae = per_hour_metrics(eval_df, TARGET_COL, pred_col)
        fig_hour = px.bar(
            x=hour_mae.index,
            y=hour_mae["mae"],
            title="MAE by Hour",
            labels={"x": "Hour", "y": "MAE"},
        )
        fig_hour.update_layout(height=400)
        st.plotly_chart(fig_hour, use_container_width=True)

    # Feature importance (LightGBM only)
    importance_df = load_feature_importance(models_dir)
    if not importance_df.empty:
        st.subheader("Feature Importance (LightGBM — Gain)")
        fig_imp = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="LightGBM Feature Importance",
            labels={"importance": "Importance (Gain)", "feature": "Feature"},
        )
        fig_imp.update_layout(
            height=450,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

"""Page 5: Drift Monitor — Executive, Analytical, and Diagnostic views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data import (
    compute_aggregate_drift,
    compute_feature_drift_df,
    compute_rolling_mae_series,
    generate_predictions,
)
from src.dataset.features import FEATURE_COLS, TARGET_COL


def render(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    models_dir: Path,
) -> None:
    """Render the three-layer drift monitoring page."""
    st.header("Drift Monitor")
    st.markdown(
        "Monitor data and model drift across three layers: "
        "**Executive** (high-level health), "
        "**Analytical** (feature ranking), and "
        "**Diagnostic** (per-feature deep dive)."
    )

    if train_df.empty or test_df.empty:
        st.warning("Insufficient data for drift analysis.")
        return

    # -- Model selector -------------------------------------------------------

    model_names = list(metrics.keys()) if metrics else []
    if not model_names:
        st.warning("No trained models found. Run `python -m src.model` first.")
        return

    selected_model = st.sidebar.selectbox(
        "Model (Drift)",
        model_names,
        index=model_names.index("lgbm") if "lgbm" in model_names else 0,
    )

    model_path = models_dir / f"{selected_model}.joblib"
    if not model_path.exists():
        st.warning(f"Model file not found: {model_path}")
        return

    # -- Compute drift data ---------------------------------------------------

    agg = compute_aggregate_drift(train_df, test_df)
    drift_df = compute_feature_drift_df(train_df, test_df)

    # Generate predictions for model error analysis
    y_pred = generate_predictions(test_df, model_path)
    y_true = test_df[TARGET_COL].values

    # =====================================================================
    # LAYER 1: Executive View
    # =====================================================================

    st.subheader("1. Executive View")
    st.markdown("High-level drift health and model error at a glance.")

    col1, col2, col3, col4 = st.columns(4)

    drift_pct = agg["drift_score"] * 100
    with col1:
        st.metric(
            "Drift Score",
            f"{drift_pct:.0f}%",
            help="Percentage of features flagged as drifted",
        )
    with col2:
        st.metric(
            "Features Drifted",
            f"{agg['n_drifted']} / {agg['n_features']}",
        )
    with col3:
        st.metric("Avg PSI", f"{agg['avg_psi']:.4f}")
    with col4:
        baseline_mae = metrics.get(selected_model, {}).get("mae", 0)
        current_mae = float(np.mean(np.abs(y_true - y_pred)))
        delta = current_mae - baseline_mae if baseline_mae else 0
        st.metric(
            f"Test MAE ({selected_model})",
            f"{current_mae:.4f}",
            delta=f"{delta:+.4f}" if baseline_mae else None,
            delta_color="inverse",
        )

    # Drift score gauge + rolling MAE side by side
    col_left, col_right = st.columns(2)

    with col_left:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=drift_pct,
                title={"text": "Drift Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 20], "color": "#d4edda"},
                        {"range": [20, 50], "color": "#fff3cd"},
                        {"range": [50, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        mae_series = compute_rolling_mae_series(
            pd.Series(y_true), pd.Series(y_pred), window=96
        )
        mae_plot = pd.DataFrame(
            {
                "observation": range(len(mae_series)),
                "rolling_mae": mae_series.values,
            }
        ).dropna()

        if not mae_plot.empty:
            fig_mae = px.line(
                mae_plot,
                x="observation",
                y="rolling_mae",
                title="Rolling MAE (24h window)",
                labels={"observation": "Observation #", "rolling_mae": "MAE"},
            )
            if baseline_mae:
                fig_mae.add_hline(
                    y=baseline_mae,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Baseline MAE",
                )
                fig_mae.add_hline(
                    y=baseline_mae * 1.2,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Alert (+20%)",
                )
            fig_mae.update_layout(height=300)
            st.plotly_chart(fig_mae, use_container_width=True)

    # =====================================================================
    # LAYER 2: Analytical View
    # =====================================================================

    st.subheader("2. Analytical View")
    st.markdown("Feature drift ranking — which features shifted the most?")

    if drift_df.empty:
        st.info("No feature drift data available.")
    else:
        # Color-coded bar chart
        drift_df["status"] = drift_df["drifted"].map({True: "Drifted", False: "Stable"})

        fig_rank = px.bar(
            drift_df,
            x="psi",
            y="feature",
            orientation="h",
            color="status",
            color_discrete_map={"Drifted": "#e74c3c", "Stable": "#2ecc71"},
            title="Feature Drift Ranking (PSI)",
            labels={"psi": "PSI", "feature": "Feature"},
        )
        fig_rank.add_vline(
            x=0.1,
            line_dash="dot",
            line_color="orange",
            annotation_text="Moderate (0.1)",
        )
        fig_rank.add_vline(
            x=0.2,
            line_dash="dash",
            line_color="red",
            annotation_text="Significant (0.2)",
        )
        fig_rank.update_layout(
            height=max(350, len(drift_df) * 30),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        # Summary table
        with st.expander("Drift Details Table"):
            display_df = drift_df[
                ["feature", "psi", "ks_statistic", "ks_p_value", "drifted"]
            ].copy()
            display_df.columns = [
                "Feature",
                "PSI",
                "KS Statistic",
                "KS p-value",
                "Drifted",
            ]
            st.dataframe(
                display_df.style.format(
                    {"PSI": "{:.4f}", "KS Statistic": "{:.4f}", "KS p-value": "{:.4f}"}
                ),
                use_container_width=True,
            )

    # =====================================================================
    # LAYER 3: Diagnostic View
    # =====================================================================

    st.subheader("3. Diagnostic View")
    st.markdown(
        "Deep dive into a single feature: distribution comparison and drift context."
    )

    selected_feature = st.selectbox(
        "Select feature to inspect",
        FEATURE_COLS,
        index=0,
    )

    ref_vals = train_df[selected_feature].dropna()
    cur_vals = test_df[selected_feature].dropna()

    col_dist, col_stats = st.columns([2, 1])

    with col_dist:
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=ref_vals,
                name="Reference (train)",
                opacity=0.6,
                marker_color="#3498db",
                nbinsx=50,
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=cur_vals,
                name="Current (test)",
                opacity=0.6,
                marker_color="#e74c3c",
                nbinsx=50,
            )
        )
        fig_dist.update_layout(
            barmode="overlay",
            title=f"Distribution: {selected_feature}",
            xaxis_title=selected_feature,
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_stats:
        st.markdown("**Summary Statistics**")

        # Find this feature's drift result
        feat_row = drift_df[drift_df["feature"] == selected_feature]
        if not feat_row.empty:
            row = feat_row.iloc[0]
            psi_val = row["psi"]
            ks_val = row["ks_statistic"]
            ks_p = row["ks_p_value"]
            is_drifted = row["drifted"]

            if is_drifted:
                st.error("DRIFT DETECTED")
            else:
                st.success("STABLE")

            st.metric("PSI", f"{psi_val:.4f}")
            st.metric("KS Statistic", f"{ks_val:.4f}")
            st.metric("KS p-value", f"{ks_p:.4f}")

        st.markdown("---")
        stats_df = pd.DataFrame(
            {
                "Metric": ["Mean", "Std", "Min", "Median", "Max"],
                "Reference": [
                    ref_vals.mean(),
                    ref_vals.std(),
                    ref_vals.min(),
                    ref_vals.median(),
                    ref_vals.max(),
                ],
                "Current": [
                    cur_vals.mean(),
                    cur_vals.std(),
                    cur_vals.min(),
                    cur_vals.median(),
                    cur_vals.max(),
                ],
            }
        )
        stats_df["Delta"] = stats_df["Current"] - stats_df["Reference"]
        st.dataframe(
            stats_df.style.format(
                {"Reference": "{:.2f}", "Current": "{:.2f}", "Delta": "{:+.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

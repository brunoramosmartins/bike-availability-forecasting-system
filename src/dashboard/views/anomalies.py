"""Page 6: Anomaly Detection — Stuck stations and statistical outliers."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomaly.detector import (
    analyze_anomalies,
    build_station_features,
    detect_statistical_anomalies,
    detect_stuck_stations,
)


def render(
    df: pd.DataFrame,
    station_names: dict[str, str],
) -> None:
    """Render the anomaly detection page."""
    st.header("Anomaly Detection")
    st.markdown(
        "Identify malfunctioning or unusual stations using **rule-based** "
        "(stuck detection) and **statistical** (Isolation Forest) methods."
    )

    if df.empty:
        st.warning("No data available.")
        return

    # -- Controls -------------------------------------------------------------

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        stuck_hours = st.slider(
            "Stuck threshold (hours)",
            min_value=0.5,
            max_value=12.0,
            value=2.0,
            step=0.5,
            help="Flag stations with no change for longer than this.",
        )
    with col_ctrl2:
        contamination = st.slider(
            "Isolation Forest contamination",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Expected fraction of anomalous stations.",
        )

    # -- Run detection --------------------------------------------------------

    threshold = timedelta(hours=stuck_hours)
    results = analyze_anomalies(
        df,
        stuck_threshold=threshold,
        contamination=contamination,
    )

    n_total = df["station_id"].nunique()
    n_anomalous = len(results)
    n_stuck = sum(1 for r in results if r.is_stuck)
    n_outlier = sum(1 for r in results if r.is_statistical_outlier)

    # -- KPIs -----------------------------------------------------------------

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stations", n_total)
    c2.metric(
        "Anomalous", n_anomalous, delta=f"{n_anomalous / max(n_total, 1) * 100:.1f}%"
    )
    c3.metric("Stuck (rule-based)", n_stuck)
    c4.metric("Outliers (Isolation Forest)", n_outlier)

    if n_anomalous == 0:
        st.success("No anomalies detected with current thresholds.")
        _render_fleet_overview(df, station_names)
        return

    # -- Anomaly table --------------------------------------------------------

    st.subheader("Anomalous Stations")
    anomaly_rows = []
    for r in results:
        name = station_names.get(r.station_id, r.station_id)
        anomaly_rows.append(
            {
                "Station": name,
                "Station ID": r.station_id,
                "Stuck": r.is_stuck,
                "Outlier": r.is_statistical_outlier,
                "Stuck Hours": round(r.stuck_duration_hours, 1),
                "Isolation Score": round(r.isolation_score, 4),
            }
        )

    anomaly_df = pd.DataFrame(anomaly_rows).sort_values("Stuck Hours", ascending=False)
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

    # -- Map of anomalies -----------------------------------------------------

    st.subheader("Anomaly Map")
    _render_anomaly_map(df, results, station_names)

    # -- Stuck stations detail ------------------------------------------------

    if n_stuck > 0:
        st.subheader("Stuck Stations Detail")
        stuck_results = detect_stuck_stations(df, threshold=threshold)

        stuck_rows = []
        for s in stuck_results:
            name = station_names.get(s.station_id, s.station_id)
            stuck_rows.append(
                {
                    "Station": name,
                    "Station ID": s.station_id,
                    "Duration (hours)": round(s.duration_hours, 1),
                    "Last Bikes": s.last_bikes,
                    "Is Renting": s.is_renting,
                    "Last Change": (
                        s.last_change.strftime("%Y-%m-%d %H:%M")
                        if s.last_change
                        else "Never (in window)"
                    ),
                }
            )

        stuck_df = pd.DataFrame(stuck_rows).sort_values(
            "Duration (hours)", ascending=False
        )
        st.dataframe(stuck_df, use_container_width=True, hide_index=True)

        # Bar chart of stuck durations
        fig_stuck = px.bar(
            stuck_df.head(20),
            x="Duration (hours)",
            y="Station",
            orientation="h",
            title="Top Stuck Stations by Duration",
            color="Duration (hours)",
            color_continuous_scale="Reds",
        )
        fig_stuck.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(300, len(stuck_df.head(20)) * 30),
        )
        st.plotly_chart(fig_stuck, use_container_width=True)

    # -- Isolation Forest detail ----------------------------------------------

    st.subheader("Station Activity Profile")
    _render_isolation_detail(df, station_names, contamination)


def _render_anomaly_map(
    df: pd.DataFrame,
    results: list,
    station_names: dict[str, str],
) -> None:
    """Render a map with anomalous stations highlighted."""
    anomaly_ids = {r.station_id for r in results}

    # Get one row per station with coordinates
    station_coords = (
        df.groupby("station_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )
    station_coords["status"] = station_coords["station_id"].apply(
        lambda x: "Anomalous" if x in anomaly_ids else "Normal"
    )
    station_coords["name"] = station_coords["station_id"].map(
        lambda x: station_names.get(x, x)
    )

    fig = px.scatter_mapbox(
        station_coords,
        lat="lat",
        lon="lon",
        color="status",
        color_discrete_map={"Anomalous": "#e74c3c", "Normal": "#2ecc71"},
        hover_name="name",
        hover_data={"station_id": True, "status": True, "lat": False, "lon": False},
        title="Station Map (Anomalies in Red)",
        zoom=11,
        height=500,
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)


def _render_isolation_detail(
    df: pd.DataFrame,
    station_names: dict[str, str],
    contamination: float,
) -> None:
    """Render Isolation Forest feature scatter and scores."""
    features = build_station_features(df)
    if features.empty:
        st.info("Insufficient data for station features.")
        return

    iso_df = detect_statistical_anomalies(features, contamination=contamination)

    iso_df = iso_df.reset_index()
    iso_df["name"] = iso_df["station_id"].map(lambda x: station_names.get(x, x))
    iso_df["status"] = iso_df["anomaly_label"].map({-1: "Outlier", 1: "Normal"})

    # Scatter: change_rate vs zero_pct (key activity indicators)
    if "change_rate" in iso_df.columns and "zero_pct" in iso_df.columns:
        fig = px.scatter(
            iso_df,
            x="change_rate",
            y="zero_pct",
            color="status",
            color_discrete_map={"Outlier": "#e74c3c", "Normal": "#3498db"},
            hover_name="name",
            hover_data={
                "avg_bikes": ":.1f",
                "anomaly_score": ":.3f",
                "status": True,
            },
            title="Station Activity: Change Rate vs Zero Availability %",
            labels={
                "change_rate": "Change Rate (fraction of intervals with change)",
                "zero_pct": "Zero Availability % (fraction at 0 bikes)",
            },
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly score distribution
    fig_hist = px.histogram(
        iso_df,
        x="anomaly_score",
        color="status",
        color_discrete_map={"Outlier": "#e74c3c", "Normal": "#3498db"},
        nbins=30,
        title="Isolation Forest Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score (lower = more anomalous)"},
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)


def _render_fleet_overview(
    df: pd.DataFrame,
    station_names: dict[str, str],
) -> None:
    """When no anomalies, show a brief fleet health overview."""
    st.subheader("Fleet Health Overview")

    features = build_station_features(df)
    if features.empty:
        return

    features = features.reset_index()
    features["name"] = features["station_id"].map(lambda x: station_names.get(x, x))

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            features,
            x="change_rate",
            nbins=30,
            title="Distribution of Change Rate Across Stations",
            labels={"change_rate": "Change Rate"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            features,
            x="zero_pct",
            nbins=30,
            title="Distribution of Zero-Availability % Across Stations",
            labels={"zero_pct": "Zero Availability %"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

"""Page 1: Bike Availability Over Time."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render(df: pd.DataFrame, station_names: dict[str, str]) -> None:
    """Render the availability timeline page."""
    st.header("Bike Availability Over Time")
    st.markdown(
        "Explore how bike availability changes throughout the day "
        "for selected stations."
    )

    # Station filter
    all_ids = sorted(df["station_id"].unique())
    default_ids = all_ids[:5]

    selected = st.multiselect(
        "Select stations",
        options=all_ids,
        default=default_ids,
        format_func=lambda x: station_names.get(x, x),
    )

    if not selected:
        st.info("Select at least one station to view the chart.")
        return

    filtered = df[df["station_id"].isin(selected)].copy()
    filtered["station_name"] = filtered["station_id"].map(
        lambda x: station_names.get(x, x)
    )

    # Date range
    min_ts = filtered["timestamp"].min()
    max_ts = filtered["timestamp"].max()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=min_ts.date(), min_value=min_ts.date())
    with col2:
        end = st.date_input("To", value=max_ts.date(), max_value=max_ts.date())

    mask = (filtered["timestamp"].dt.date >= start) & (
        filtered["timestamp"].dt.date <= end
    )
    filtered = filtered[mask]

    if filtered.empty:
        st.warning("No data in the selected range.")
        return

    fig = px.line(
        filtered,
        x="timestamp",
        y="num_bikes_available",
        color="station_name",
        title="Bikes Available Over Time",
        labels={
            "timestamp": "Time",
            "num_bikes_available": "Bikes Available",
            "station_name": "Station",
        },
    )
    fig.update_layout(hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    st.subheader("Summary")
    cols = st.columns(len(selected))
    for i, sid in enumerate(selected):
        station_data = filtered[filtered["station_id"] == sid]
        with cols[i]:
            name = station_names.get(sid, sid)
            st.metric(
                label=name[:25],
                value=f"{station_data['num_bikes_available'].mean():.1f}",
                delta=None,
            )

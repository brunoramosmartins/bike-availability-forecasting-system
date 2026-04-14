"""Page 2: Station Geographic Heatmap."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.data import compute_station_summary


def render(df: pd.DataFrame, station_names: dict[str, str]) -> None:
    """Render the geographic station heatmap page."""
    st.header("Station Heatmap")
    st.markdown(
        "Geographic distribution of bike availability across "
        "Sao Paulo. Marker size and color represent average availability."
    )

    # Hour filter
    hour = st.slider("Filter by hour of day", 0, 23, 12)
    hour_df = df[df["hour"] == hour]

    if hour_df.empty:
        st.warning("No data for the selected hour.")
        return

    summary = compute_station_summary(hour_df)
    summary["station_name"] = summary["station_id"].map(
        lambda x: station_names.get(x, x)
    )

    fig = px.scatter_mapbox(
        summary,
        lat="lat",
        lon="lon",
        size="avg_bikes",
        color="avg_bikes",
        color_continuous_scale="YlOrRd_r",
        size_max=20,
        hover_name="station_name",
        hover_data={
            "avg_bikes": ":.1f",
            "capacity": True,
            "fill_pct": ":.1f",
            "lat": False,
            "lon": False,
        },
        mapbox_style="open-street-map",
        zoom=12,
        center={"lat": summary["lat"].mean(), "lon": summary["lon"].mean()},
        title=f"Average Bike Availability at {hour:02d}:00",
    )
    fig.update_layout(height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stations", len(summary))
    with col2:
        st.metric("Avg Bikes/Station", f"{summary['avg_bikes'].mean():.1f}")
    with col3:
        st.metric("Avg Fill %", f"{summary['fill_pct'].mean():.1f}%")

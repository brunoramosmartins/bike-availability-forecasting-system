"""Page 3: Peak Usage Hours Analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.data import compute_hourly_availability, compute_weekday_hour_heatmap


def render(df: pd.DataFrame) -> None:
    """Render the peak usage hours page."""
    st.header("Peak Usage Hours")
    st.markdown(
        "Understand when bikes are most and least available, "
        "comparing weekday and weekend patterns."
    )

    hourly = compute_hourly_availability(df)

    # KPI row
    weekday_avg = hourly[hourly["day_type"] == "Weekday"]["avg_bikes"].mean()
    weekend_avg = hourly[hourly["day_type"] == "Weekend"]["avg_bikes"].mean()
    busiest = hourly.loc[hourly["avg_bikes"].idxmin()]
    quietest = hourly.loc[hourly["avg_bikes"].idxmax()]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Weekday Avg", f"{weekday_avg:.1f} bikes")
    with col2:
        st.metric("Weekend Avg", f"{weekend_avg:.1f} bikes")
    with col3:
        st.metric(
            "Busiest Hour",
            f"{int(busiest['hour']):02d}:00",
            delta=f"{busiest['avg_bikes']:.1f} bikes",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "Quietest Hour",
            f"{int(quietest['hour']):02d}:00",
            delta=f"{quietest['avg_bikes']:.1f} bikes",
        )

    # Grouped bar chart
    fig_bar = px.bar(
        hourly,
        x="hour",
        y="avg_bikes",
        color="day_type",
        barmode="group",
        title="Average Bike Availability by Hour",
        labels={
            "hour": "Hour of Day",
            "avg_bikes": "Avg Bikes Available",
            "day_type": "Day Type",
        },
        color_discrete_map={"Weekday": "#636EFA", "Weekend": "#EF553B"},
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Weekday x Hour heatmap
    st.subheader("Weekly Pattern")
    heatmap_data = compute_weekday_hour_heatmap(df)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig_heat = px.imshow(
        heatmap_data.values,
        x=[f"{h:02d}" for h in range(24)],
        y=day_labels[: len(heatmap_data)],
        color_continuous_scale="YlOrRd_r",
        aspect="auto",
        title="Avg Availability: Weekday x Hour",
        labels={"x": "Hour", "y": "Day", "color": "Avg Bikes"},
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

"""Streamlit dashboard for Bike Availability Forecasting.

Launch with::

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.io as pio
import streamlit as st

from src.dashboard.data import (
    PROCESSED_DIR,
    load_metrics,
    load_parquet_data,
    load_station_names,
)

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Bike Availability Forecasting",
    page_icon="\U0001f6b2",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Cached data loading -----------------------------------------------------


@st.cache_data(show_spinner="Loading data...")
def _load_data() -> pd.DataFrame:
    return load_parquet_data()


@st.cache_data(show_spinner=False)
def _load_names() -> dict[str, str]:
    return load_station_names()


@st.cache_data(show_spinner=False)
def _load_metrics_cached() -> dict:
    return load_metrics()


# -- Sidebar ------------------------------------------------------------------

st.sidebar.title("\U0001f6b2 Bike Forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Availability Timeline",
        "Station Heatmap",
        "Peak Usage Hours",
        "Model Performance",
        "Drift Monitor",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "End-to-end ML system for bike availability forecasting in Sao Paulo."
)

# -- Load data ----------------------------------------------------------------

try:
    df = _load_data()
    station_names = _load_names()
    metrics = _load_metrics_cached()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run the data pipeline first: `python -m src.dataset`")
    st.stop()

# -- Page dispatch ------------------------------------------------------------

if page == "Availability Timeline":
    from src.dashboard.views.availability import render

    render(df, station_names)

elif page == "Station Heatmap":
    from src.dashboard.views.heatmap import render

    render(df, station_names)

elif page == "Peak Usage Hours":
    from src.dashboard.views.peak_hours import render

    render(df)

elif page == "Model Performance":
    from src.dashboard.views.performance import render

    render(metrics, df[df["split"] == "test"], PROCESSED_DIR)

elif page == "Drift Monitor":
    from src.dashboard.views.drift_monitor import render

    render(
        train_df=df[df["split"] == "train"],
        test_df=df[df["split"] == "test"],
        metrics=metrics,
        models_dir=PROCESSED_DIR,
    )

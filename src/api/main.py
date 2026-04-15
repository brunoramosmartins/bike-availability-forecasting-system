"""FastAPI application for Bike Availability Forecasting.

Launch with::

    uvicorn src.api.main:app --reload

Or::

    python -m uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, startup


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model and data at startup, clean up on shutdown."""
    startup()
    yield


app = FastAPI(
    title="Bike Availability Forecasting API",
    description=(
        "Predict bike availability at Sao Paulo bike-sharing stations "
        "(t+15 min) and detect anomalous station behavior."
    ),
    version="0.9.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

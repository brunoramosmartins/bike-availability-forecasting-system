"""Time-based dataset splitting for temporal data.

Splits the feature-engineered dataset into train, validation, and test sets
using temporal ordering — no shuffling — to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10  # last 10% of the train portion


@dataclass
class DatasetSplit:
    """Container for train / validation / test DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> DatasetSplit:
    """Split a feature DataFrame by temporal order.

    The split is performed on the **global** timeline (not per-station) so
    that the test set always represents the most recent period.

    Layout::

        |--- train (excl. val) ---|--- val ---|--- test ---|
        |<----- train_frac ------>|           |
                          |<- val_frac ->|
                                              |<-- rest -->|

    Args:
        df: ML-ready DataFrame with a ``timestamp`` column.
        train_frac: Fraction of the timeline used for training (including
            the validation slice).  Default ``0.80``.
        val_frac: Fraction of the training portion used for validation
            (taken from the end of the train set).  Default ``0.10``.

    Returns:
        A :class:`DatasetSplit` with train, val, and test DataFrames.
    """
    df = df.sort_values("timestamp").copy()

    timestamps = df["timestamp"].sort_values().unique()
    n = len(timestamps)

    train_end_idx = int(n * train_frac)
    train_cutoff = timestamps[train_end_idx - 1]
    test_start = timestamps[train_end_idx]

    # Validation split: last val_frac of the train portion
    val_start_idx = int(train_end_idx * (1 - val_frac))
    val_cutoff = timestamps[val_start_idx]

    train = df[df["timestamp"] <= train_cutoff].copy()
    test = df[df["timestamp"] >= test_start].copy()

    val = train[train["timestamp"] >= val_cutoff].copy()
    train_only = train[train["timestamp"] < val_cutoff].copy()

    logger.info(
        "Split: train=%d rows [%s → %s], val=%d rows [%s → %s], test=%d rows [%s → %s]",
        len(train_only),
        timestamps[0],
        val_cutoff,
        len(val),
        val_cutoff,
        train_cutoff,
        len(test),
        test_start,
        timestamps[-1],
    )

    return DatasetSplit(train=train_only, val=val, test=test)

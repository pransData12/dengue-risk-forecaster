"""Baselines any real model must beat."""
from __future__ import annotations
import numpy as np
import pandas as pd


def predict_baseline(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Predictions on the ORIGINAL case scale."""
    if kind == "last_month":
        return np.expm1(df["cases_lag1"].to_numpy())
    if kind == "seasonal_naive":      # same month last year
        return np.expm1(df["cases_lag12"].to_numpy())
    raise ValueError(kind)

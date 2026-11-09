"""Leak-free feature engineering.

Rule: a row for month t may only use information available at the END of month t-1.
Every feature is built with shift(>=1) inside each region. Target = log1p(cases_t).
tests/test_features.py enforces the no-leakage rule.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CASE_LAGS = (1, 2, 3, 12)
RAIN_LAGS = (1, 2, 3)      # 1 month ~ 4 weeks, 2 months ~ 8 weeks (breeding + incubation delay)
TEMP_LAGS = (1, 2)
K_NEIGHBOURS = 3


def neighbour_map(df: pd.DataFrame, k: int = K_NEIGHBOURS) -> dict[str, list[str]]:
    """k nearest regions by centroid distance (static geography, no leakage)."""
    cent = df.groupby("region")[["lat", "lon"]].first()
    xy = cent.to_numpy()
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    names = cent.index.to_list()
    return {names[i]: [names[j] for j in np.argsort(d[i])[:k]] for i in range(len(names))}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "date"]).copy()
    df["y"] = np.log1p(df["cases"])
    g = df.groupby("region")
    for L in CASE_LAGS:
        df[f"cases_lag{L}"] = g["y"].shift(L)
    df["cases_roll3_lag1"] = g["y"].transform(lambda s: s.shift(1).rolling(3).mean())
    for L in RAIN_LAGS:
        df[f"rain_lag{L}"] = g["rain_mm"].shift(L)
    df["rain_roll3_lag1"] = g["rain_mm"].transform(lambda s: s.shift(1).rolling(3).sum())
    for L in TEMP_LAGS:
        df[f"temp_lag{L}"] = g["temp_c"].shift(L)
    m = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    nb = neighbour_map(df)
    wide = df.pivot(index="date", columns="region", values="cases_lag1")
    nb_mean = pd.DataFrame({r: wide[nb[r]].mean(axis=1) for r in wide.columns})
    nb_long = nb_mean.stack().rename("neighbour_cases_lag1").reset_index()
    nb_long.columns = ["date", "region", "neighbour_cases_lag1"]
    return df.merge(nb_long, on=["date", "region"], how="left")


FEATURES = ([f"cases_lag{L}" for L in CASE_LAGS] + ["cases_roll3_lag1"]
            + [f"rain_lag{L}" for L in RAIN_LAGS] + ["rain_roll3_lag1"]
            + [f"temp_lag{L}" for L in TEMP_LAGS]
            + ["month_sin", "month_cos", "neighbour_cases_lag1"])


def model_frame(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Rows with complete features and a known target."""
    return df_feat.dropna(subset=FEATURES + ["y"]).reset_index(drop=True)

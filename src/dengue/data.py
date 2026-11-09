"""Loading, validating and merging the case / climate panel."""
from __future__ import annotations
import pandas as pd
from .synthetic import make_synthetic_panel

REQUIRED = ["region", "date", "cases", "rain_mm", "temp_c", "lat", "lon"]


def load_panel(source: str) -> pd.DataFrame:
    """source = 'synthetic' or a path to a CSV with the REQUIRED columns."""
    df = make_synthetic_panel() if source == "synthetic" else pd.read_csv(source)
    return validate_panel(df)


def validate_panel(df: pd.DataFrame, min_months: int = 60, min_regions: int = 5) -> pd.DataFrame:
    """Fail loudly on bad input; this protects every later step."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    if df.duplicated(["region", "date"]).any():
        raise ValueError("Duplicate (region, date) rows. Aggregate before loading.")
    if (df["cases"].dropna() < 0).any():
        raise ValueError("Negative case counts found.")
    n_months, n_regions = df["date"].nunique(), df["region"].nunique()
    if n_months < min_months or n_regions < min_regions:
        raise ValueError(
            f"Not enough data: {n_months} months x {n_regions} regions "
            f"(need >= {min_months} months and >= {min_regions} regions). "
            "Pick a dataset with longer / wider coverage.")
    full = pd.MultiIndex.from_product(
        [sorted(df["region"].unique()),
         pd.date_range(df["date"].min(), df["date"].max(), freq="MS")],
        names=["region", "date"])
    df = df.set_index(["region", "date"]).reindex(full).reset_index()
    for c in ["lat", "lon"]:
        df[c] = df.groupby("region")[c].transform(lambda s: s.ffill().bfill())
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region data availability. Run this BEFORE committing to a dataset."""
    g = df.groupby("region")
    return pd.DataFrame({
        "first": g["date"].min(), "last": g["date"].max(), "months": g["date"].count(),
        "pct_missing_cases": g["cases"].apply(lambda s: round(100 * s.isna().mean(), 1)),
        "pct_zero_cases": g["cases"].apply(lambda s: round(100 * (s == 0).mean(), 1)),
    })

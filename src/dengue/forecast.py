"""Fit on ALL history and forecast next month, with an 80% interval, per region."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import FEATURES, build_features, model_frame
from .models import QuantileForecaster


def forecast_next_month(panel: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    nxt = panel["date"].max() + pd.offsets.MonthBegin(1)
    regions = panel[["region", "lat", "lon"]].drop_duplicates("region")
    future = regions.assign(date=nxt, cases=np.nan, rain_mm=np.nan, temp_c=np.nan)
    feat = build_features(pd.concat([panel, future], ignore_index=True))
    train = model_frame(feat)
    qf = QuantileForecaster(seed=seed).fit(train[FEATURES], train["y"])
    row = feat[feat["date"] == nxt].dropna(subset=FEATURES)
    q = qf.predict(row[FEATURES])
    out = row[["region", "lat", "lon"]].copy()
    out["forecast_month"] = nxt.date()
    out["q10"], out["q50"], out["q90"] = q[0.1], q[0.5], q[0.9]
    # risk tier relative to each region's own recent normal (trailing 36-month median)
    hist = panel.sort_values("date").groupby("region")["cases"].apply(lambda s: s.tail(36).median())
    out["recent_median"] = out["region"].map(hist).clip(lower=1)
    out["ratio_vs_normal"] = (out["q50"] / out["recent_median"]).round(2)
    out["tier"] = pd.cut(out["ratio_vs_normal"], [-np.inf, 1.15, 1.5, np.inf],
                         labels=["Normal", "Elevated", "High"]).astype(str)
    out["confident_elevated"] = out["q10"] > out["recent_median"]   # even the pessimistic-low case is above normal
    return out.sort_values("q50", ascending=False).reset_index(drop=True).round(1)

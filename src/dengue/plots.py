"""Figures: risk map, backtest chart, importance bars."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_COLOURS = {"Normal": "#4daf4a", "Elevated": "#ff7f00", "High": "#e41a1c"}


def risk_map(fc: pd.DataFrame, path, title=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    size = 60 + 700 * fc["q50"] / fc["q50"].max()
    ax.scatter(fc["lon"], fc["lat"], s=size, c=fc["tier"].map(TIER_COLOURS), alpha=.75, edgecolor="k")
    for _, r in fc.iterrows():
        ax.annotate(f"{r['q50']:.0f}\n[{r['q10']:.0f}-{r['q90']:.0f}]", (r["lon"], r["lat"]),
                    ha="center", va="center", fontsize=6)
    for t, c in TIER_COLOURS.items():
        ax.scatter([], [], c=c, label=t, edgecolor="k")
    ax.legend(title="Risk vs. region's own normal", loc="lower right")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title or f"Predicted cases, {fc['forecast_month'].iloc[0]}  (median [80% interval])")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def backtest_chart(pred: pd.DataFrame, path, region=None):
    region = region or pred.groupby("region")["cases"].sum().idxmax()
    g = pred[pred["region"] == region].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(g["date"], g["q10"], g["q90"], alpha=.25, label="80% interval")
    ax.plot(g["date"], g["cases"], "k-o", ms=3, label="actual")
    ax.plot(g["date"], g["q50"], label="GBM median")
    ax.plot(g["date"], g["seasonal_naive"], "--", label="seasonal naive")
    ax.set_title(f"Out-of-sample forecasts, {region}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def importance_bar(imp: pd.DataFrame, path, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d["mae_increase"], xerr=d["std"])
    ax.set_xlabel("Increase in MAE (log scale) when feature is shuffled")
    ax.set_title("Permutation importance on held-out future"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def folium_map(fc: pd.DataFrame, path) -> bool:
    try:
        import folium
    except ImportError:
        return False
    m = folium.Map(location=[fc["lat"].mean(), fc["lon"].mean()], zoom_start=5)
    for _, r in fc.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=6 + 20 * r["q50"] / fc["q50"].max(),
                            color=TIER_COLOURS[r["tier"]], fill=True,
                            tooltip=f"{r['region']}: {r['q50']:.0f} cases (80% PI {r['q10']:.0f}-{r['q90']:.0f}), {r['tier']}").add_to(m)
    m.save(str(path)); return True

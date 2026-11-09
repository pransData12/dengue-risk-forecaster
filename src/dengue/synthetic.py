"""Synthetic monthly dengue panel with a *known* climate -> cases mechanism.

Purpose: let the whole pipeline run end to end with no downloads, and let the
tests check that the pipeline can recover a lag structure we planted.
NEVER report results from this data as real-world performance.

Mechanism (per region r, month t):
    rain_t      ~ seasonal monsoon * noise
    temp_t      ~ seasonal + noise
    log_mu_t    = baseline + 0.9*z(rain_{t-2}) + 0.4*z(temp_{t-1})
                  + 0.45*log1p(cases_{t-1}) + 0.15*neighbour_log_cases_{t-1}
                  + year-level outbreak shock + noise
    cases_t     ~ NegativeBinomial(mean = exp(log_mu_t))
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic_panel(n_regions: int = 24, start: str = "2010-01-01",
                         n_months: int = 156, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    lats = rng.uniform(8, 28, n_regions)
    lons = rng.uniform(70, 88, n_regions)
    names = [f"Region_{i:02d}" for i in range(n_regions)]
    base = rng.normal(3.2, 0.6, n_regions)
    rain_scale = rng.uniform(0.6, 1.4, n_regions)
    d = np.hypot(lats[:, None] - lats[None, :], lons[:, None] - lons[None, :])
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    for i in range(n_regions):
        W[i, np.argsort(d[i])[:3]] = 1 / 3

    years = sorted({t.year for t in dates})
    year_shock = {y: rng.normal(0, 0.45) for y in years}

    rain = np.zeros((n_regions, n_months)); temp = np.zeros_like(rain)
    for t, dt in enumerate(dates):
        m = dt.month
        monsoon = np.exp(-0.5 * ((m - 7.5) / 1.6) ** 2)
        rain[:, t] = np.maximum(0, rain_scale * (30 + 260 * monsoon) * rng.gamma(4, 0.25, n_regions))
        temp[:, t] = 27 + 4 * np.cos(2 * np.pi * (m - 5) / 12) + rng.normal(0, 0.8, n_regions)
    z = lambda x: (x - x.mean()) / x.std()
    rz, tz = z(rain), z(temp)

    cases = np.zeros((n_regions, n_months))
    cases[:, :3] = np.exp(base)[:, None]
    for t in range(3, n_months):
        neigh = W @ np.log1p(cases[:, t - 1])
        log_mu = (base * 0.35 + 0.9 * rz[:, t - 2] + 0.4 * tz[:, t - 1]
                  + 0.45 * np.log1p(cases[:, t - 1]) + 0.15 * neigh
                  + year_shock[dates[t].year] + rng.normal(0, 0.12, n_regions))
        mu = np.exp(np.clip(log_mu, -2, 9))
        k = 12.0
        cases[:, t] = rng.negative_binomial(k, k / (k + mu))
    rows = [(names[r], dt, int(cases[r, t]), float(rain[r, t]), float(temp[r, t]),
             float(lats[r]), float(lons[r]))
            for r in range(n_regions) for t, dt in enumerate(dates)]
    return pd.DataFrame(rows, columns=["region", "date", "cases", "rain_mm", "temp_c", "lat", "lon"])

"""Statistical checks: stationarity (ADF) and whether the rainfall-lag signal holds."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

ADF_CRIT_5PCT = -2.86          # MacKinnon, constant-only case, large sample


def adf_test(x, maxlag: int = 12) -> dict:
    """Augmented Dickey-Fuller (null: unit root). Uses statsmodels if installed,
    otherwise a small numpy implementation (lag chosen by AIC, constant included)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, p, *_ = adfuller(x, maxlag=maxlag, autolag="AIC")
        return {"stat": float(stat), "pvalue": float(p), "stationary": bool(p < 0.05), "backend": "statsmodels"}
    except ImportError:
        pass
    dx = np.diff(x)
    best = None
    for k in range(0, min(maxlag, len(x) // 4) + 1):
        n = len(dx) - k
        if n < 10:
            break
        Y = dx[k:]
        cols = [np.ones(n), x[k:-1][:n]]
        cols += [dx[k - i - 1: len(dx) - i - 1] for i in range(k)]
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        sigma2 = resid @ resid / (n - X.shape[1])
        aic = n * np.log(resid @ resid / n) + 2 * X.shape[1]
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        tstat = beta[1] / np.sqrt(cov[1, 1])
        if best is None or aic < best[0]:
            best = (aic, tstat)
    stat = float(best[1])
    return {"stat": stat, "pvalue": float("nan"), "stationary": stat < ADF_CRIT_5PCT, "backend": "numpy"}


def stationarity_table(df: pd.DataFrame) -> pd.DataFrame:
    """ADF on log cases, and on the seasonally differenced series (lag-12 difference)."""
    rows = []
    for r, g in df.groupby("region"):
        s = np.log1p(g.sort_values("date")["cases"]).dropna()
        a, b = adf_test(s), adf_test(s.diff(12))
        rows.append({"region": r, "adf_stat_level": round(a["stat"], 2), "stationary_level": a["stationary"],
                     "adf_stat_seasdiff": round(b["stat"], 2), "stationary_seasdiff": b["stationary"]})
    return pd.DataFrame(rows)


def _deseason(s: pd.Series, months: pd.Series) -> pd.Series:
    return s - s.groupby(months).transform("mean")


def _autocorr1(x):
    x = np.asarray(x); x = x - x.mean()
    return float((x[1:] @ x[:-1]) / (x @ x)) if (x @ x) > 0 else 0.0


def rainfall_lag_correlation(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Does rainfall at lag k correlate with cases AFTER removing seasonality?

    Raw monthly series share a monsoon cycle, so naive correlations are inflated.
    We (1) subtract each calendar month's mean, and (2) correct the p-value for
    autocorrelation with Bretherton's effective sample size. We report, for each lag,
    the median correlation across regions and the share of regions significant at 5%.
    """
    rows = []
    for lag in range(0, max_lag + 1):
        rs, ps = [], []
        for _, g in df.groupby("region"):
            g = g.sort_values("date")
            y = _deseason(np.log1p(g["cases"]), g["date"].dt.month)
            x = _deseason(np.log1p(g["rain_mm"]).shift(lag), g["date"].dt.month)
            ok = y.notna() & x.notna()
            y, x = y[ok].to_numpy(), x[ok].to_numpy()
            n = len(y)
            if n < 24:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            a, b = _autocorr1(x), _autocorr1(y)
            n_eff = float(np.clip(n * (1 - a * b) / (1 + a * b), 4, n))
            t = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r ** 2))
            rs.append(r); ps.append(2 * stats.t.sf(abs(t), n_eff - 2))
        rows.append({"rain_lag_months": lag, "median_r": round(float(np.median(rs)), 3),
                     "share_regions_p<0.05": round(float(np.mean(np.array(ps) < 0.05)), 2),
                     "n_regions": len(rs)})
    return pd.DataFrame(rows)

"""Time-based validation (rolling origin). Never a random split."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .baselines import predict_baseline
from .features import FEATURES
from .models import QuantileForecaster, make_gbm, make_ridge


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def skill(mae_model: float, mae_ref: float) -> float:
    """1 = perfect, 0 = no better than reference, <0 = worse than reference."""
    return 1.0 - mae_model / mae_ref


def pinball(y, q_pred, q):
    d = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def rolling_origin_splits(dates: pd.Series, n_folds: int = 3, test_months: int = 12, gap: int = 0):
    """Expanding window: train on everything before the test block (+ optional gap)."""
    ud = np.sort(dates.unique())
    folds = []
    for k in range(n_folds, 0, -1):
        test_lo = len(ud) - k * test_months
        test_hi = test_lo + test_months
        folds.append((ud[:test_lo - gap], ud[test_lo:test_hi]))
    return folds


def run_backtest(mf: pd.DataFrame, n_folds: int = 3, test_months: int = 12, seed: int = 0):
    """Per-row out-of-sample predictions for every model and fold."""
    out = []
    for i, (tr_d, te_d) in enumerate(rolling_origin_splits(mf["date"], n_folds, test_months)):
        tr, te = mf[mf["date"].isin(tr_d)], mf[mf["date"].isin(te_d)]
        assert tr["date"].max() < te["date"].min(), "temporal leakage!"
        Xtr, Xte = tr[FEATURES], te[FEATURES]
        res = te[["region", "date", "cases", "lat", "lon"]].copy()
        res["fold"] = i
        res["last_month"] = predict_baseline(te, "last_month")
        res["seasonal_naive"] = predict_baseline(te, "seasonal_naive")
        res["ridge"] = np.expm1(np.maximum(make_ridge().fit(Xtr, tr["y"]).predict(Xte), 0))
        res["gbm"] = np.expm1(np.maximum(make_gbm(seed).fit(Xtr, tr["y"]).predict(Xte), 0))
        qf = QuantileForecaster(seed=seed).fit(Xtr, tr["y"])
        for q, p in qf.predict(Xte).items():
            res[f"q{int(round(q * 100)):02d}"] = p
        out.append(res)
    return pd.concat(out, ignore_index=True)


MODELS = ["last_month", "seasonal_naive", "ridge", "gbm", "q50"]


def summarise(pred: pd.DataFrame) -> pd.DataFrame:
    ref_sn = mae(pred["cases"], pred["seasonal_naive"])
    ref_lm = mae(pred["cases"], pred["last_month"])
    rows = []
    for m in MODELS:
        e = mae(pred["cases"], pred[m])
        rows.append({"model": m, "MAE": e, "RMSE": rmse(pred["cases"], pred[m]),
                     "skill_vs_seasonal_naive": skill(e, ref_sn),
                     "skill_vs_last_month": skill(e, ref_lm)})
    return pd.DataFrame(rows).round(3)


def per_fold(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, g in pred.groupby("fold"):
        best_base = min(mae(g["cases"], g["last_month"]), mae(g["cases"], g["seasonal_naive"]))
        for m in ["gbm", "ridge"]:
            e = mae(g["cases"], g[m])
            rows.append({"fold": f, "test_start": g["date"].min().date(), "model": m,
                         "MAE": round(e, 2), "skill_vs_best_baseline": round(skill(e, best_base), 3)})
    return pd.DataFrame(rows)


def interval_report(pred: pd.DataFrame) -> dict:
    inside = ((pred["cases"] >= pred["q10"]) & (pred["cases"] <= pred["q90"])).mean()
    return {"nominal_coverage": 0.80, "empirical_coverage": round(float(inside), 3),
            "mean_interval_width": round(float((pred["q90"] - pred["q10"]).mean()), 2),
            "pinball_q10": round(pinball(pred["cases"], pred["q10"], 0.1), 3),
            "pinball_q90": round(pinball(pred["cases"], pred["q90"], 0.9), 3)}


def bootstrap_skill_ci(pred: pd.DataFrame, model="gbm", ref="seasonal_naive",
                       n_boot: int = 500, seed: int = 0, block: int = 3):
    """Moving-block bootstrap over MONTHS (keeps cross-region correlation inside a
    month and short-range autocorrelation across months) -> 95% CI for the skill score."""
    rng = np.random.default_rng(seed)
    months = np.sort(pred["date"].unique())
    by_month = {m: g for m, g in pred.groupby("date")}
    n = len(months); stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        b = pd.concat([by_month[months[i]] for i in idx])
        stats.append(skill(mae(b["cases"], b[model]), mae(b["cases"], b[ref])))
    return tuple(float(x) for x in np.percentile(stats, [2.5, 97.5]).round(3))

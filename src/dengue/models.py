"""Models: regularised linear -> gradient boosting -> quantile gradient boosting."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.1, 0.5, 0.9)      # 80% prediction interval + median


def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def make_gbm(seed: int = 0):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=300, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def make_quantile_gbm(q: float, seed: int = 0):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.05, max_iter=250, max_depth=3,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


class QuantileForecaster:
    """One GBM per quantile; enforces monotone quantiles (no crossing)."""

    def __init__(self, quantiles=QUANTILES, seed: int = 0):
        self.quantiles = tuple(sorted(quantiles))
        self.models = {q: make_quantile_gbm(q, seed) for q in self.quantiles}

    def fit(self, X, y):
        for m in self.models.values():
            m.fit(X, y)
        return self

    def predict(self, X) -> dict:
        raw = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        raw = np.maximum.accumulate(raw, axis=1)
        return {q: np.expm1(np.maximum(raw[:, i], 0)) for i, q in enumerate(self.quantiles)}

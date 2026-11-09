"""Explainability: permutation importance (always) and SHAP (if installed)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .features import FEATURES
from .models import make_gbm


def permutation_table(mf: pd.DataFrame, test_months: int = 12, seed: int = 0) -> pd.DataFrame:
    """Fit on the past, measure importance on the held-out FUTURE block."""
    cut = np.sort(mf["date"].unique())[-test_months]
    tr, te = mf[mf["date"] < cut], mf[mf["date"] >= cut]
    model = make_gbm(seed).fit(tr[FEATURES], tr["y"])
    r = permutation_importance(model, te[FEATURES], te["y"], n_repeats=10, random_state=seed,
                               scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": FEATURES, "mae_increase": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def shap_summary(mf: pd.DataFrame, out_png, test_months: int = 12, seed: int = 0) -> bool:
    try:
        import shap, matplotlib.pyplot as plt
    except ImportError:
        return False
    cut = np.sort(mf["date"].unique())[-test_months]
    tr, te = mf[mf["date"] < cut], mf[mf["date"] >= cut]
    model = make_gbm(seed).fit(tr[FEATURES], tr["y"])
    bg = tr[FEATURES].sample(min(100, len(tr)), random_state=seed)
    expl = shap.Explainer(model.predict, bg)
    sv = expl(te[FEATURES].sample(min(300, len(te)), random_state=seed))
    shap.summary_plot(sv, show=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return True

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dengue import features as F, data as D
from dengue.synthetic import make_synthetic_panel


def _panel():
    return D.validate_panel(make_synthetic_panel(n_regions=8, n_months=96))


def test_no_target_leakage():
    """Perturbing the cases in month t must not change ANY feature in row t."""
    p = _panel()
    base = F.build_features(p).set_index(["region", "date"])
    p2 = p.copy()
    t = p2["date"].unique()[50]
    p2.loc[p2["date"] == t, ["cases", "rain_mm", "temp_c"]] *= 7
    pert = F.build_features(p2).set_index(["region", "date"])
    row_t = pd.Timestamp(t)
    a = base.xs(row_t, level="date")[F.FEATURES]
    b = pert.xs(row_t, level="date")[F.FEATURES]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_lags_are_calendar_lags():
    p = _panel(); f = F.build_features(p)
    r = f[f["region"] == "Region_00"].reset_index(drop=True)
    assert np.isclose(r.loc[20, "cases_lag12"], r.loc[8, "y"])
    assert np.isclose(r.loc[20, "rain_lag2"], r.loc[18, "rain_mm"])


def test_validation_rejects_short_or_duplicate():
    p = make_synthetic_panel(n_regions=6, n_months=30)
    try: D.validate_panel(p); assert False
    except ValueError: pass
    p = make_synthetic_panel(n_regions=6, n_months=80)
    try: D.validate_panel(pd.concat([p, p.iloc[:1]])); assert False
    except ValueError: pass

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dengue import evaluate as ev, features as F, data as D, stats_tests as S, forecast as FC
from dengue.synthetic import make_synthetic_panel


def test_splits_are_strictly_temporal():
    d = pd.Series(pd.date_range("2010-01-01", periods=100, freq="MS"))
    for tr, te in ev.rolling_origin_splits(d, 3, 12):
        assert tr.max() < te.min() and len(te) == 12


def test_skill_score_definition():
    assert ev.skill(5, 10) == 0.5 and ev.skill(10, 10) == 0 and ev.skill(20, 10) < 0


def test_model_beats_baselines_on_planted_signal():
    mf = F.model_frame(F.build_features(D.validate_panel(make_synthetic_panel(n_regions=12, n_months=120))))
    pred = ev.run_backtest(mf, n_folds=2, test_months=12)
    s = ev.summarise(pred).set_index("model")
    assert s.loc["gbm", "MAE"] < min(s.loc["last_month", "MAE"], s.loc["seasonal_naive", "MAE"])
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()


def test_planted_rain_lag_is_recovered():
    p = D.validate_panel(make_synthetic_panel(n_regions=12, n_months=120))
    t = S.rainfall_lag_correlation(p).set_index("rain_lag_months")
    assert t["median_r"].idxmax() == 2          # the lag we planted in the simulator


def test_adf_flags_random_walk_but_not_white_noise():
    rng = np.random.default_rng(0)
    assert S.adf_test(rng.normal(size=300))["stationary"]
    assert not S.adf_test(np.cumsum(rng.normal(size=300)))["stationary"]


def test_forecast_shape():
    p = D.validate_panel(make_synthetic_panel(n_regions=8, n_months=96))
    fc = FC.forecast_next_month(p)
    assert len(fc) == 8 and (fc["q10"] <= fc["q90"]).all()
    assert set(fc["tier"]) <= {"Normal", "Elevated", "High"}

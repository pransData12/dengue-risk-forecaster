"""End-to-end: load -> validate -> stats -> features -> backtest -> explain -> forecast -> map."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from dengue import data, evaluate as ev, explain, features, forecast, plots, stats_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", help="'synthetic' or path to panel CSV")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    panel = data.load_panel(a.data)
    data.coverage_report(panel).to_csv(out / "coverage_report.csv")
    print(f"Panel: {panel['region'].nunique()} regions x {panel['date'].nunique()} months "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})")

    # 1. statistics
    st = stats_tests.stationarity_table(panel); st.to_csv(out / "stationarity.csv", index=False)
    lagc = stats_tests.rainfall_lag_correlation(panel); lagc.to_csv(out / "rain_lag_correlation.csv", index=False)
    print(f"\nStationary (ADF, 5% level): raw {100 * st['stationary_level'].mean():.0f}% | "
          f"seasonally differenced {100 * st['stationary_seasdiff'].mean():.0f}% of regions")
    print("\nDeseasonalised rainfall-lag correlation:\n", lagc.to_string(index=False))

    # 2. features + time-based backtest
    mf = features.model_frame(features.build_features(panel))
    pred = ev.run_backtest(mf, a.folds, a.test_months); pred.to_csv(out / "backtest_predictions.csv", index=False)
    summ = ev.summarise(pred); summ.to_csv(out / "metrics.csv", index=False)
    folds = ev.per_fold(pred); folds.to_csv(out / "metrics_per_fold.csv", index=False)
    ci = ev.bootstrap_skill_ci(pred, "gbm", "seasonal_naive")
    iv = ev.interval_report(pred)
    print("\nBacktest (rolling origin, test on future only):\n", summ.to_string(index=False))
    print("\nPer fold:\n", folds.to_string(index=False))
    print(f"\nGBM skill vs seasonal naive 95% block-bootstrap CI: {ci}")
    print("Prediction interval:", iv)

    # 3. explainability
    imp = explain.permutation_table(mf); imp.to_csv(out / "permutation_importance.csv", index=False)
    plots.importance_bar(imp, out / "permutation_importance.png")
    shap_ok = explain.shap_summary(mf, out / "shap_summary.png")
    print("\nTop features:\n", imp.head(6).to_string(index=False), "\nSHAP:", "done" if shap_ok else "skipped (not installed)")

    # 4. forecast + map
    fc = forecast.forecast_next_month(panel); fc.to_csv(out / "next_month_forecast.csv", index=False)
    plots.risk_map(fc, out / "risk_map.png"); plots.backtest_chart(pred, out / "backtest_example.png")
    plots.folium_map(fc, out / "risk_map.html")
    print("\nNext-month forecast (top 5):\n", fc.head(5).to_string(index=False))

    g = summ.set_index("model")
    (out / "headline.json").write_text(json.dumps({
        "gbm_skill_vs_seasonal_naive": float(g.loc["gbm", "skill_vs_seasonal_naive"]),
        "gbm_skill_ci95": ci, "gbm_mae": float(g.loc["gbm", "MAE"]),
        "seasonal_naive_mae": float(g.loc["seasonal_naive", "MAE"]),
        "last_month_mae": float(g.loc["last_month", "MAE"]),
        "interval": iv}, indent=2))
    print("\nDone. Outputs in", out.resolve())

if __name__ == "__main__":
    main()

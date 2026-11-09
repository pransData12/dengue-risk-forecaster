"""Build data/panel.csv from REAL public data (needs internet).

  1. Download OpenDengue's national/subnational extract from https://opendengue.org
     (CSV with columns adm_0_name, adm_1_name, calendar_start_date, dengue_total, T_res, S_res).
  2. Make a centroids CSV with columns: region,lat,lon  (one row per admin-1 region;
     template: data/centroids_template.csv). Natural Earth admin-1 shapefiles give these.
  3. Run:
       python scripts/fetch_data.py --opendengue Temporal_extract.csv --country INDIA \
              --centroids data/centroids.csv --start 2010 --end 2022
  Climate comes from the NASA POWER monthly point API (no key needed).

NOTE: written against the documented OpenDengue / NASA POWER schemas, but the author could not
run it with network access in the build sandbox. Run the data-coverage report first and read it.
"""
import argparse, json, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

POWER = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR,T2M"
         "&community=AG&longitude={lon}&latitude={lat}&start={s}&end={e}&format=JSON")


def load_cases(path, country, start, end):
    df = pd.read_csv(path)
    df = df[(df["adm_0_name"].str.upper() == country.upper()) & (df["S_res"] == "Admin1")]
    df["date"] = pd.to_datetime(df["calendar_start_date"])
    df = df[(df["date"].dt.year >= start) & (df["date"].dt.year <= end)]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    # weekly -> monthly by week start date; monthly rows pass through. Partial first/last
    # weeks are attributed to the start month: a small, documented approximation.
    out = (df.groupby(["adm_1_name", "month"], as_index=False)["dengue_total"].sum()
             .rename(columns={"adm_1_name": "region", "month": "date", "dengue_total": "cases"}))
    return out


def fetch_climate(lat, lon, start, end):
    with urllib.request.urlopen(POWER.format(lat=lat, lon=lon, s=start, e=end), timeout=60) as r:
        p = json.load(r)["properties"]["parameter"]
    rows = []
    for key, rain in p["PRECTOTCORR"].items():
        if key.endswith("13"):          # annual summary row
            continue
        d = pd.Timestamp(year=int(key[:4]), month=int(key[4:]), day=1)
        rows.append((d, max(rain, 0) * d.days_in_month, p["T2M"][key]))   # mm/day -> mm/month
    return pd.DataFrame(rows, columns=["date", "rain_mm", "temp_c"]).replace(-999, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opendengue", required=True); ap.add_argument("--country", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--start", type=int, default=2010); ap.add_argument("--end", type=int, default=2022)
    ap.add_argument("--out", default="data/panel.csv")
    a = ap.parse_args()
    cases = load_cases(a.opendengue, a.country, a.start, a.end)
    cent = pd.read_csv(a.centroids)
    parts = []
    for _, r in cent.iterrows():
        c = cases[cases["region"] == r["region"]]
        if c.empty:
            print("no cases for", r["region"]); continue
        clim = fetch_climate(r["lat"], r["lon"], a.start, a.end); time.sleep(1)
        m = c.merge(clim, on="date", how="left").assign(lat=r["lat"], lon=r["lon"])
        parts.append(m)
    panel = pd.concat(parts)[["region", "date", "cases", "rain_mm", "temp_c", "lat", "lon"]]
    Path(a.out).parent.mkdir(exist_ok=True); panel.to_csv(a.out, index=False)
    print("wrote", a.out, panel.shape)

if __name__ == "__main__":
    main()

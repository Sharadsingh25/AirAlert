"""Generate a realistic SAMPLE dataset for AirAlert (3 cities, daily, 2019-2024).

Replace data/aqi_raw.csv with real data (CPCB + Open-Meteo + health data) when you have it.
Required columns:
date, city, pm25, pm10, no2, so2, co, o3, temp, humidity, wind_speed, resp_cases
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
CITIES = {"Delhi": 1.25, "Lucknow": 1.0, "Kanpur": 1.1}
dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
DIWALI = {2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04",
          2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01"}

rows = []
for city, scale in CITIES.items():
    doy = dates.dayofyear.values
    # winter peak (Nov-Jan) + stubble burning bump (mid Oct - Nov)
    season = 1 + 0.9 * np.cos(2 * np.pi * (doy - 355) / 365)
    stubble = ((dates.month == 10) & (dates.day >= 15)) | (dates.month == 11)
    base = 70 * season * scale + 55 * stubble * scale
    wind = np.clip(rng.gamma(3, 1.2, len(dates)) + 0.8 * np.cos(2 * np.pi * (doy - 150) / 365), 0.3, None)
    temp = 25 - 12 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2, len(dates))
    hum = np.clip(55 + 20 * np.sin(2 * np.pi * (doy - 120) / 365) + rng.normal(0, 8, len(dates)), 15, 100)

    # AR(1) noise gives day-to-day persistence
    noise = np.zeros(len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 18)
    pm25 = base * (1.6 / (0.6 + wind / 3)) / 1.6 + noise
    for y, d in DIWALI.items():
        idx = (dates >= pd.Timestamp(d)) & (dates <= pd.Timestamp(d) + pd.Timedelta(days=3))
        pm25[idx] += 90 * scale
    pm25 = np.clip(pm25, 8, 600)
    pm10 = pm25 * rng.uniform(1.4, 1.9, len(dates))
    no2 = 20 + 0.15 * pm25 + rng.normal(0, 6, len(dates))
    so2 = 10 + 0.04 * pm25 + rng.normal(0, 3, len(dates))
    co = 0.8 + 0.006 * pm25 + rng.normal(0, 0.2, len(dates))
    o3 = 35 + 0.5 * temp - 0.08 * pm25 + rng.normal(0, 6, len(dates))

    # respiratory cases react to PM2.5 of the previous 3 days
    lagged = pd.Series(pm25).rolling(3, min_periods=1).mean().shift(1).bfill().values
    resp = rng.poisson(np.clip(20 + 0.35 * lagged * scale, 5, None))

    df = pd.DataFrame(dict(date=dates, city=city, pm25=pm25, pm10=pm10, no2=no2, so2=so2,
                           co=co, o3=o3, temp=temp, humidity=hum, wind_speed=wind, resp_cases=resp))
    rows.append(df)

df = pd.concat(rows, ignore_index=True)

# inject sensor problems so the cleaning step has real work to do
n = len(df)
miss = rng.choice(n, int(0.02 * n), replace=False)
df.loc[miss, ["pm25", "pm10"]] = np.nan
neg = rng.choice(n, 25, replace=False)
df.loc[neg, "pm25"] = -999
spike = rng.choice(n, 15, replace=False)
df.loc[spike, "pm25"] = 5000

Path("data").mkdir(exist_ok=True)
df.to_csv("data/aqi_raw.csv", index=False)
print("Saved data/aqi_raw.csv", df.shape)

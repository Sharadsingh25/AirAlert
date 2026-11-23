"""AirAlert pipeline: clean -> EDA -> features -> forecast -> risk classification -> explain.

Run:  python pipeline.py
Outputs go to outputs/ (metrics.json, figures) and models/ (joblib files).
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, recall_score)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMA = True
except ImportError:
    HAS_SARIMA = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

OUT, MODELS = Path("outputs"), Path("models")
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
HORIZONS = list(range(1, 8))  # forecast 1..7 days ahead

# ------------------------------------------------------------------ 1. cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["city", "date"]).sort_values(["city", "date"])
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    for c in pollutants:                      # impossible readings -> NaN
        df.loc[df[c] < 0, c] = np.nan
    df.loc[df["pm25"] > 999, "pm25"] = np.nan  # sensor spikes
    df.loc[df["pm10"] > 1500, "pm10"] = np.nan
    filled = []
    for _, g in df.groupby("city"):           # align to a full daily index, then interpolate
        g = g.set_index("date").asfreq("D")
        g["city"] = g["city"].ffill().bfill()
        num = g.select_dtypes("number").columns
        g[num] = g[num].interpolate(limit_direction="both")
        filled.append(g.reset_index())
    return pd.concat(filled, ignore_index=True)


def pm25_to_aqi(pm):
    """India CPCB AQI sub-index for PM2.5 (piecewise linear)."""
    bp = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
          (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)]
    pm = np.clip(pm, 0, 500)
    aqi = np.zeros_like(pm, dtype=float)
    for lo, hi, ilo, ihi in bp:
        m = (pm >= lo) & (pm <= hi)
        aqi[m] = ilo + (pm[m] - lo) * (ihi - ilo) / (hi - lo)
    return aqi


def aqi_category(aqi):
    bins = [-1, 50, 100, 200, 300, 400, 10_000]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    return pd.cut(aqi, bins=bins, labels=labels)


# --------------------------------------------------------------------- 2. EDA
def eda(df: pd.DataFrame):
    m = df.assign(month=df["date"].dt.month).pivot_table(index="city", columns="month", values="aqi")
    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(m.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12), range(1, 13))
    ax.set_yticks(range(len(m.index)), m.index)
    ax.set_title("Average AQI by city and month")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "eda_heatmap.png", dpi=130)
    plt.close(fig)
    corr = df[["aqi", "pm10", "no2", "so2", "co", "o3", "temp", "humidity", "wind_speed", "resp_cases"]].corr()["aqi"]
    corr.round(3).to_csv(OUT / "eda_correlation_with_aqi.csv")


# ------------------------------------------------------------- 3. features
DIWALI = ["2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).copy()
    d = df["date"]
    df["month"] = d.dt.month
    df["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365)
    df["stubble_flag"] = (((d.dt.month == 10) & (d.dt.day >= 15)) | (d.dt.month == 11)).astype(int)
    dw = pd.to_datetime(DIWALI)
    df["diwali_flag"] = d.apply(lambda x: int(any(0 <= (x - t).days <= 3 for t in dw)))
    g = df.groupby("city")
    for lag in (1, 2, 3, 7):
        df[f"aqi_lag{lag}"] = g["aqi"].shift(lag)
    df["aqi_roll7"] = g["aqi"].transform(lambda s: s.rolling(7).mean())
    df["pm25_roll3"] = g["pm25"].transform(lambda s: s.rolling(3).mean())
    df["wind_roll3"] = g["wind_speed"].transform(lambda s: s.rolling(3).mean())
    df["city_code"] = df["city"].astype("category").cat.codes
    for h in HORIZONS:
        df[f"target_aqi_h{h}"] = g["aqi"].shift(-h)
    df["target_risk"] = g["resp_cases"].shift(-1)  # next-day respiratory load
    return df.dropna().reset_index(drop=True)


FEATURES = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "temp", "humidity", "wind_speed",
            "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7", "aqi_roll7", "pm25_roll3", "wind_roll3",
            "month", "doy_sin", "doy_cos", "stubble_flag", "diwali_flag", "city_code"]


def metrics(y, p):
    return {"MAE": round(mean_absolute_error(y, p), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "R2": round(r2_score(y, p), 3)}


def main():
    raw = pd.read_csv("data/aqi_raw.csv")
    df = clean(raw)
    df["aqi"] = pm25_to_aqi(df["pm25"].values)
    df.to_csv("data/aqi_clean.csv", index=False)
    eda(df)
    data = add_features(df)

    cut = data["date"].max() - pd.Timedelta(days=365)  # last year = test set (no shuffling)
    train, test = data[data["date"] <= cut], data[data["date"] > cut]
    results = {"forecast": {}, "classification": {}}

    # ---------------------------------------------------------- 4. forecasting
    for h in (1, 3, 7):
        y_tr, y_te = train[f"target_aqi_h{h}"], test[f"target_aqi_h{h}"]
        res = {"Persistence (baseline)": metrics(y_te, test["aqi"])}
        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
        rf.fit(train[FEATURES], y_tr)
        res["Random Forest"] = metrics(y_te, rf.predict(test[FEATURES]))
        if HAS_LGBM:
            lg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)
            lg.fit(train[FEATURES], y_tr)
            res["LightGBM"] = metrics(y_te, lg.predict(test[FEATURES]))
        results["forecast"][f"{h}-day ahead"] = res

    # SARIMA baseline on one city (slow on many series, so Delhi only)
    if HAS_SARIMA:
        s = df[df["city"] == "Delhi"].set_index("date")["aqi"]
        s_tr, s_te = s[s.index <= cut], s[s.index > cut][:30]
        sar = SARIMAX(s_tr[-730:], order=(2, 0, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
        results["forecast"]["SARIMA Delhi, 30-day horizon"] = metrics(s_te, sar.forecast(30))

    # one RF per horizon is what the dashboard uses (1..7 days)
    bundle = {}
    full = data
    for h in HORIZONS:
        m = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
        m.fit(full[FEATURES], full[f"target_aqi_h{h}"])
        bundle[h] = m
    joblib.dump({"models": bundle, "features": FEATURES}, MODELS / "forecast_models.joblib")

    # --------------------------------------------------- 5. risk classification
    q1, q2 = train["target_risk"].quantile([0.33, 0.66])
    risk = lambda x: pd.cut(x, [-1, q1, q2, 1e9], labels=["Low", "Medium", "High"])
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                 n_jobs=-1, random_state=0)
    clf.fit(train[FEATURES], risk(train["target_risk"]))
    pred = clf.predict(test[FEATURES])
    true = risk(test["target_risk"])
    labels = ["Low", "Medium", "High"]
    results["classification"]["health_risk"] = {
        "F1_macro": round(f1_score(true, pred, average="macro"), 3),
        "High_risk_recall": round(recall_score(true, pred, labels=["High"], average="macro"), 3),
    }
    ConfusionMatrixDisplay(confusion_matrix(true, pred, labels=labels), display_labels=labels).plot(cmap="Blues")
    plt.title("Health risk: confusion matrix (test year)")
    plt.savefig(OUT / "risk_confusion_matrix.png", dpi=130)
    plt.close()

    # AQI category from the 1-day forecast, with focus on Severe/Very Poor days
    rf1 = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
    rf1.fit(train[FEATURES], train["target_aqi_h1"])
    cat_p = aqi_category(rf1.predict(test[FEATURES]))
    cat_t = aqi_category(test["target_aqi_h1"])
    severe = ["Very Poor", "Severe"]
    results["classification"]["aqi_category"] = {
        "F1_macro": round(f1_score(cat_t, cat_p, average="macro"), 3),
        "VeryPoor_or_Severe_recall": round(
            float(((cat_p.isin(severe)) & (cat_t.isin(severe))).sum() / max(cat_t.isin(severe).sum(), 1)), 3),
    }
    joblib.dump({"model": clf, "q1": q1, "q2": q2}, MODELS / "risk_model.joblib")

    # ---------------------------------------------------------- 6. explainability
    try:
        import shap
        sample = test[FEATURES].sample(min(300, len(test)), random_state=0)
        sv = shap.TreeExplainer(bundle[1]).shap_values(sample)
        shap.summary_plot(sv, sample, show=False)
        plt.tight_layout()
        plt.savefig(OUT / "shap_summary.png", dpi=130)
        plt.close()
    except Exception:
        imp = pd.Series(bundle[1].feature_importances_, FEATURES).sort_values().tail(12)
        imp.plot.barh(title="Feature importance (1-day AQI model)")
        plt.tight_layout()
        plt.savefig(OUT / "feature_importance.png", dpi=130)
        plt.close()

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

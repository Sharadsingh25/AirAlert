"""AirAlert dashboard.  Run:  streamlit run app.py   (after python pipeline.py)"""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import FEATURES

st.set_page_config(page_title="AirAlert", page_icon="🌫️", layout="wide")
COORDS = {"Delhi": (28.61, 77.21), "Lucknow": (26.85, 80.95), "Kanpur": (26.45, 80.33)}
COLORS = [(50, "#2e9e5b", "Good"), (100, "#8bc34a", "Satisfactory"), (200, "#f2c230", "Moderate"),
          (300, "#f28c30", "Poor"), (400, "#e04a3a", "Very Poor"), (10_000, "#7a1f1f", "Severe")]
TIPS = {
    "Low": "Air is acceptable. Normal outdoor activity is fine.",
    "Medium": "Sensitive people (children, elderly, asthma) should limit long outdoor exposure. Keep a mask handy.",
    "High": "Wear an N95 mask outdoors, keep children indoors, and hospitals should keep respiratory beds ready.",
}


def band(aqi):
    return next((c, n) for lim, c, n in COLORS if aqi <= lim)


@st.cache_data
def load():
    return pd.read_csv("data/aqi_clean.csv", parse_dates=["date"])


@st.cache_resource
def models():
    return joblib.load("models/forecast_models.joblib"), joblib.load("models/risk_model.joblib")


df = load()
fc, rk = models()

st.title("🌫️ AirAlert: AQI forecast & respiratory risk")
city = st.sidebar.selectbox("City", sorted(df["city"].unique()))

# build features for the most recent day of the chosen city
hist = df[df["city"] == city].copy()
hist["city_code"] = sorted(df["city"].unique()).index(city)
g = hist.sort_values("date").copy()
d = g["date"]
g["month"] = d.dt.month
g["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365)
g["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365)
g["stubble_flag"] = (((d.dt.month == 10) & (d.dt.day >= 15)) | (d.dt.month == 11)).astype(int)
g["diwali_flag"] = 0
for lag in (1, 2, 3, 7):
    g[f"aqi_lag{lag}"] = g["aqi"].shift(lag)
g["aqi_roll7"] = g["aqi"].rolling(7).mean()
g["pm25_roll3"] = g["pm25"].rolling(3).mean()
g["wind_roll3"] = g["wind_speed"].rolling(3).mean()
latest = g.dropna(subset=FEATURES).iloc[[-1]]
last_date = latest["date"].iloc[0]

days = list(range(1, 8))
preds = [float(fc["models"][h].predict(latest[FEATURES])[0]) for h in days]
dates = [last_date + pd.Timedelta(days=h) for h in days]
risk = rk["model"].predict(latest[FEATURES])[0]

c1, c2 = st.columns([2, 1])
with c1:
    fig = go.Figure(go.Bar(x=[x.strftime("%a %d %b") for x in dates], y=preds,
                           marker_color=[band(p)[0] for p in preds],
                           text=[f"{p:.0f}" for p in preds], textposition="outside"))
    fig.update_layout(title=f"{city}: 7-day AQI forecast (from {last_date:%d %b %Y})",
                      yaxis_title="AQI", height=420)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.metric("Tomorrow's AQI", f"{preds[0]:.0f}", band(preds[0])[1])
    st.subheader(f"Health risk: {risk}")
    st.info(TIPS[risk])

st.subheader("Map")
st.map(pd.DataFrame([{"lat": a, "lon": b, "city": c} for c, (a, b) in COORDS.items()]))
st.caption("Trained on the data in data/aqi_clean.csv. Replace the sample data with real CPCB data for real use.")

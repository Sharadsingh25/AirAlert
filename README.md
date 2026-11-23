# AirAlert: AQI Forecasting & Respiratory Risk Prediction

Forecasts AQI for the next 1 to 7 days for each city, then classifies respiratory health risk (Low / Medium / High). Includes a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_data.py     # creates SAMPLE data in data/aqi_raw.csv (skip if you have real data)
python pipeline.py          # clean -> EDA -> features -> train -> evaluate -> explain
streamlit run app.py        # dashboard
```

## Using real data

Replace `data/aqi_raw.csv` with a file that has these columns, one row per city per day:

`date, city, pm25, pm10, no2, so2, co, o3, temp, humidity, wind_speed, resp_cases`

- Pollutants: CPCB (cpcb.nic.in) or the CPCB data on data.gov.in
- Weather: Open-Meteo or NASA POWER
- `resp_cases`: daily respiratory cases from a hospital or health dataset you can access

## What the pipeline does

1. **Clean**: negative readings and sensor spikes become missing, series aligned to a daily index, gaps interpolated.
2. **EDA**: city x month AQI heatmap, correlation of AQI with other variables (in `outputs/`).
3. **Features**: lags (1, 2, 3, 7 days), rolling means, wind, season (sin/cos), Diwali flag, stubble-burning flag.
4. **Forecast**: persistence baseline, Random Forest and LightGBM (if installed) for 1, 3 and 7 days ahead; SARIMA baseline if statsmodels is installed. The last 365 days are the test set (no shuffling).
5. **Classify**: health risk level and AQI category, with F1, confusion matrix and recall on the worst days.
6. **Explain**: SHAP summary plot (feature importance plot if SHAP is not installed).

## Note

The bundled sample data is synthetic, so the scores are only for checking that the code runs. Real results will differ.

from dotenv import load_dotenv
import requests
import streamlit as st

import os
load_dotenv()
BASE_URL = os.getenv("BASE_URL") or st.secrets.get("BASE_URL")
# ==========================================
# CALL HISTORICAL DATA API
# ==========================================
def get_historical_data(period="week", start_date=None, end_date=None, columns=None):
    params = {
        "period": period
    }

    if period == "custom":
        params["start_date"] = start_date
        params["end_date"] = end_date

    if columns:
        params["columns"] = columns

    res = requests.get(f"{BASE_URL}/historical_data", params=params)
    res.raise_for_status()
    return res.json()


# ==========================================
# CALL PREDICT API
# ==========================================
def get_forecast(n_days=7):
    params = {
        "n_forecast_days": n_days
    }

    res = requests.get(f"{BASE_URL}/predict", params=params)
    res.raise_for_status()
    return res.json()
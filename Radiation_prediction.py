import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import datetime
import gdown

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Radiation Predictor",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Radiation Predictor")
st.divider()

# ─────────────────────────────────────────────────────────────
# Google Drive Links (FINAL)
# ─────────────────────────────────────────────────────────────
MODEL_URL = "https://drive.google.com/uc?id=1mm3VptDC7eFaDikOHaQV6iR7mHzzubj9"
SCALER_URL = "https://drive.google.com/uc?id=1KS9M8uF_FE7V5yI6Fzz7MiaR2606UwWD"

# ─────────────────────────────────────────────────────────────
# Load model + scaler
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.pkl"):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, "model.pkl", quiet=False)

    if not os.path.exists("scaler.pkl"):
        with st.spinner("Downloading scaler..."):
            gdown.download(SCALER_URL, "scaler.pkl", quiet=False)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

# ─────────────────────────────────────────────────────────────
# COLUMN 1 — WEATHER
# ─────────────────────────────────────────────────────────────
with col1:
    st.subheader("🌡️ Weather Conditions")

    temperature = st.slider("Temperature (°F)", 20.0, 80.0, 50.0)
    pressure = st.slider("Barometric Pressure (Hg)", 25.0, 32.0, 30.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

    st.subheader("💨 Wind Conditions")

    wind_speed = st.slider("Wind Speed (mph)", 0.0, 100.0, 10.0)
    wind_direction = st.slider("Wind Direction (°)", 0.0, 360.0, 180.0)

# ─────────────────────────────────────────────────────────────
# COLUMN 2 — DATE & TIME
# ─────────────────────────────────────────────────────────────
with col2:
    st.subheader("📅 Date")

    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day of Month", 1, 31, 15)

    st.subheader("🕒 Time of Day")

    selected_time = st.time_input(
        "Select time",
        value=datetime.time(12, 0),
        step=900
    )

    hour = selected_time.hour
    minute = selected_time.minute

    st.subheader("🌅 Sunrise & Sunset")

    risehour = st.slider("Sunrise Hour", 0, 23, 6)
    sethour = st.slider("Sunset Hour", 0, 23, 18)

# ─────────────────────────────────────────────────────────────
# Feature Engineering (MUST MATCH TRAINING)
# ─────────────────────────────────────────────────────────────
def make_model_features():
    temp = np.sqrt(temperature + 1)
    pressure_t = np.sqrt(pressure + 1)
    humidity_t = np.sqrt(humidity + 1)
    speed_t = np.sqrt(wind_speed + 1)

    wind_rad = np.deg2rad(wind_direction)
    wind_combined = wind_speed * np.cos(wind_rad)

    features = np.array([
        temp,
        pressure_t,
        humidity_t,
        speed_t,
        month,
        day,
        hour,
        minute,
        risehour,
        sethour,
        wind_combined
    ], dtype=float)

    return features.reshape(1, -1)

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
X_input = make_model_features()
X_scaled = scaler.transform(X_input)

prediction = float(model.predict(X_scaled)[0])
prediction = max(0.0, prediction)

# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
with col3:
    st.subheader("🔮 Live Prediction")

    st.metric("Predicted Solar Radiation", f"{prediction:.2f} W/m²")

    if prediction < 50:
        label, color = "🌑 Very Low / Night", "#888888"
    elif prediction < 200:
        label, color = "🌤️ Low", "#f0a500"
    elif prediction < 500:
        label, color = "⛅ Moderate", "#e6c200"
    elif prediction < 900:
        label, color = "🌞 High", "#f07800"
    else:
        label, color = "☀️ Very High / Peak", "#e03000"

    st.markdown(
        f"<div style='font-size:1.3rem; font-weight:600; color:{color};'>{label}</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(prediction / 1400.0, 1.0))

    st.divider()

    with st.expander("🔍 See transformed values sent to the model"):
        df = pd.DataFrame({
            "Raw Features": X_input.flatten().round(4),
            "Scaled Features": X_scaled.flatten().round(4),
        })
        st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Solar Radiation Prediction · LightGBM / Stacking Model")
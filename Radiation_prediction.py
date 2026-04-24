import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import datetime
import gdown
import plotly.graph_objects as go

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
# Session state — history log
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts

# ─────────────────────────────────────────────────────────────
# Layout — inputs
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

    compass_options = {
        "N  (0°)":    0.0,
        "NE (45°)":  45.0,
        "E  (90°)":  90.0,
        "SE (135°)": 135.0,
        "S  (180°)": 180.0,
        "SW (225°)": 225.0,
        "W  (270°)": 270.0,
        "NW (315°)": 315.0,
    }
    compass_label = st.select_slider(
        "Wind Direction",
        options=list(compass_options.keys()),
        value="S  (180°)",
    )
    wind_direction = compass_options[compass_label]

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
# Auto-log every unique prediction to history
# ─────────────────────────────────────────────────────────────
current_record = {
    "point": len(st.session_state.history) + 1,
    "prediction_wm2": round(prediction, 2),
    "temperature_F": temperature,
    "pressure_Hg": pressure,
    "humidity_pct": humidity,
    "wind_speed_mph": wind_speed,
    "wind_direction_deg": wind_direction,
    "month": month,
    "day": day,
    "hour": hour,
    "minute": minute,
    "sunrise_hour": risehour,
    "sunset_hour": sethour,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# Append only when the prediction differs from the last logged value
if (
    not st.session_state.history
    or st.session_state.history[-1]["prediction_wm2"] != current_record["prediction_wm2"]
    or st.session_state.history[-1]["temperature_F"] != temperature
    or st.session_state.history[-1]["humidity_pct"] != humidity
):
    current_record["point"] = len(st.session_state.history) + 1
    st.session_state.history.append(current_record)

# ─────────────────────────────────────────────────────────────
# OUTPUT — col3
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
        df_debug = pd.DataFrame({
            "Raw Features": X_input.flatten().round(4),
            "Scaled Features": X_scaled.flatten().round(4),
        })
        st.dataframe(df_debug, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SCATTER PLOT SECTION
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Prediction History — Scatter Plot")

history_df = pd.DataFrame(st.session_state.history)

plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Plot X-axis**")
    x_axis_options = [
        "point", "temperature_F", "pressure_Hg", "humidity_pct",
        "wind_speed_mph", "wind_direction_deg", "month", "day",
        "hour", "sunrise_hour", "sunset_hour",
    ]
    x_axis = st.selectbox("X-axis variable", x_axis_options, index=0)

    color_by_options = ["None"] + [
        "temperature_F", "humidity_pct", "wind_speed_mph",
        "month", "hour", "pressure_Hg",
    ]
    color_by = st.selectbox("Color points by", color_by_options, index=1)

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

    # ── CSV download ──────────────────────────────────────────
    if not history_df.empty:
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="solar_radiation_predictions.csv",
            mime="text/csv",
        )

with plot_col:
    if history_df.empty:
        st.info("Adjust any slider to start logging predictions here.")
    else:
        if color_by != "None":
            color_vals = history_df[color_by]
            color_kwargs = dict(
                color=color_vals,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title=color_by),
                size=10,
                line=dict(width=0.5, color="white"),
            )
        else:
            color_kwargs = dict(
                color="#f07800",
                size=10,
                line=dict(width=0.5, color="white"),
            )

        fig = go.Figure()

        # connecting line
        fig.add_trace(go.Scatter(
            x=history_df[x_axis],
            y=history_df["prediction_wm2"],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        # scatter points
        fig.add_trace(go.Scatter(
            x=history_df[x_axis],
            y=history_df["prediction_wm2"],
            mode="markers",
            marker=color_kwargs,
            text=history_df.apply(
                lambda r: (
                    f"Point #{int(r['point'])}<br>"
                    f"Prediction: {r['prediction_wm2']:.1f} W/m²<br>"
                    f"Temp: {r['temperature_F']}°F  Humidity: {r['humidity_pct']}%<br>"
                    f"Wind: {r['wind_speed_mph']} mph @ {r['wind_direction_deg']}°<br>"
                    f"Time: {int(r['hour']):02d}:{int(r['minute']):02d}  Month/Day: {int(r['month'])}/{int(r['day'])}"
                ),
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        # highlight current point
        last = history_df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last[x_axis]],
            y=[last["prediction_wm2"]],
            mode="markers",
            marker=dict(color="#00ffcc", size=14, symbol="star",
                        line=dict(color="white", width=1)),
            name="Current",
            hovertemplate=f"Current: {last['prediction_wm2']:.1f} W/m²<extra></extra>",
        ))

        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            xaxis=dict(
                title=x_axis.replace("_", " ").title(),
                gridcolor="#2a2a3a",
                zeroline=False,
            ),
            yaxis=dict(
                title="Predicted Solar Radiation (W/m²)",
                gridcolor="#2a2a3a",
                zeroline=False,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="#444",
                borderwidth=1,
            ),
            margin=dict(l=60, r=20, t=20, b=50),
            height=420,
        )

        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Data table (collapsible)
# ─────────────────────────────────────────────────────────────
if not history_df.empty:
    with st.expander("📋 View full history table"):
        st.dataframe(
            history_df.sort_values("point", ascending=False),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
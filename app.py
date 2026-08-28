"""
app.py  -  WaterIntel: Water Quality Forecast & Advisory Dashboard
Run with:  streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WaterIntel",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = dict(conductivity=5000, water_temp=22.0, dayofweek=2, month=6, hour=12)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Preset definitions ────────────────────────────────────────────────────────
PRESETS = {
    "Custom Manual Input": None,
    "Normal Baseline (Safe)":          dict(conductivity=5000,  water_temp=22.0, month=6,  hour=12),
    "Post-Storm Silt Surge (Caution)": dict(conductivity=45000, water_temp=24.0, month=2,  hour=8),
    "Summer Heatwave (Warning)":       dict(conductivity=12000, water_temp=29.5, month=1,  hour=14),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Monitoring Station")
_STATION_OPTIONS = [
    "Johnstone River, Coquette Point, QLD, Australia",
    "Thane Creek, Maharashtra, India",
    "Ulhas River, Maharashtra, India",
    "Custom Location...",
]
station_choice = st.sidebar.selectbox(
    "Select Station",
    options=_STATION_OPTIONS,
    index=0,
)

# Resolve effective station name + optional context
if station_choice == "Custom Location...":
    station = st.sidebar.text_input(
        "Custom Location Name",
        placeholder="e.g. Thane Creek, Maharashtra",
        help="This name will be referenced in the AI advisory directive.",
    )
    custom_context = st.sidebar.text_area(
        "Custom Context / Regional Notes (Optional)",
        placeholder="Enter specific local conditions, water body type, or notes...",
        height=100,
    )
    if not station:
        station = "Custom Location"   # safe fallback if field left blank
else:
    station        = station_choice
    custom_context = ""

st.sidebar.markdown("---")
st.sidebar.markdown("### Scenario Preset")
preset_choice = st.sidebar.selectbox(
    "Load Scenario Preset",
    options=list(PRESETS.keys()),
    index=0,
)

# Apply preset values to session state on selection (skip Custom)
if preset_choice != "Custom Manual Input":
    for k, v in PRESETS[preset_choice].items():
        st.session_state[k] = v

st.sidebar.markdown("---")
st.sidebar.markdown("### Sensor Inputs")

conductivity = st.sidebar.slider(
    "Conductivity (uS/cm)", 0, 60000, step=100,
    value=int(st.session_state["conductivity"]),
    key="conductivity",
)
water_temp = st.sidebar.slider(
    "Water Temperature (C)", 10.0, 35.0, step=0.5,
    value=float(st.session_state["water_temp"]),
    key="water_temp",
)
dayofweek = st.sidebar.selectbox(
    "Day of Week", [0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
    index=int(st.session_state["dayofweek"]),
    key="dayofweek",
)
month = st.sidebar.slider(
    "Month", 1, 12, step=1,
    value=int(st.session_state["month"]),
    key="month",
)
hour = st.sidebar.slider(
    "Hour of Day", 0, 23, step=1,
    value=int(st.session_state["hour"]),
    key="hour",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Appearance")
glass_alpha = st.sidebar.slider(
    "Glass Transparency",
    min_value=0.0, max_value=1.0,
    value=0.12, step=0.01,
)

st.sidebar.markdown("---")
api_key = st.sidebar.text_input(
    "Gemini API Key", type="password",
    help="Required for the AI advisory. Get one at aistudio.google.com",
)
st.sidebar.caption("Adjust sliders or pick a preset, then click Forecast.")

# ── Dynamic CSS injection ─────────────────────────────────────────────────────
# glass_alpha is already a Python float at this point — no Jinja, no template engine.
# We f-string it directly into the CSS so Streamlit's rerun cycle always stays
# in sync with the slider without a full page reload.

glass_css = f"""
<style>
/* ── Base: vibrant mesh gradient (ocean-to-coral spectrum, no slate/violet) ── */
.stApp {{
    background:
        radial-gradient(ellipse at 0% 0%,   #00c6fb 0%,  transparent 55%),
        radial-gradient(ellipse at 100% 0%,  #005bea 0%,  transparent 55%),
        radial-gradient(ellipse at 0% 100%,  #f7971e 0%,  transparent 55%),
        radial-gradient(ellipse at 100% 100%,#f64f59 0%,  transparent 55%),
        radial-gradient(ellipse at 50% 50%,  #12c2e9 0%,  transparent 60%);
    background-color: #0a2540;
    min-height: 100vh;
}}

/* ── Shared glass token ───────────────────────────────────────────────────── */
.glass {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
}}

/* ── Main content block ───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main > .block-container {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    padding: 32px 32px 32px 32px !important;  /* 8pt grid: 32px = 4 units */
    margin-top: 24px !important;
    margin-bottom: 24px !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15) !important;
    border-radius: 0 16px 16px 0 !important;
}}

[data-testid="stSidebar"] > div:first-child {{
    background: transparent !important;
    padding: 24px 20px 32px 20px !important;
}}

[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {{
    background: transparent !important;
    padding: 0 !important;
}}

/* Sidebar text legibility over glass */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.55) !important;
}}

/* Sidebar slider track / thumb legibility */
[data-testid="stSidebar"] [data-testid="stSlider"] > div {{
    padding: 8px 0 !important;
    margin-bottom: 16px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}


/* ── Metric cards (st.metric) ────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    padding: 16px 24px !important;
}}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    padding: 8px 16px !important;
    margin-top: 16px !important;
}}

/* ── Input widgets (sliders, text inputs, selectbox) ─────────────────────── */
[data-testid="stTextInput"] > div,
[data-testid="stSelectbox"] > div,
.stSlider > div {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(12px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(12px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    padding: 8px !important;
    margin-bottom: 8px !important;
}}

/* ── Alert / info / warning / error boxes ────────────────────────────────── */
[data-testid="stAlert"] {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.20) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}}

/* ── Typography: WCAG-compliant text-shadow over glass ───────────────────── */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp .stMarkdown,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.55) !important;
}}

/* ── Subheaders ──────────────────────────────────────────────────────────── */
.stApp h2 {{
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 24px !important;
    margin-bottom: 16px !important;
}}

/* ── Custom metric-box cards (used for predicted values) ─────────────────── */
.metric-box {{
    background: rgba(255, 255, 255, {glass_alpha}) !important;
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.20);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border-radius: 16px;
    padding: 24px 24px;
    margin-bottom: 16px;
}}
.metric-box .m-label {{
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.70) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.50);
    margin-bottom: 8px;
}}
.metric-box .m-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.55);
    line-height: 1.1;
}}
.metric-box .m-unit {{
    font-size: 0.9rem;
    font-weight: 400;
    opacity: 0.75;
    margin-left: 4px;
}}

/* ── Primary button ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {{
    background: rgba(255, 255, 255, 0.18) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.30) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 16px 24px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.50) !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    background: rgba(255, 255, 255, 0.28) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18) !important;
}}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {{
    border-color: rgba(255,255,255,0.15) !important;
    margin: 24px 0 !important;
}}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: rgba(255,255,255,0.25);
    border-radius: 8px;
}}
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <h1 style="margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.02em;
             color:#ffffff; text-shadow:0 1px 2px rgba(0,0,0,0.55);">
    WaterIntel
  </h1>
  <p style="margin:8px 0 0 0; font-size:0.88rem; color:rgba(255,255,255,0.70);
            text-shadow:0 1px 2px rgba(0,0,0,0.50);">
    Johnstone River &middot; Coquette Point &mdash; Water Quality Forecast &amp; Advisory
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models ...")
def load_models():
    if not os.path.exists("model_turbidity.pkl") or not os.path.exists("model_no3.pkl"):
        st.error("Model files not found. Please run: python train_model.py")
        st.stop()
    return (
        joblib.load("model_turbidity.pkl"),
        joblib.load("model_no3.pkl"),
    )

model_turbidity, model_no3 = load_models()

# ── Live sensor reading cards ─────────────────────────────────────────────────
st.markdown("#### Current Sensor Configuration")
col1, col2, col3 = st.columns(3)
col1.metric("Conductivity",  f"{conductivity:,} uS/cm")
col2.metric("Water Temp",    f"{water_temp:.1f} C")
col3.metric("Hour / Month",  f"{hour:02d}h  M{month:02d}")

st.divider()

# ── Forecast button ───────────────────────────────────────────────────────────
if st.button("Generate Water Quality Forecast & Advisory",
             use_container_width=True, type="primary"):

    feature_vector = np.array([[conductivity, water_temp, dayofweek, month, hour]])

    with st.spinner("Running ML inference ..."):
        turbidity_pred = float(model_turbidity.predict(feature_vector)[0])
        no3_pred       = float(model_no3.predict(feature_vector)[0])

    # ── Predicted metric cards ────────────────────────────────────────────────
    st.markdown("#### Predicted Water Quality Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="m-label">Turbidity</div>
          <div class="m-value">{turbidity_pred:.2f}<span class="m-unit">NTU</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
          <div class="m-label">Nitrate (NO&#8323;)</div>
          <div class="m-value">{no3_pred:.2f}<span class="m-unit">mg/L</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Offline fallback ──────────────────────────────────────────────────────
    def get_offline_advisory(turbidity, no3, temp, location):
        """Pure-Python deterministic fallback — mirrors Gemini JSON schema."""
        def _no3_status(v):
            return "SAFE" if v < 5 else ("MONITOR" if v <= 10 else "HAZARD")
        def _turb_status(v):
            return "CLEAR" if v < 5 else ("CAUTION" if v <= 50 else "DANGER")
        def _temp_status(v):
            return "COLD_WARNING" if v < 10.0 else ("NORMAL" if v <= 28.0 else "HEAT_WARNING")

        SEVERITY = {
            "CLEAR": 0, "SAFE": 0, "NORMAL": 0,
            "MONITOR": 1, "CAUTION": 1, "COLD_WARNING": 1, "HEAT_WARNING": 1,
            "HAZARD": 2, "DANGER": 2,
        }
        s_no3  = _no3_status(no3)
        s_turb = _turb_status(turbidity)
        s_temp = _temp_status(temp)

        worst = max([s_no3, s_turb, s_temp], key=lambda s: SEVERITY.get(s, 0))

        DIRECTIVE = {
            0: f"Water quality at {location} is within safe limits — no immediate action required.",
            1: f"Elevated readings detected at {location} — increased monitoring is advised.",
            2: f"HAZARDOUS conditions detected at {location} — restrict water contact and notify authorities immediately.",
        }
        return {
            "overall_status": worst,
            "metrics_evaluated": {
                "no3":       {"status": s_no3,  "value": round(no3, 2)},
                "turbidity": {"status": s_turb, "value": round(turbidity, 2)},
                "water_temp":{"status": s_temp, "value": round(temp, 1)},
            },
            "public_directive": DIRECTIVE[SEVERITY.get(worst, 0)],
        }

    # ── Gemini advisory ───────────────────────────────────────────────────────
    st.markdown("#### AI Advisory")

    if not api_key:
        st.warning("Enter your Gemini API Key in the sidebar to enable AI advisory.")
    else:
        _context_clause = (
            f" Additional regional context: {custom_context}."
            if custom_context.strip() else ""
        )
        SYSTEM_PROMPT = (
            "You are an automated, deterministic data-interpretation API. "
            "Evaluate the input metrics against these exact predefined thresholds — "
            "NO3 (mg/L): <5 = SAFE, 5-10 = MONITOR, >10 = HAZARD; "
            "Turbidity (NTU): <5 = CLEAR, 5-50 = CAUTION, >50 = DANGER; "
            "WaterTemp (C): <10.0 = COLD_WARNING, 10.0-28.0 = NORMAL, >28.0 = HEAT_WARNING. "
            "Determine the single worst overall_status across all three metrics. "
            f"The monitoring station is: {station}.{_context_clause} "
            "Include the station name naturally in your public_directive sentence. "
            "Output STRICTLY valid JSON with exactly three keys: "
            "overall_status (string), "
            "metrics_evaluated (object with keys no3, turbidity, water_temp each having status and value), "
            "and public_directive (single sentence, plain English, no markdown). "
            "Do not output code fences, pleasantries, or any text outside the JSON object."
        )
        user_message = (
            f"Station: {station} | "
            f"Turbidity={turbidity_pred:.2f} NTU, "
            f"NO3={no3_pred:.2f} mg/L, "
            f"WaterTemp={water_temp:.1f} C"
            + (f" | Context: {custom_context}" if custom_context.strip() else "")
        )

        advisory     = None
        is_offline   = False

        with st.spinner("Consulting Gemini ..."):
            try:
                client   = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                raw_text = response.text.strip()

                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()

                advisory = json.loads(raw_text)

            except json.JSONDecodeError as e:
                st.error(f"Could not parse Gemini response as JSON: {e}")
                st.code(raw_text, language="text")
            except Exception:
                is_offline = True
                advisory   = get_offline_advisory(
                    turbidity_pred, no3_pred, water_temp, station
                )

        if is_offline:
            st.warning(
                "Network offline. Running local deterministic fallback advisory."
            )

        if advisory:
            overall   = advisory.get("overall_status", "UNKNOWN").upper()
            directive = advisory.get("public_directive", "No directive available.")
            metrics   = advisory.get("metrics_evaluated", {})

            BANNER_PALETTE = {
                "SAFE":         ("#10B981", "#064e3b", "Safe"),
                "CLEAR":        ("#10B981", "#064e3b", "Clear"),
                "NORMAL":       ("#10B981", "#064e3b", "Normal"),
                "MONITOR":      ("#F59E0B", "#451a03", "Monitor"),
                "CAUTION":      ("#F59E0B", "#451a03", "Caution"),
                "COLD_WARNING": ("#F59E0B", "#451a03", "Cold Warning"),
                "HEAT_WARNING": ("#F59E0B", "#451a03", "Heat Warning"),
                "HAZARD":       ("#EF4444", "#450a0a", "Hazard"),
                "DANGER":       ("#EF4444", "#450a0a", "Danger"),
            }
            accent, dark, label = BANNER_PALETTE.get(
                overall, ("#6B7280", "#111827", overall.title())
            )

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,
                    {accent}28 0%, {accent}12 100%);
                border: 1px solid {accent}66;
                border-left: 4px solid {accent};
                border-radius: 16px;
                padding: 20px 24px;
                margin: 8px 0 16px 0;
                backdrop-filter: blur(16px) saturate(180%);
                -webkit-backdrop-filter: blur(16px) saturate(180%);
                box-shadow: 0 4px 24px {accent}22;
            ">
              <div style="
                font-size: 0.72rem; font-weight: 600; letter-spacing: 0.10em;
                text-transform: uppercase; color: {accent};
                text-shadow: 0 1px 2px rgba(0,0,0,0.4);
                margin-bottom: 6px;
              ">Overall Status</div>
              <div style="
                font-size: 1.6rem; font-weight: 700; color: #ffffff;
                text-shadow: 0 1px 3px rgba(0,0,0,0.55);
                margin-bottom: 10px;
              ">{label}</div>
              <div style="
                font-size: 0.88rem; color: rgba(255,255,255,0.85);
                text-shadow: 0 1px 2px rgba(0,0,0,0.45);
                line-height: 1.5;
                border-top: 1px solid rgba(255,255,255,0.12);
                padding-top: 10px; margin-top: 4px;
              ">{directive}</div>
            </div>
            """, unsafe_allow_html=True)

            if metrics:
                with st.expander("Detailed Metric Evaluation"):
                    st.json(metrics)

    # ── Threshold reference ───────────────────────────────────────────────────
    with st.expander("Threshold Reference"):
        st.markdown("""
        | Parameter | Safe / Normal | Monitor / Caution / Warning | Hazard / Danger |
        |-----------|---------------|-----------------------------|-----------------|
        | **NO&#8323;** (mg/L) | < 5 &mdash; SAFE | 5 &ndash; 10 &mdash; MONITOR | > 10 &mdash; HAZARD |
        | **Turbidity** (NTU) | < 5 &mdash; CLEAR | 5 &ndash; 50 &mdash; CAUTION | > 50 &mdash; DANGER |
        | **Water Temp** (&#176;C) | 10.0 &ndash; 28.0 &mdash; NORMAL | &mdash; | < 10.0 COLD&#8202;WARNING &nbsp;&nbsp; > 28.0 HEAT&#8202;WARNING |
        """)

URL : https://waterintel-tuwzegwbjacqsdnhhgswmk.streamlit.app/
# WaterIntel 🌊💧
*Real-Time Water Quality Intelligence, ML Prediction, & Gemini-Powered Deterministic Safety Adjudication*

WaterIntel is an advanced, production-grade telemetry forecasting engine designed for environmental monitoring. It combines classical machine learning with structured Large Language Model (LLM) reasoning to evaluate river health, detect ecological anomalies, and issue public safety directives instantly.

---

## 🚀 Key Features
* **Liquid Glass UI:** Custom Apple-inspired glassmorphism design featuring dynamic transparency controls, frosted glass cards, and a responsive layout.
* **Hybrid Architecture:** Decouples numerical regression from semantic reasoning:
  * **Scikit-Learn (`RandomForestRegressor`):** Mathematically predicts live `Turbidity` and `Nitrate (NO3)` based on environmental sensor sliders.
  * **Google Gemini API:** Acts as a deterministic reasoning gate, translating numeric outputs into strict JSON safety advisories.
* **Fault-Tolerant Offline Fallback:** Includes a local Python fallback engine that automatically engages via a `try/except` block if network connectivity drops.
* **Geographic Localization & Custom Input:** Supports regional monitoring stations alongside custom water body inputs and user-defined text data notes.
* **Instant Demo Presets:** Pre-configured simulation scenarios for rapid testing.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Styling:** Custom CSS, Backdrop-Filter Glassmorphism
* **Machine Learning:** Pandas, Scikit-Learn
* **AI / Orchestration:** Google GenAI SDK

---

## ⚙️ Installation & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ManavGopale27/WaterIntel.git](https://github.com/ManavGopale27/WaterIntel.git)
   cd WaterIntel'''

  pip install -r requirements.txt
  streamlit run app.py
  ### 2. `requirements.txt`

streamlit>=1.30.0
pandas>=2.0.0
scikit-learn>=1.2.0
google-genai>=0.1.0
python-dotenv>=1.0.0
joblib>=1.2.0

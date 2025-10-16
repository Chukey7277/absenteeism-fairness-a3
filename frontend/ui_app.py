import streamlit as st
import requests

# =========================
# 1️⃣  Basic setup
# =========================
st.set_page_config(
    page_title="Absenteeism Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "https://absenteeism-fairness-a3.onrender.com"  # 👈 Make sure backend is running here

st.title("📉 Absenteeism Risk — HR Support Dashboard")
st.markdown(
    "Empowering HR teams to identify and support employees "
    "**at risk of absenteeism**, fairly and transparently."
)
st.divider()

# =========================
# 2️⃣  Sidebar inputs
# =========================
st.sidebar.header("🧾 Input Employee Features")

reason = st.sidebar.number_input("Reason for absence (code)", min_value=1, max_value=28, value=1)
month = st.sidebar.selectbox("Month of absence", list(range(1, 13)), index=4)
dow = st.sidebar.selectbox("Day of the week", list(range(2, 7)), index=2)
season = st.sidebar.selectbox("Season", [1, 2, 3, 4], index=0)
edu = st.sidebar.selectbox("Education", [1, 2, 3, 4])
discipline = st.sidebar.selectbox("Disciplinary failure", [0, 1])
drinker = st.sidebar.selectbox("Social drinker", [0, 1])
smoker = st.sidebar.selectbox("Social smoker", [0, 1])
service_time = st.sidebar.slider("Service time (years)", 0, 30, 8)
age = st.sidebar.slider("Age", 18, 60, 35)
bmi = st.sidebar.slider("BMI", 15.0, 35.0, 24.5)

features = {
    "Reason for absence": reason,
    "Month of absence": month,
    "Day of week": dow,
    "Seasons": season,
    "Education": edu,
    "Disciplinary failure": discipline,
    "Social drinker": drinker,
    "Social smoker": smoker,
    "Service time": service_time,
    "Age": age,
    "BMI": bmi,
}
# Add missing numeric columns expected by model (default safe values)
features.update({
    "Day of the week": dow,
    "Transportation expense": 200,   # adjust typical mean
    "Distance from Residence to Work": 10,
    "Work load Average/day": 250,
    "Hit target": 95,
    "Son": 1,
    "Pet": 0,
    "Weight": 70,
    "Height": 175,
    "Body mass index": bmi,
})


# =========================
# 3️⃣  Prediction area
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔍 Predict Absenteeism Risk"):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json={"features": features})
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Predicted Risk: {result['proba']:.2%}**")
                st.metric("Predicted Label", result["label"])
                st.metric("Decision Threshold", result["threshold"])
                st.write("**Top contributing factors:**")
                for e in result["explanations"]:
                    st.markdown(f"- {e}")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Request error: {e}")

with col2:
    st.info("ℹ️ Adjust inputs in the sidebar and click *Predict* to see updated results.")

st.divider()

# =========================
# 4️⃣  Model Info & Metrics
# =========================
st.subheader("📘 Model Information & Fairness Metrics")
col1, col2 = st.columns(2)

with col1:
    try:
        info = requests.get(f"{BACKEND_URL}/model-info").json()
        st.markdown(f"### 🧠 {info.get('purpose', '')}")
        st.markdown(f"**Intended User:** {info.get('intended_user', '')}")
        st.markdown(f"**Decision Supported:** {info.get('decision_supported', '')}")
        st.markdown("#### Fairness")
        fair = info.get("fairness", {})
        st.markdown(f"- Protected attribute: {fair.get('protected_attribute', '')}")
        st.markdown(f"- Mitigations: {', '.join(fair.get('mitigations', []))}")
        st.markdown(f"- Metrics: {', '.join(fair.get('metrics', []))}")
    except Exception as e:
        st.warning(f"Could not fetch model info: {e}")

with col2:
    try:
        metrics = requests.get(f"{BACKEND_URL}/metrics").json()
        st.markdown("#### 📈 Metrics (Before vs After)")
        before = metrics.get("overall_before", {})
        after = metrics.get("overall_after", {})
        for k in before.keys():
            st.metric(label=k, value=f"{after[k]:.3f}", delta=f"{after[k]-before[k]:+.3f}")
    except Exception as e:
        st.warning(f"Could not fetch metrics: {e}")

# =========================
# 5️⃣  Style polish (optional)
# =========================
st.markdown(
    """
    <style>
    .stMetricLabel { font-size: 16px !important; font-weight: 600 !important; }
    div[data-testid="stMetricDelta"] > span { color: green !important; }
    </style>
    """,
    unsafe_allow_html=True
)

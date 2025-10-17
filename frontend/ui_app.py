# ui_app.py
import streamlit as st
import requests
import os
from typing import Dict

# Temporary environment setup (for Hugging Face / restricted hosts)
os.environ["STREAMLIT_HOME"] = "/tmp"
os.environ["STREAMLIT_RUNTIME_DIR"] = "/tmp"

# -------------------------
# 1) Basic page setup
# -------------------------
st.set_page_config(
    page_title="Absenteeism Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# <-- Change this to your backend deployment if different -->
BACKEND_URL = "http://127.0.0.1:8000"

st.title("📉 Absenteeism Risk Assessment Dashboard")
st.markdown(
    "A simple tool to help HR/team leader to spot employees who **may** be at higher risk of frequent absence. "
    "Use the prediction as a **supporting signal** — always combine with human judgement."
)
st.divider()

# -------------------------
# 2) Sidebar: human-friendly inputs
# -------------------------
st.sidebar.header("🧾 Employee information (enter details below)")
st.sidebar.caption("Values are translated for the model automatically.")

reason_map = {
    1: "Infectious diseases",
    2: "Injury / poisoning",
    3: "Respiratory diseases",
    4: "Digestive diseases",
    5: "Pregnancy-related",
    23: "Medical consultation",
    28: "Other reasons",
}
month_map = {i: name for i, name in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}
season_map = {1: "Summer", 2: "Autumn", 3: "Winter", 4: "Spring"}
education_map = {1: "High school", 2: "Graduate", 3: "Postgraduate", 4: "PhD"}

reason = st.sidebar.selectbox("Reason for absence", list(reason_map.keys()),
                              format_func=lambda k: reason_map.get(k, str(k)))
month = st.sidebar.selectbox("Month of absence", list(month_map.keys()),
                             format_func=lambda k: month_map.get(k, str(k)))
dow = st.sidebar.selectbox("Day of the week", [2, 3, 4, 5, 6],
                           format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri"][x - 2])
season = st.sidebar.selectbox("Season", list(season_map.keys()),
                              format_func=lambda k: season_map.get(k, str(k)))
edu = st.sidebar.selectbox("Education level", list(education_map.keys()),
                           format_func=lambda k: education_map.get(k, str(k)))
discipline = st.sidebar.radio("Disciplinary failure?", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")
drinker = st.sidebar.radio("Social drinker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
smoker = st.sidebar.radio("Social smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
service_time = st.sidebar.slider("Service time (years)", 0, 40, 8)
age = st.sidebar.slider("Age", 18, 70, 35)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 40.0, 24.5)


# -------------------------
# 3) Build features dict (include both name variations the model may expect)
# -------------------------
# Some saved pipelines expect slightly different column names; include both safe keys.
features: Dict[str, object] = {
    "Reason for absence": reason,
    "Month of absence": month,
    "Day of the week": dow,      # many pipelines use 'Day of the week'
    "Day of week": dow,          # some pipelines use 'Day of week'
    "Seasons": season,
    "Education": edu,
    "Disciplinary failure": discipline,
    "Social drinker": drinker,
    "Social smoker": smoker,
    "Service time": service_time,
    "Age": age,
    "BMI": bmi,
    "Body mass index": bmi,  # alternative naming
    # Add safe defaults for numeric features that the model may expect
    "Transportation expense": 200,
    "Distance from Residence to Work": 10,
    "Work load Average/day": 250,
    "Hit target": 95,
    "Son": 1,
    "Pet": 0,
    "Weight": 70,
    "Height": 170,
}

# -------------------------
# 4) Prediction area (big clear clickable button)
# -------------------------
st.subheader("🎯 Predict absenteeism risk")

left, right = st.columns([1, 1])

# Make the predict button visually easy to spot (CSS)
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #0078D4;
        color: white;
        border-radius: 8px;
        height: 44px;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with left:
    if st.button("🔍 Click here to predict absenteeism risk"):
        # show spinner while calling backend
        with st.spinner("Checking model and generating prediction…"):
            try:
                resp = requests.post(f"{BACKEND_URL}/predict", json={"features": features}, timeout=10)
            except Exception as e:
                st.error(f"⚠️ Could not contact backend at {BACKEND_URL}: {e}")
                resp = None

        if resp is None:
            st.stop()

        # handle response
        if resp.status_code == 200:
            data = resp.json()
            proba = float(data.get("proba", 0.0))
            label = int(data.get("label", 0))
            threshold = float(data.get("threshold", 0.5))
            explanations = data.get("explanations", []) or []

            # Friendly headline
            if label == 1:
                st.error(f"🚨 High risk — {proba:.1%} probability")
                st.write("This prediction indicates the employee is **more likely** to be frequently absent. "
                         "Consider supportive outreach (check-in, workload review, wellbeing resources).")
            else:
                st.success(f"✅ Low risk — {proba:.1%} probability")
                st.write("Attendance looks stable. No immediate action suggested from the model.")

            # Threshold display (human friendly)
            st.metric("Decision cutoff (model threshold)", f"{threshold * 100:.0f}%")
            st.caption("Predictions with probability above the cutoff are labelled **High risk**.")

            st.caption("This prediction is based on multiple attendance-related factors. "
           "Feature-by-feature explanations are not shown in this version.")

            st.markdown("---")
            st.caption("⚖️ Note: This is a predictive signal, not a decision. Combine with HR context and review before action.")
        else:
            # Show backend error message (friendly)
            try:
                err = resp.json()
                # some errors arrive as {"detail": "..."}; show politely
                detail = err.get("detail", str(err))
            except Exception:
                detail = resp.text
            st.error(f"Prediction failed: {detail}")

with right:
    st.info(
        """
        **How to read the result**
        - The percentage is the model's estimate of the *chance* the employee will be frequently absent.
        - **High risk (red)** → consider follow-up and support.
        - **Low risk (green)** → no action required from model alone.
        - Always verify with human context (manager notes, leave policies).
        """
    )

st.divider()

# # -------------------------
# 5) About the model & Performance FIRST (click to expand)
# -------------------------
st.subheader("📘 Model overview & performance")

# Fetch model-info + metrics safely
model_info, metrics_resp = {}, {}
try:
    model_info = requests.get(f"{BACKEND_URL}/model-info", timeout=5).json()
    metrics_resp = requests.get(f"{BACKEND_URL}/metrics", timeout=5).json()
except Exception:
    pass

# ----- EXPANDER 1: HOW WELL MODEL PERFORMS -----
with st.expander("📈 How well does the model perform?", expanded=False):
    before = metrics_resp.get("overall_before", {})
    after = metrics_resp.get("overall_after", {})
    if before and after:
        acc = after.get("acc", 0)
        f1 = after.get("f1", 0)
        auc = after.get("auc", 0)

        st.success(f"✅ The model predicts correctly about **{acc*100:.0f}%** of the time.")
        st.info(f"⚖️ It balances errors with an overall stability (F1) of **{f1*100:.0f}%**.")
        st.caption(f"✨ It distinguishes high-risk vs low-risk employees correctly about **{auc*100:.0f}%** of the time.")
    else:
        st.info("Performance information is not available for this deployment.")

# ----- EXPANDER 2: MODEL DETAILS -----
with st.expander("🔍 Model overview", expanded=False):
    st.markdown(f"**Purpose:** {model_info.get('purpose', '')}")
    st.markdown(f"**Intended user:** {model_info.get('intended_user', '')}")
    st.markdown(f"**Decision supported:** {model_info.get('decision_supported', '')}")
    st.caption("⚠️ This tool should not be used for hiring, firing, or disciplinary action.")

# ----- EXPANDER 3: FAIRNESS -----
with st.expander("⚖️ Fairness & transparency", expanded=False):
    fair = model_info.get("fairness", {})
    fair_after = metrics_resp.get("fairness_after", {})

    st.markdown(f"**Protected attribute:** {fair.get('protected_attribute','Age (≥40)')}")
    st.markdown("**What we did to reduce bias:**")
    mapping = {
        "drop_age_feature": "Removed age from inputs",
        "reweight_by_AxY": "Balanced training data",
        "probability_calibration": "Calibrated predictions for fairness"
    }
    for m in fair.get("mitigations", []):
        st.markdown(f"- {mapping.get(m,m)}")

    if fair_after:
        st.markdown("**Fairness after mitigation:**")
        name_map = {"SPD":"Parity gap","EOD":"TPR gap","FPR_diff":"False alert gap"}
        for k,v in fair_after.items():
            st.metric(label=name_map.get(k,k), value=f"{v:.3f}")
        st.caption("Closer to 0 → fairer between groups.")
    else:
        st.info("Fairness metrics not available.")

st.divider()



# -------------------------
# 6) Footer / small help
# -------------------------
st.markdown(
    """
    <small>
    Built for course assignment — uses a predictive model. Predictions are probabilistic signals, not final decisions..
    </small>
    """,
    unsafe_allow_html=True,
)

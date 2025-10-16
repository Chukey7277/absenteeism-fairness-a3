from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import json, os
import pandas as pd

# === GLOBALS ===
MODEL = None
PREPROC = None
THRESHOLD = 0.5
CALIBRATED = True
FEATURE_SCHEMA = None

ARTIF_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# === ARTIFACT LOADER ===
def _load_artifacts():
    global MODEL, PREPROC, THRESHOLD, FEATURE_SCHEMA, CALIBRATED

    try:
        import joblib
        mpath = os.path.join(ARTIF_DIR, "model_final.pkl")

        if os.path.exists(mpath):
            raw_model = joblib.load(mpath)
            print(f"✅ Loaded raw model: {type(raw_model)}")

            # If it's a CalibratedClassifierCV, extract its fitted base estimator
            if hasattr(raw_model, "calibrated_classifiers_"):
               inner = raw_model.calibrated_classifiers_[0]
              # Handle both attribute styles
            if hasattr(inner, "base_estimator"):
              MODEL = inner.base_estimator
            elif hasattr(inner, "estimator"):
              MODEL = inner.estimator
            else:
              raise AttributeError("No estimator found inside calibrated_classifiers_[0]")
            CALIBRATED = True
            print(f"✅ Extracted fitted pipeline: {type(MODEL)}")

            # Try to get preprocessor if available
            if hasattr(MODEL, "named_steps"):
                PREPROC = MODEL.named_steps.get("pre", None)
                print("✅ Found preprocessor in pipeline.")
            elif hasattr(MODEL, "estimator") and hasattr(MODEL.estimator, "named_steps"):
                PREPROC = MODEL.estimator.named_steps.get("pre", None)
                print("✅ Extracted preprocessor from nested estimator.")
            else:
                PREPROC = None
                print("⚠️ No preprocessor found in model.")
        else:
            print("❌ model_final.pkl not found in artifacts/")
            MODEL = None

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        MODEL = None
        PREPROC = None

    # Load threshold
    tpath = os.path.join(ARTIF_DIR, "threshold.json")
    if os.path.exists(tpath):
        try:
            THRESHOLD = float(json.load(open(tpath))["threshold"])
        except Exception:
            THRESHOLD = 0.5
    else:
        THRESHOLD = 0.5

    print(f"Model loaded calibrated={CALIBRATED}, threshold={THRESHOLD}")


# Load once at startup
_load_artifacts()


# === FASTAPI APP ===
app = FastAPI(title="Assignment UI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# === SCHEMAS ===
class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    proba: float
    label: int
    threshold: float
    calibrated: bool
    explanations: List[str]
    model_ready: bool


# === ROUTES ===
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None}


@app.get("/model-info")
def model_info():
    path = os.path.join(STATIC_DIR, "model_info.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="model_info.json missing")
    return json.load(open(path))


@app.get("/metrics")
def metrics():
    bpath = os.path.join(ARTIF_DIR, "metrics_before.json")
    apath = os.path.join(ARTIF_DIR, "metrics_after.json")

    before = json.load(open(bpath)) if os.path.exists(bpath) else {}
    after = json.load(open(apath)) if os.path.exists(apath) else {}

    def split(m):
        if not m:
            return {}, {}
        overall = {k: v for k, v in m.items() if k in {"acc", "prec", "rec", "f1", "auc"}}
        fairness = {k: v for k, v in m.items() if k in {"SPD", "EOD", "FPR_diff"}}
        return overall, fairness

    overall_b, fair_b = split(before)
    overall_a, fair_a = split(after)

    return {
        "overall_before": overall_b,
        "overall_after": overall_a,
        "fairness_before": fair_b,
        "fairness_after": fair_a,
    }


def _drop_fairness_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["age", "Age", "AGE"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _explain_linear(proba: float) -> List[str]:
    """Simple explanation for linear/logistic models"""
    try:
        import numpy as np
        if hasattr(PREPROC, "get_feature_names_out"):
            names = list(PREPROC.get_feature_names_out())
        else:
            names = [f"f{i}" for i in range(getattr(MODEL, "coef_", [[0]])[0].shape[0])]
        coefs = getattr(MODEL, "coef_", None)
        if coefs is None:
            return ["Model coefficients not available."]
        coefs = coefs[0]
        top_idx = np.argsort(np.abs(coefs))[::-1][:3]
        msgs = []
        for i in top_idx:
            direction = "↑" if coefs[i] > 0 else "↓"
            msgs.append(f"{names[i]} {direction} (impact)")
        return msgs
    except Exception:
        return ["Explanations unavailable; showing probability only."]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    if MODEL is None:
        return PredictResponse(
            proba=0.0,
            label=0,
            threshold=THRESHOLD,
            calibrated=CALIBRATED,
            explanations=["Model not loaded. Add artifacts."],
            model_ready=False
        )

    try:
        X_raw = pd.DataFrame([req.features])
        X_raw = _drop_fairness_columns(X_raw)

        # ✅ Step 1: Try using the first calibrated classifier (fitted one)
        try:
            if hasattr(MODEL, "calibrated_classifiers_"):
                inner = MODEL.calibrated_classifiers_[0].base_estimator
                proba = float(inner.predict_proba(X_raw)[:, 1][0])
            else:
                proba = float(MODEL.predict_proba(X_raw)[:, 1][0])
        except Exception as e1:
            # ✅ Step 2: fallback to direct estimator
            if hasattr(MODEL, "estimator") and hasattr(MODEL.estimator, "predict_proba"):
                proba = float(MODEL.estimator.predict_proba(X_raw)[:, 1][0])
            else:
                raise HTTPException(status_code=400, detail=f"Prediction failed: {e1}")

        label = int(proba >= THRESHOLD)

        try:
            explanations = _explain_linear(proba)
        except Exception:
            explanations = ["Explanations unavailable; showing probability only."]

        return PredictResponse(
            proba=proba,
            label=label,
            threshold=THRESHOLD,
            calibrated=CALIBRATED,
            explanations=explanations,
            model_ready=True
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")





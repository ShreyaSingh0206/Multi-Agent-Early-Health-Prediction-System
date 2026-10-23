
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve paths relative to this file so the app works from any working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ==========================================================
   APP BACKGROUND
========================================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #F5F9FF 0%,
        #EDF4FF 50%,
        #F8FAFC 100%
    );
}

/* ==========================================================
   HEADER
========================================================== */

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    color: #163B65;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #6B7280;
    font-size: 1rem;
    margin-bottom: 25px;
}

/* ==========================================================
   SUMMARY CARDS
========================================================== */

.card {
    background: linear-gradient(
        135deg,
        #FFFFFF 0%,
        #F8FBFF 100%
    );

    border-radius: 16px;
    padding: 18px;
    text-align: center;

    border: 1px solid #E5EDF7;

    box-shadow:
        0 4px 12px rgba(0,0,0,0.06);

    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 8px 20px rgba(0,0,0,0.10);
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: #64748B;
}

.card-value {
    font-size: 26px;
    font-weight: 700;
    color: #163B65;
    margin-top: 8px;
}

/* ==========================================================
   RISK BANNERS
========================================================== */

.risk-high {
    background: linear-gradient(
        135deg,
        #FFEAEA,
        #FFF5F5
    );

    border-left: 8px solid #E24B4A;
    border-radius: 14px;
    padding: 18px;
}

.risk-medium {
    background: linear-gradient(
        135deg,
        #FFF7E8,
        #FFFDF5
    );

    border-left: 8px solid #EF9F27;
    border-radius: 14px;
    padding: 18px;
}

.risk-low {
    background: linear-gradient(
        135deg,
        #EAF4FB,
        #F5FAFF
    );

    border-left: 8px solid #3B8BD4;
    border-radius: 14px;
    padding: 18px;
}

.risk-normal {
    background: linear-gradient(
        135deg,
        #E9FAF2,
        #F5FFF9
    );

    border-left: 8px solid #1D9E75;
    border-radius: 14px;
    padding: 18px;
}

/* ==========================================================
   SECTION HEADERS
========================================================== */

.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #163B65;

    padding-bottom: 8px;
    margin-bottom: 15px;

    border-bottom: 3px solid #3B82F6;
}

/* ==========================================================
   METRIC BOXES
========================================================== */

.metric-box {
    background: white;

    border-radius: 14px;

    padding: 18px;

    border: 1px solid #E5EDF7;

    box-shadow:
        0 2px 10px rgba(0,0,0,0.05);

    text-align: center;
}

/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #E5EDF7;
}

/* ==========================================================
   TABS
========================================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 12px;
    padding: 10px 20px;

    border: 1px solid #E5EDF7;

    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    );

    color: white;
}

/* ==========================================================
   BUTTONS
========================================================== */

.stButton button {
    width: 100%;
    border-radius: 12px;

    background: linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    );

    color: white;
    font-weight: 600;

    border: none;

    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37,99,235,0.3);
}

/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #163B65,
        #1F4E79
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL AND ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    ann_model  = tf.keras.models.load_model(MODEL_DIR / "ann_model.keras")
    lstm_model = tf.keras.models.load_model(MODEL_DIR / "lstm_model.keras")
    svm_model  = joblib.load(MODEL_DIR / "svm_model.pkl")
    xgb_model  = joblib.load(MODEL_DIR / "xgb_model.pkl")
    rf_model   = joblib.load(MODEL_DIR / "rf_model.pkl")
    lr_model   = joblib.load(MODEL_DIR / "lr_model.pkl")
    scaler     = joblib.load(MODEL_DIR / "scaler.pkl")
    scaler_seq = joblib.load(MODEL_DIR / "scaler_seq.pkl")
    le         = joblib.load(MODEL_DIR / "label_encoder.pkl")
    X_train    = np.load(DATA_DIR / "X_train.npy")
    return (ann_model, lstm_model, svm_model, xgb_model,
            rf_model, lr_model, scaler, scaler_seq, le, X_train)

try:
    (ann_model, lstm_model, svm_model, xgb_model, rf_model,
    lr_model, scaler, scaler_seq, le, X_train) = load_artifacts()

    FEATURES      = ["Systolic_BP", "Respiratory_Rate", "Oxygen_Saturation", "Heart_Rate"]
    class_names   = le.classes_
    risk_order    = {"High":0, "Low":1, "Medium":2, "Normal":3}
    risk_severity = {"Normal":0, "Low":1, "Medium":2, "High":3}
    artifacts_loaded = True
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Check that model files are in the 'models' folder and "
        "X_train.npy is in the 'data' folder."
    )
    artifacts_loaded = False
    st.stop()

class BaseAgent:
    def __init__(self, name, model, weight, scaler, is_temporal=False):
        self.name, self.model, self.weight = name, model, weight
        self.scaler, self.is_temporal = scaler, is_temporal
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
        else:
            probs = self.model.predict(X_scaled, verbose=0)
        return np.argmax(probs[0]), probs[0]

class TemporalAgent(BaseAgent):
    def predict(self, X_sequence):
        X_scaled = self.scaler.transform(X_sequence.reshape(-1,4)).reshape(1,3,4)
        probs = self.model.predict(X_scaled, verbose=0)
        return np.argmax(probs[0]), probs[0]

@st.cache_resource
def get_agents():
    return [
        BaseAgent("ANN", ann_model, 0.9200, scaler),
        BaseAgent("SVM", svm_model, 0.9067, scaler),
        BaseAgent("XGBoost", xgb_model, 0.9067, scaler),
        BaseAgent("RF", rf_model, 0.9000, scaler),
        BaseAgent("LR", lr_model, 0.9000, scaler),
        TemporalAgent("LSTM", lstm_model, 0.9200, scaler_seq, is_temporal=True),
    ]
agents = get_agents()

class MetaAgent:
    def __init__(self, agents, class_names, risk_severity):
        self.agents, self.class_names, self.risk_severity = agents, class_names, risk_severity
    def aggregate(self, X_static, X_sequence):
        results = []
        for a in self.agents:
            pred_idx, probs = a.predict(X_sequence if a.is_temporal else X_static)
            results.append({"agent":a.name, "pred_idx":pred_idx,
                "pred_label":self.class_names[pred_idx],
                "probs":probs, "confidence":float(np.max(probs)), "weight":a.weight})
        wp = np.zeros(len(self.class_names))
        tw = sum(r["weight"] for r in results)
        for r in results: wp += (r["weight"]/tw) * r["probs"]
        final_idx = np.argmax(wp)
        static_preds = [r["pred_label"] for r in results if r["agent"]!="LSTM"]
        static_majority = max(set(static_preds), key=static_preds.count)
        lstm_pred = next(r["pred_label"] for r in results if r["agent"]=="LSTM")
        early_warning = self.risk_severity[lstm_pred] > self.risk_severity[static_majority]
        agreement = sum(1 for r in results if r["pred_idx"]==final_idx)/len(results)
        return {"final_prediction":self.class_names[final_idx], "final_idx":final_idx,
                "weighted_probs":wp, "agent_results":results,
                "early_warning":early_warning, "agreement_score":agreement,
                "static_prediction":static_majority, "lstm_prediction":lstm_pred}

meta_agent = MetaAgent(agents, class_names, risk_severity)

#Specialist Agents
class ClinicalRuleAgent:
    def __init__(self):
        self.name = "ClinicalRuleAgent"

    def score_vital(self, feature, value):
        """Returns NEWS2 score for each vital sign"""
        scores = {
            "Respiratory_Rate": [
                (value <= 8,   3),
                (9  <= value <= 11, 1),
                (12 <= value <= 20, 0),
                (21 <= value <= 24, 2),
                (value >= 25,  3),
            ],
            "Oxygen_Saturation": [
                (value <= 91,  3),
                (92 <= value <= 93, 2),
                (94 <= value <= 95, 1),
                (value >= 96,  0),
            ],
            "Systolic_BP": [
                (value <= 90,  3),
                (91 <= value <= 100, 2),
                (101 <= value <= 110, 1),
                (111 <= value <= 219, 0),
                (value >= 220, 3),
            ],
            "Heart_Rate": [
                (value <= 40,  3),
                (41 <= value <= 50, 1),
                (51 <= value <= 90, 0),
                (91 <= value <= 110, 1),
                (111 <= value <= 130, 2),
                (value >= 131, 3),
            ],
        }
        for condition, score in scores.get(feature, []):
            if condition:
                return score
        return 0

    def assess(self, patient_values):
        """
        Returns NEWS2 total score and corresponding risk level
        """
        features = ["Systolic_BP", "Respiratory_Rate",
                    "Oxygen_Saturation", "Heart_Rate"]
        scores   = {}
        total    = 0

        for feat, val in zip(features, patient_values):
            s = self.score_vital(feat, val)
            scores[feat] = s
            total += s

        # NEWS2 risk mapping
        if total <= 4:
            risk = "Low"
            action = "Routine monitoring"
        elif total <= 6:
            risk = "Medium"
            action = "Urgent review by nurse/doctor"
        else:
            risk = "High"
            action = "Emergency — immediate medical review"

        if total == 0:
            risk   = "Normal"
            action = "Continue routine care"

        return {
            "agent"         : self.name,
            "news2_score"   : total,
            "feature_scores": scores,
            "risk_level"    : risk,
            "action"        : action,
            "message"       : f"NEWS2 Score: {total} → {risk} Risk — {action}"
        }

class TrendAgent:
    def __init__(self):
        self.name = "TrendAgent"
        # For each feature: positive change = increasing risk (True)
        #                   negative change = increasing risk (False)
        self.risk_direction = {
            "Systolic_BP"      : "high",   # high BP = risk
            "Respiratory_Rate" : "high",   # high RR = risk
            "Oxygen_Saturation": "low",    # low O2  = risk
            "Heart_Rate"       : "high",   # high HR = risk
        }
        self.features = ["Systolic_BP", "Respiratory_Rate",
                         "Oxygen_Saturation", "Heart_Rate"]

    def analyze(self, reading_2hr, reading_1hr, reading_now):
        """
        Computes trend direction, velocity and acceleration
        for each vital sign
        """
        readings = np.array([reading_2hr, reading_1hr, reading_now])
        trends   = {}
        deteriorating_count = 0

        for i, feat in enumerate(self.features):
            vals = readings[:, i]

            # Rate of change (velocity)
            delta_1 = vals[1] - vals[0]   # T-2hr to T-1hr
            delta_2 = vals[2] - vals[1]   # T-1hr to T-now

            # Acceleration (is it speeding up?)
            acceleration = delta_2 - delta_1

            # Direction
            if abs(delta_2) < 0.5:
                direction = "stable"
            elif delta_2 > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            # Is this trend worsening?
            risk_dir = self.risk_direction[feat]
            if risk_dir == "high":
                worsening = delta_2 > 2    # rising = bad
            else:
                worsening = delta_2 < -2   # falling = bad

            if worsening:
                deteriorating_count += 1

            trends[feat] = {
                "values"      : vals.tolist(),
                "delta"       : round(float(delta_2), 2),
                "acceleration": round(float(acceleration), 2),
                "direction"   : direction,
                "worsening"   : worsening
            }

        # Overall trend assessment
        if deteriorating_count >= 3:
            trend_risk   = "High"
            trend_status = "RAPIDLY DETERIORATING"
        elif deteriorating_count == 2:
            trend_risk   = "Medium"
            trend_status = "DETERIORATING"
        elif deteriorating_count == 1:
            trend_risk   = "Low"
            trend_status = "MILD DETERIORATION"
        else:
            trend_risk   = "Normal"
            trend_status = "STABLE"

        return {
            "agent"               : self.name,
            "feature_trends"      : trends,
            "deteriorating_count" : deteriorating_count,
            "trend_risk"          : trend_risk,
            "trend_status"        : trend_status,
            "early_warning"       : deteriorating_count >= 2
        }

class ConfidenceAgent:
    def __init__(self):
        self.name = "ConfidenceAgent"

    def evaluate(self, agent_results):
        """
        agent_results: list of dicts from model agents
        """
        confidences  = [a["confidence"] for a in agent_results]
        predictions  = [a["pred_label"] for a in agent_results]
        pred_set     = set(predictions)

        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        disagreement   = len(pred_set)       # how many unique predictions

        # Identify outlier agents (disagree with majority)
        majority_pred = max(set(predictions), key=predictions.count)
        outliers = [a["agent"] for a in agent_results
                    if a["pred_label"] != majority_pred]

        # Confidence level
        if avg_confidence >= 0.80 and disagreement <= 2:
            confidence_level = "HIGH"
            reliable         = True
        elif avg_confidence >= 0.60 and disagreement <= 3:
            confidence_level = "MEDIUM"
            reliable         = True
        else:
            confidence_level = "LOW"
            reliable         = False

        return {
            "agent"            : self.name,
            "avg_confidence"   : round(avg_confidence, 4),
            "min_confidence"   : round(min_confidence, 4),
            "max_confidence"   : round(max_confidence, 4),
            "unique_predictions": disagreement,
            "majority_pred"    : majority_pred,
            "outlier_agents"   : outliers,
            "confidence_level" : confidence_level,
            "reliable"         : reliable,
            "message"          : (
                f"Confidence: {confidence_level} "
                f"(avg={avg_confidence*100:.1f}%) — "
                f"{'Reliable' if reliable else 'Uncertain — review manually'}"
            )
        }

class AlertAgent:
    def __init__(self):
        self.name = "AlertAgent"

    def generate_alert(self, meta_result, clinical_result,
                       trend_result, confidence_result,
                       shap_result, cf_result):

        final_risk    = meta_result["final_prediction"]
        news2_score   = clinical_result["news2_score"]
        trend_status  = trend_result["trend_status"]
        early_warning = (meta_result["early_warning"] or
                         trend_result["early_warning"])
        confidence    = confidence_result["confidence_level"]

        # Alert level decision
        if final_risk == "High" or news2_score >= 7:
            alert_level = "🚨 CRITICAL ALERT"
            alert_color = "RED"
            action      = "Immediate emergency intervention required"
        elif final_risk == "Medium" or news2_score >= 5:
            alert_level = "⚠️ WARNING"
            alert_color = "ORANGE"
            action      = "Urgent clinical review within 30 minutes"
        elif final_risk == "Low" or news2_score >= 1:
            alert_level = "🔵 CAUTION"
            alert_color = "YELLOW"
            action      = "Increase monitoring frequency"
        else:
            alert_level = "✅ NORMAL"
            alert_color = "GREEN"
            action      = "Continue routine care"

        # Early warning upgrade
        if early_warning and final_risk != "High":
            alert_level = "⚠️ EARLY WARNING — " + alert_level
            action      = "Trending toward higher risk — " + action

        return {
            "agent"         : self.name,
            "alert_level"   : alert_level,
            "alert_color"   : alert_color,
            "final_risk"    : final_risk,
            "news2_score"   : news2_score,
            "trend_status"  : trend_status,
            "early_warning" : early_warning,
            "action"        : action,
            "confidence"    : confidence,
            "top_driver"    : shap_result["top_feature"],
            "cf_available"  : cf_result["needed"],
        }
    
clinical_agent = ClinicalRuleAgent()
trend_agent = TrendAgent()
confidence_agent = ConfidenceAgent()
alert_agent = AlertAgent()

# SHAP explainer (cached so it loads only once)
@st.cache_resource
def get_explainer(_model, _X_train):
    background = _X_train[:100]
    return shap.DeepExplainer(_model, background)

explainer = get_explainer(ann_model, X_train)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
RISK_COLORS = {
    "High":   "#E24B4A",
    "Medium": "#EF9F27",
    "Low":    "#3B8BD4",
    "Normal": "#1D9E75"
}
RISK_ICONS = {"High":"🔴", "Medium":"🟡", "Low":"🔵", "Normal":"🟢"}
NEXT_CLASS = {
    "High":   ("Medium", 2),
    "Medium": ("Low",    1),
    "Low":    ("Normal", 3),
    "Normal": (None,     None)
}


def get_shap(scaled):
    sv = explainer.shap_values(scaled)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]
    return sv

def get_counterfactuals(patient_values, pred_class_idx,
                        target_class_idx, n=30000):
    bounds = {
        "Systolic_BP"      : (60,               patient_values[0]),
        "Respiratory_Rate" : (10,               patient_values[1]),
        "Oxygen_Saturation": (patient_values[2], 100),
        "Heart_Rate"       : (50,               patient_values[3]),
    }
    samples = np.column_stack([
        np.random.uniform(bounds[f][0], bounds[f][1], n)
        for f in FEATURES
    ])
    preds = np.argmax(
        ann_model.predict(scaler.transform(samples), verbose=0), axis=1)
    cf_samples = samples[preds == target_class_idx]
    if len(cf_samples) == 0:
        return None
    distances = np.sum(np.abs(cf_samples - np.array(patient_values)), axis=1)
    top3 = cf_samples[np.argsort(distances)[:3]]
    return top3

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Patient Input
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Vital Signs (3 Readings)")
    st.markdown("---")

    st.markdown("**T-2hr (2 hours ago)**")
    sbp_2 = st.slider("Systolic BP", 60, 180, 115, key="sbp2")
    rr_2  = st.slider("Respiratory Rate", 10, 40, 19, key="rr2")
    o2_2  = st.slider("Oxygen Saturation", 85, 100, 95, key="o22")
    hr_2  = st.slider("Heart Rate", 50, 180, 88, key="hr2")

    st.markdown("**T-1hr (1 hour ago)**")
    sbp_1 = st.slider("Systolic BP", 60, 180, 118, key="sbp1")
    rr_1  = st.slider("Respiratory Rate", 10, 40, 22, key="rr1")
    o2_1  = st.slider("Oxygen Saturation", 85, 100, 93, key="o21")
    hr_1  = st.slider("Heart Rate", 50, 180, 105, key="hr1")

    st.markdown("**T-now (current)**")
    sbp_0 = st.slider("Systolic BP", 60, 180, 120, key="sbp0")
    rr_0  = st.slider("Respiratory Rate", 10, 40, 26, key="rr0")
    o2_0  = st.slider("Oxygen Saturation", 85, 100, 90, key="o20")
    hr_0  = st.slider("Heart Rate", 50, 180, 140, key="hr0")

    predict_btn = st.button("🔍 Run Multi-Agent Prediction",
                            use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("### 📊 Normal Ranges")
    st.markdown("""
    | Vital Sign | Normal Range |
    |-----------|-------------|
    | Systolic BP | 90–120 mmHg |
    | Respiratory Rate | 12–20 /min |
    | Oxygen Saturation | 95–100% |
    | Heart Rate | 60–100 bpm |
    """)

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Patient Health Risk Prediction</div>',
            unsafe_allow_html=True)


# ── Default state message ────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("👈 Enter patient vital signs in the sidebar")
    with col2:
        st.info("🔍 Click **Predict Risk Level** to get the prediction")
    with col3:
        st.info("📊 SHAP and counterfactual explanations will appear below")
    st.stop()

reading_2hr = [sbp_2, rr_2, o2_2, hr_2]
reading_1hr = [sbp_1, rr_1, o2_1, hr_1]
reading_now = [sbp_0, rr_0, o2_0, hr_0]

X_static   = np.array([reading_now])
X_sequence = np.stack([reading_2hr, reading_1hr, reading_now])

patient_values = reading_now
scaled = scaler.transform(X_static)

meta_result       = meta_agent.aggregate(X_static, X_sequence)
clinical_result   = clinical_agent.assess(reading_now)
trend_result      = trend_agent.analyze(reading_2hr, reading_1hr, reading_now)
confidence_result = confidence_agent.evaluate(meta_result["agent_results"])

pred_class = meta_result["final_prediction"]
pred_idx   = meta_result["final_idx"]
probs      = meta_result["weighted_probs"]

# ==========================================================
# RISK SUMMARY CARDS
# ==========================================================

st.markdown("""
<style>
.card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.card-title {
    font-size: 14px;
    color: #666;
    margin-bottom: 8px;
}

.card-value {
    font-size: 24px;
    font-weight: bold;
    color: #1F3864;
}
</style>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Risk Level</div>
        <div class="card-value">{RISK_ICONS[pred_class]} {pred_class}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">NEWS2 Score</div>
        <div class="card-value">{clinical_result['news2_score']}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    confidence = confidence_result["avg_confidence"] * 100
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Confidence</div>
        <div class="card-value">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Trend Status</div>
        <div class="card-value">{trend_result['trend_risk']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Prediction",
    "🤖 Agents",
    "📈 Trend Analysis",
    "🔍 SHAP",
    "💡 Counterfactuals"
])

# ───────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────


# ── Two column layout — probabilities + bar chart ────────────
with tab1:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="section-header">Class Probabilities</div>',
                    unsafe_allow_html=True)
        for i, name in enumerate(class_names):
            prob = float(probs[i])
            color_bar = RISK_COLORS[name]
            is_pred   = (name == pred_class)
            st.markdown(
                f"**{RISK_ICONS[name]} {name}** {'← Predicted' if is_pred else ''}",
                unsafe_allow_html=True
            )
            st.progress(prob)
            st.markdown(f"<small style='color:#666'>{prob*100:.2f}%</small>",
                        unsafe_allow_html=True)
            st.markdown("")

    with col2:
        st.markdown('<div class="section-header">Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            class_names,
            [float(probs[i]) * 100 for i in range(len(class_names))],
            color=[RISK_COLORS[n] for n in class_names],
            edgecolor="white", linewidth=1.5, width=0.5
        )
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{float(prob)*100:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Probability (%)", fontsize=11)
        ax.set_title("Risk Level Probabilities", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

# ── Agent-wise predictions table ─────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">🤖 Individual Agent Predictions</div>',
        unsafe_allow_html=True
    )

    agent_df = pd.DataFrame([
        {
            "Agent": a["agent"],
            "Prediction": a["pred_label"],
            "Confidence": f"{a['confidence']*100:.1f}%"
        }
        for a in meta_result["agent_results"]
    ])

    st.dataframe(
        agent_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"🤝 Agreement Score: {meta_result['agreement_score']*100:.0f}%"
    )

# ── Trend Agent ───────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📈 Trend Analysis</div>',
                unsafe_allow_html=True)
    st.write(f"**Status:** {trend_result['trend_status']} "
            f"({trend_result['deteriorating_count']}/4 vitals worsening)")
    trend_df = pd.DataFrame([
        {"Vital": k, "Δ (last hour)": v["delta"],
        "Direction": v["direction"], "Worsening": "⚠️" if v["worsening"] else "✅"}
        for k, v in trend_result["feature_trends"].items()
    ])
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

    # ── Clinical NEWS2 score ──────────────────────────────────────
    st.markdown('<div class="section-header">🏥 Clinical NEWS2 Score</div>',
                unsafe_allow_html=True)
    st.metric("NEWS2 Score", clinical_result["news2_score"])
    st.write(clinical_result["message"])

    # ── Confidence ────────────────────────────────────────────────
    st.info(confidence_result["message"])
    if confidence_result["outlier_agents"]:
        st.warning(f"Outlier agents (disagreed): {', '.join(confidence_result['outlier_agents'])}")

    # ── Early warning banner ─────────────────────────────────────
    if meta_result["early_warning"] or trend_result["early_warning"]:
        st.error(
            f"⚠️ **EARLY WARNING**: Static prediction is "
            f"'{meta_result['static_prediction']}' but temporal trend "
            f"(LSTM: '{meta_result['lstm_prediction']}') and trend analysis "
            f"suggest the patient is deteriorating. Increase monitoring frequency."
        )
    else:
        st.success("✅ No early warning — patient trend is stable.")

# ─────────────────────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📊 SHAP Explanation — Why this risk level?</div>',
                unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        shap_values = get_shap(scaled)

    col3, col4 = st.columns(2)

    with col3:
        # Waterfall plot
        shap_exp = shap.Explanation(
            values        = shap_values[pred_idx][0],
            base_values   = float(explainer.expected_value[pred_idx]),
            data          = scaled[0],
            feature_names = FEATURES
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        shap.waterfall_plot(shap_exp, show=False)
        plt.title(f"Feature contributions → {pred_class} Risk", fontsize=11)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close("all")

    with col4:
        # Manual SHAP bar chart (cleaner for display)
        sv   = shap_values[pred_idx][0]
        feat_df = pd.DataFrame({
            "Feature"   : FEATURES,
            "SHAP Value": sv,
            "Direction" : ["↑ Increases Risk" if v > 0
                        else "↓ Decreases Risk" for v in sv]
        }).sort_values("SHAP Value", key=abs, ascending=True)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        colors_shap = ["#E24B4A" if v > 0 else "#3B8BD4"
                    for v in feat_df["SHAP Value"]]
        ax3.barh(feat_df["Feature"], feat_df["SHAP Value"],
                color=colors_shap, edgecolor="white")
        ax3.axvline(0, color="black", linewidth=0.8)
        ax3.set_xlabel("SHAP Value", fontsize=11)
        ax3.set_title("Feature Impact on Prediction\n(Red = increases risk, Blue = decreases risk)",
                    fontsize=10)
        ax3.grid(axis="x", alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Text interpretation
        st.markdown("**Interpretation:**")
        for _, row in feat_df.sort_values("SHAP Value", key=abs,
                                        ascending=False).iterrows():
            direction = "🔴 increases" if row["SHAP Value"] > 0 else "🔵 decreases"
            st.markdown(
                f"- **{row['Feature']}** {direction} "
                f"the {pred_class} risk prediction by "
                f"`{abs(row['SHAP Value']):.3f}`"
            )

    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# COUNTERFACTUAL EXPLANATION
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">💡 Counterfactual Explanation — How to reduce risk?</div>',
                unsafe_allow_html=True)

    if pred_class == "Normal":
        st.success("✅ Patient is already at **Normal** risk level. "
                "No intervention needed.")
    else:
        next_name, next_idx = NEXT_CLASS[pred_class]
        st.markdown(
            f"The following changes to vital signs could move the patient "
            f"from **{pred_class}** → **Normal** risk:"
        )

        with st.spinner("Generating counterfactual recommendations..."):
            cf_results = get_counterfactuals(
                patient_values, pred_idx, 3)   # 3 = Normal class

        if cf_results is None:
            st.warning("Could not find counterfactuals. Try adjusting vital signs.")
        else:
            # Table
            rows = []
            for i, cf in enumerate(cf_results):
                row = {"Suggestion": f"Option {i+1}"}
                for j, feat in enumerate(FEATURES):
                    diff  = cf[j] - patient_values[j]
                    arrow = "↑" if diff > 0 else "↓"
                    row[feat] = f"{cf[j]:.1f} ({arrow}{abs(diff):.1f})"
                rows.append(row)

            cf_df = pd.DataFrame(rows).set_index("Suggestion")
            st.dataframe(cf_df, use_container_width=True)

            # Grouped bar chart
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            x     = np.arange(len(FEATURES))
            width = 0.2
            opt_colors = ["#3B8BD4", "#1D9E75", "#EF9F27"]
            opt_labels = ["Option 1", "Option 2", "Option 3"]

            ax4.bar(x - 1.5*width, patient_values, width,
                    label="Original", color="#E24B4A", alpha=0.9)
            for i, (cf, color, label) in enumerate(
                    zip(cf_results, opt_colors, opt_labels)):
                ax4.bar(x + (i - 0.5)*width, cf, width,
                        label=label, color=color, alpha=0.9)

            ax4.set_xticks(x)
            ax4.set_xticklabels(FEATURES, fontsize=10)
            ax4.set_ylabel("Feature Value", fontsize=11)
            ax4.set_title(f"Counterfactual Suggestions: "
                        f"{pred_class} → Normal Risk", fontsize=12)
            ax4.legend()
            ax4.grid(axis="y", alpha=0.3)
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            st.info(
                "💊 **Clinical Note:** Suggestions constrain Heart Rate, "
                "Respiratory Rate and Systolic BP to decrease, "
                "and Oxygen Saturation to increase — "
                "reflecting medically valid intervention directions."
            )

    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem'>"
    "Patient Health Risk Prediction System · "
    "Multi Agent Multi-model Framework · "
    "For research purposes only"
    "</div>",
    unsafe_allow_html=True
)

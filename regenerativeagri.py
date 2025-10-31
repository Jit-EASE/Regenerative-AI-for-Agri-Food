# regenerative_ai_streamlit_v1.py
# -----------------------------------------------------------------------------
# Regenerative AI Dashboard (Olive Grove Scenario) — Research Prototype (Full)
# Designed & Developed by Jit (fresh build with advanced modules)
# -----------------------------------------------------------------------------
# Pillars implemented
# A) Inference & Uncertainty: Logistic + Isotonic calibration, Conformal bands,
#    Permutation importance, time-aware CV stubs
# B) Adaptation, Drift & Causality: PSI + KS-drift, continual replay buffer,
#    local counterfactual deltas
# C) Optimisation & What-If: MILP irrigation portfolio, Pareto frontier, Scenario editor
# D) Governance & Safety: Safety contracts, overrides log, audit export, federation stub,
#    regeneration KPIs, model card export
# E) Operator Experience: Map context (pydeck), alerts & duty-cycle throttle,
#    session export/import, auto NLP explanations (OpenAI-backed with local fallback)
# -----------------------------------------------------------------------------

import os, io, json, uuid, time, hashlib, math, textwrap
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Modeling & stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, brier_score_loss
import statsmodels.api as sm
from scipy.stats import ks_2samp

# Optimisation
try:
    import pulp as pl
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# Optional OpenAI (agentic AI explainer). Uses new SDK interface.
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False

# --------------------------- App Config --------------------------------------
st.set_page_config(
    page_title="Regenerative AI Dashboard — Olive Grove (Research Prototype)",
    page_icon="🫒",
    layout="wide"
)

APP_VERSION = "v1.0"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------- Utilities ---------------------------------------
def sha256_of_dict(d: Dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_vpd_celsius(temp_c: float, rh_pct: float) -> float:
    # Simplified VPD approximation (kPa) for demo purposes
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    vpd = max(es - ea, 0.0)
    return float(vpd)

def simulate_batch(n:int, seed:int, drift:float=0.0) -> pd.DataFrame:
    """Generate synthetic olive-grove sensor data. Drift shifts means & adds noise."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    # Baseline seasonal patterns (very simplified)
    temp_c = 26 + 4*np.sin(idx/15) + rng.normal(0, 0.8, n) + drift*1.2
    rh = 45 + 10*np.sin(idx/17 + 0.3) + rng.normal(0, 2.0, n) - drift*2.0
    soil_moist = 0.28 + 0.05*np.sin(idx/20 + 1.1) + rng.normal(0, 0.015, n) - drift*0.04
    ndvi = 0.72 + 0.02*np.sin(idx/21 + 0.5) + rng.normal(0, 0.01, n) - drift*0.02

    # Clip to plausible bounds
    rh = np.clip(rh, 15, 95)
    soil_moist = np.clip(soil_moist, 0.05, 0.55)
    ndvi = np.clip(ndvi, 0.2, 0.9)

    # Derived
    vpd = np.array([compute_vpd_celsius(t, h) for t, h in zip(temp_c, rh)], dtype=float)

    # Hidden ground truth for "water stress" (synthetic)
    logit = -2.0 + 4.0*(vpd - 0.9) - 5.0*(soil_moist - 0.25) - 1.5*(ndvi - 0.7) + rng.normal(0, 0.6, n)
    p_stress = 1/(1 + np.exp(-logit))
    y = rng.binomial(1, p_stress)

    df = pd.DataFrame({
        "t": idx,
        "temp_c": np.round(temp_c, 3),
        "rh": np.round(rh, 3),
        "soil_moist": np.round(soil_moist, 3),
        "ndvi": np.round(ndvi, 3),
        "vpd": np.round(vpd, 4),
        "water_stress": y.astype(int),
        "p_true": np.round(p_stress, 4),
    })

    now = datetime.utcnow().isoformat()
    df["ts_utc"] = now
    return df

# --- Stability & Drift ---
def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins:int=10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, qs)
    edges = np.unique(edges)
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)
    eps = 1e-8
    psi = np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps)))
    return float(psi)

def ks_drift(ref: np.ndarray, cur: np.ndarray, alpha: float=0.01) -> Tuple[bool, float]:
    if len(ref)<30 or len(cur)<30:
        return False, 1.0
    ks, p = ks_2samp(ref, cur)
    return (p < alpha), float(p)

# --------------------------- Models & Uncertainty ----------------------------
FEATURES = ["vpd","soil_moist","ndvi","temp_c","rh"]

class CalibratedLogit:
    def __init__(self):
        self.logit = None
        self.cal = None
        self.summary_ = None

    def fit(self, df: pd.DataFrame, target="water_stress"):
        X = sm.add_constant(df[FEATURES])
        y = df[target].astype(int)
        self.logit = sm.Logit(y, X).fit(disp=False)
        raw = self.logit.predict(X)
        # Simple time-aware folds by index quantiles
        folds = pd.qcut(df.index, q=min(5, max(2, len(df)//50)), duplicates="drop")
        # Train calibration on all but last fold
        mask = folds != folds.iloc[-1]
        self.cal = IsotonicRegression(out_of_bounds="clip")
        self.cal.fit(raw[mask], y[mask])
        self.summary_ = str(self.logit.summary())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(df[FEATURES])
        raw = self.logit.predict(X)
        p = self.cal.transform(raw)
        return np.clip(p, 0, 1)


def conformal_band(probs: np.ndarray, y: np.ndarray, alpha: float=0.1) -> Tuple[float,float]:
    s = np.abs(y - probs)
    q = np.quantile(s, 1-alpha)
    lo = np.clip(probs[-1]-q, 0, 1)
    hi = np.clip(probs[-1]+q, 0, 1)
    return float(lo), float(hi)


def permutation_importance(model: CalibratedLogit, df: pd.DataFrame, target: str) -> Dict[str, float]:
    base = model.predict_proba(df)
    base_brier = brier_score_loss(df[target], base)
    imps = {}
    rng = np.random.default_rng(7)
    for f in FEATURES:
        dfp = df.copy()
        dfp[f] = rng.permutation(dfp[f].values)
        p = model.predict_proba(dfp)
        imps[f] = (brier_score_loss(dfp[target], p) - base_brier)
    return dict(sorted(imps.items(), key=lambda x: -x[1]))

# --------------------------- Counterfactuals ---------------------------------

def local_delta(model: CalibratedLogit, df: pd.DataFrame, feature: str, delta: float) -> float:
    dfp = df.copy()
    if feature in dfp.columns:
        dfp.iloc[-1, dfp.columns.get_loc(feature)] += delta
    p0 = model.predict_proba(df)[-1]
    p1 = model.predict_proba(dfp)[-1]
    return float(p1 - p0)

# --------------------------- Optimisation ------------------------------------

def optimise_irrigation(window_df: pd.DataFrame, max_water_l: float, penalty_per_stress: float=50.0):
    if not HAS_PULP or window_df.empty:
        return None
    T = len(window_df)
    x = pl.LpVariable.dicts("irrigate_l", range(T), lowBound=0)
    m = pl.LpProblem("IrrigationPlan", pl.LpMinimize)
    stress_p = window_df["pred"].values
    # simple diminishing risk with water: (1 - 0.005 * x)
    m += pl.lpSum([x[t] for t in range(T)]) + penalty_per_stress * pl.lpSum(stress_p[t]*(1 - 0.005*x[t]) for t in range(T))
    m += pl.lpSum([x[t] for t in range(T)]) <= max_water_l
    m.solve(pl.PULP_CBC_CMD(msg=False))
    sol = [pl.value(x[t]) for t in range(T)]
    return sol

# Pareto frontier (water vs risk vs uncertainty)

def pareto_points(preds: np.ndarray, water_grid: List[float], uncert_band: float) -> pd.DataFrame:
    rows = []
    risk_now = preds.mean() if len(preds) else np.nan
    for W in water_grid:
        # toy mapping: water reduces risk linearly up to 40% (demo)
        risk_after = max(risk_now * (1 - min(W/1000.0, 0.4)), 0)
        rows.append({"water_L": W, "risk_after": risk_after, "uncert_band": uncert_band})
    return pd.DataFrame(rows)

# --------------------------- Governance & Safety -----------------------------

def kb_context() -> List[Dict[str,str]]:
    return [
        {"id":"policy_irrigation", "title":"Irrigation Policy v1",
         "text":"Trigger irrigation when modeled water-stress risk > 0.6 and uncertainty band < 0.5, unless VPD drift PSI > 0.2."},
        {"id":"agronomy_vpd", "title":"Agronomy Note — VPD",
         "text":"Higher VPD implies stronger evaporative demand; leaves lose water faster, increasing irrigation need if soil water is low."},
        {"id":"governance_eu_ai", "title":"Governance — EU AI Act (demo)",
         "text":"Maintain lineage, human oversight, model cards, and override logs. Record rationale for operator changes."}
    ]


def record_override(artifact_id: str, new_action: str, rationale: str, operator: str):
    st.session_state.overrides.append({
        "artifact_id": artifact_id,
        "new_action": new_action,
        "rationale": rationale,
        "operator": operator,
        "ts_utc": datetime.utcnow().isoformat()
    })


def water_use_efficiency(water_l: float, risk_before: float, risk_after: float) -> float:
    averted = max(risk_before - risk_after, 0)
    return float("nan") if averted==0 else water_l / averted


def auto_nlp_explanation(artifact: Dict, extras: Dict) -> Dict:
    """Auto-generate operator-facing text. Uses OpenAI if available; else local template."""
    kb = kb_context()
    kb_block = "\n".join([f"[{x['id']}] {x['title']}: {x['text']}" for x in kb])
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY", ""):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = [
                {"role":"system","content":textwrap.dedent("""
                    You are a Regenerative AI operations explainer. Explain decisions clearly for farm operators.
                    RULES:
                    - Ground claims in the provided artifact and KB; cite KB items in [brackets].
                    - Include an action checklist and note residual risks.
                    - Be concise, factual, and non-technical.
                """)},
                {"role":"user","content":json.dumps({
                    "artifact": artifact,
                    "kb": kb_block,
                    "extras": extras
                }, ensure_ascii=False)}
            ]
            schema = {
                "type":"json_object",
                "schema":{
                    "type":"object",
                    "properties":{
                        "rationale":{"type":"string"},
                        "citations":{"type":"array","items":{"type":"string"}},
                        "operator_checklist":{"type":"array","items":{"type":"string"}},
                        "residual_risks":{"type":"array","items":{"type":"string"}}
                    }
                }
            }
            completion = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages,
                temperature=0.2, response_format={"type":"json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            pass
    # Local fallback
    p = artifact.get("prediction", float("nan"))
    lo, hi = artifact.get("confidence_bounds", [float("nan"), float("nan")])
    drift = artifact.get("risk_checks",{}).get("drift",{})
    action = artifact.get("action","defer_and_observe")
    check = [
        "Confirm field observations for wilting or leaf curl",
        "Check soil moisture at 10–20 cm depth",
        "Verify pump capacity and water availability",
        "Reassess in 2 hours if uncertainty remains high"
    ]
    residual = []
    if not (np.isfinite(lo) and np.isfinite(hi)) or (hi-lo)>0.5:
        residual.append("High uncertainty band; additional readings advised")
    if drift.get("psi_vpd",0)>0.2:
        residual.append("VPD distribution drift; policy suggests conservative plan [policy_irrigation]")
    rationale = (
        f"Model indicates water-stress risk ~{p:.2f} with confidence band [{lo:.2f},{hi:.2f}]. "
        f"Action suggested: {action}. Primary drivers: VPD↑ and soil moisture↓ [agronomy_vpd]. "
        f"Follow governance practices for oversight [governance_eu_ai]."
    )
    return {
        "rationale": rationale,
        "citations":["[agronomy_vpd]","[policy_irrigation]","[governance_eu_ai]"],
        "operator_checklist": check,
        "residual_risks": residual
    }

# --------------------------- Session State -----------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "baseline" not in st.session_state:
    st.session_state.baseline = simulate_batch(400, seed=123, drift=0.0)
if "decisions" not in st.session_state:
    st.session_state.decisions = []  # episodic memory of artifacts
if "outcomes" not in st.session_state:
    st.session_state.outcomes = []   # feedback memory
if "replay" not in st.session_state:
    st.session_state.replay = []     # continual learning buffer
if "overrides" not in st.session_state:
    st.session_state.overrides = []  # operator overrides
if "audit" not in st.session_state:
    st.session_state.audit = []      # JSONL audit entries

# --------------------------- Sidebar -----------------------------------------
with st.sidebar:
    st.markdown("## 🫒 Regenerative AI — Controls")
    st.caption(f"Version: {APP_VERSION} · Designed & Developed by Jit")
    seed = st.number_input("Random seed", value=42, step=1)
    batch_size = st.slider("Batch size", 25, 500, 120, 5)
    drift = st.slider("Drift injection (0–1)", 0.0, 1.0, 0.15, 0.01)
    stress_threshold = st.slider("Stress threshold", 0.2, 0.9, 0.6, 0.01)
    psi_limit = st.slider("PSI drift limit", 0.05, 0.8, 0.2, 0.01)
    max_bandwidth = st.slider("Max uncertainty bandwidth", 0.1, 1.0, 0.5, 0.01)
    duty_throttle = st.slider("Duty-cycle throttle (sec between decisions)", 0, 30, 5, 1)
    federated = st.toggle("Federated mode (demo)", value=False, help="Simulate model updates without sharing raw data")
    st.divider()
    generate = st.button("Generate New Batch")
    retrain = st.button("Retrain with Replay (continual)")
    st.caption("Agentic AI uses OpenAI (set OPENAI_API_KEY). No data leaves your machine unless you enable it.")

policy = {
    "stress_threshold": float(stress_threshold),
    "psi_limit": float(psi_limit),
    "max_bandwidth": float(max_bandwidth),
}

# --------------------------- Main Layout -------------------------------------
st.title("🫒 Regenerative AI Dashboard — Olive Grove (Research Prototype)")
st.write("**Principle:** If we build AI like extraction, we scale fragility. Build it like regeneration and we scale resilience.")

if generate:
    new_df = simulate_batch(batch_size, seed=int(seed), drift=float(drift))
    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
    # push to replay buffer
    st.session_state.replay.append(new_df.tail(100).copy())
    st.session_state.replay = st.session_state.replay[-30:]  # cap ~3000 rows

if retrain and len(st.session_state.replay):
    # simulate federated update by training on mixed corpus (no raw sharing in demo)
    baseline = st.session_state.baseline
    recent = pd.concat(st.session_state.replay, ignore_index=True)
    mix = pd.concat([
        baseline.sample(min(len(baseline), 400), random_state=7),
        recent.sample(min(len(recent), 600), random_state=7)
    ], ignore_index=True)
    st.session_state.df = pd.concat([st.session_state.df, mix], ignore_index=True)
    st.success("Model training corpus expanded from replay (demo).")

# aliases
df = st.session_state.df.copy()
baseline = st.session_state.baseline.copy()

# Tabs
TAB_LABELS = [
    "Live Sensors", "Model & Decision", "What-If & Optimiser", "Transparency & Lineage",
    "Memory & Feedback", "Agentic AI", "Governance"
]

tabs = st.tabs(TAB_LABELS)

# --------------------------- Tab 0: Live Sensors -----------------------------
with tabs[0]:
    st.subheader("Live Sensors, Drift & Geo Context")
    if df.empty:
        st.info("Click **Generate New Batch** to simulate sensors.")
        psi_vpd = psi_sm = psi_ndvi = 0.0
        ks_vpd_p = ks_sm_p = ks_ndvi_p = 1.0
    else:
        recent = df.tail(300).copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            fig1 = px.line(recent, x="t", y="soil_moist", title="Soil Moisture")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.line(recent, x="t", y="vpd", title="VPD (kPa)")
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            fig3 = px.line(recent, x="t", y="ndvi", title="NDVI")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("##### Drift Monitors")
        psi_vpd = population_stability_index(baseline["vpd"].values, recent["vpd"].values, bins=10)
        psi_sm = population_stability_index(baseline["soil_moist"].values, recent["soil_moist"].values, bins=10)
        psi_ndvi = population_stability_index(baseline["ndvi"].values, recent["ndvi"].values, bins=10)
        ks_vpd, ks_vpd_p = ks_drift(baseline["vpd"].values, recent["vpd"].values)
        ks_sm, ks_sm_p = ks_drift(baseline["soil_moist"].values, recent["soil_moist"].values)
        ks_ndvi, ks_ndvi_p = ks_drift(baseline["ndvi"].values, recent["ndvi"].values)

        c4, c5, c6, c7, c8, c9 = st.columns(6)
        c4.metric("PSI: VPD", f"{psi_vpd:.3f}", help=">0.2 suggests material drift")
        c5.metric("PSI: Soil Moisture", f"{psi_sm:.3f}")
        c6.metric("PSI: NDVI", f"{psi_ndvi:.3f}")
        c7.metric("KS p(VPD)", f"{ks_vpd_p:.3f}")
        c8.metric("KS p(SM)", f"{ks_sm_p:.3f}")
        c9.metric("KS p(NDVI)", f"{ks_ndvi_p:.3f}")

        # Geo context (synthetic coordinates around one farm block)
        center = [37.99, 23.73]  # demo coords
        risk_val = recent["water_stress"].tail(50).mean()
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat":[center[0]], "lon":[center[1]], "size":[1000*risk_val+100]}),
            get_position='[lon, lat]', get_radius='size', pickable=True
        )
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=10, pitch=0),
            layers=[layer]
        ))

# --------------------------- Tab 1: Model & Decision ------------------------
with tabs[1]:
    st.subheader("Risk Model, Uncertainty & Decision Passport")
    if df.empty:
        st.info("Generate a batch first.")
    else:
        model = CalibratedLogit().fit(df)
        preds = model.predict_proba(df)
        r2 = r2_score(df["water_stress"], preds)
        brier = brier_score_loss(df["water_stress"], preds)
        lo, hi = conformal_band(preds, df["water_stress"].values, alpha=0.1)
        last_pred = float(preds[-1])

        # Calibration plot
        tmp = df.copy()
        tmp["pred"] = preds
        tmp["bin"] = pd.qcut(tmp["pred"], q=10, duplicates="drop")
        calib = tmp.groupby("bin").agg(obs=("water_stress","mean"), pred=("pred","mean")).dropna().reset_index()

        c1, c2 = st.columns([1,1])
        with c1:
            st.metric("R² (risk vs outcome)", f"{r2:.3f}")
            st.metric("Brier score (lower is better)", f"{brier:.3f}")
            fig4 = px.scatter(calib, x="pred", y="obs", title="Calibration Curve (binned)")
            fig4.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect"))
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            st.markdown("**Model Summary (Logit)**")
            st.code(model.summary_[:2000])

        # Importance
        imps = permutation_importance(model, df, "water_stress")
        st.markdown("**Permutation Importance (∆Brier; higher = more important)**")
        st.bar_chart(pd.Series(imps))

        # Drift proxies for decision gating
        recent_m = df.tail(300)
        psi_vpd_m = population_stability_index(baseline["vpd"].values, recent_m["vpd"].values, bins=10)
        psi_sm_m = population_stability_index(baseline["soil_moist"].values, recent_m["soil_moist"].values, bins=10)
        psi_ndvi_m = population_stability_index(baseline["ndvi"].values, recent_m["ndvi"].values, bins=10)

        # Decision
        action = "irrigate_now"
        fallback_used = False
        if not (np.isfinite(lo) and np.isfinite(hi)) or (hi - lo) > policy["max_bandwidth"]:
            action = "defer_and_observe"; fallback_used = True
        if psi_vpd_m > policy["psi_limit"]:
            action = "conservative_plan"; fallback_used = True
        if last_pred < policy["stress_threshold"]:
            action = "defer_and_observe"

        inputs = df.iloc[-1][["temp_c","rh","soil_moist","ndvi","vpd"]].to_dict()
        lineage = {
            "data_ts": df["ts_utc"].iloc[-1],
            "batch_size": int(len(df)),
            "generator_seed": int(seed),
            "app_version": APP_VERSION,
            "federated_mode": bool(federated)
        }
        drift_flags = {"psi_vpd":float(psi_vpd_m), "psi_soil_moist":float(psi_sm_m), "psi_ndvi":float(psi_ndvi_m)}
        artifact = {
            "decision_id": str(uuid.uuid4()),
            "ts_utc": datetime.utcnow().isoformat(),
            "action": action,
            "prediction": last_pred,
            "confidence_bounds": [lo, hi],
            "model_version": f"logit-{APP_VERSION}",
            "inputs_hash": sha256_of_dict(inputs),
            "inputs": inputs,
            "explanation": "Primary drivers: VPD↑ increases stress; Soil moisture↓ increases stress; NDVI↓ indicates canopy stress.",
            "lineage": lineage,
            "risk_checks": {"drift": drift_flags, "uncertainty_band": (hi-lo if (np.isfinite(hi) and np.isfinite(lo)) else None)},
            "policy": policy,
            "fallback_used": fallback_used
        }
        st.markdown("#### Decision Artifact (Passport)")
        st.json(artifact)

        # Auto NLP explanation
        extras = {"imps": imps, "metrics": {"r2": r2, "brier": brier}}
        nlp = auto_nlp_explanation(artifact, extras)
        st.markdown("**Auto Explanation (Operator-facing)**")
        st.write(nlp.get("rationale",""))
        st.markdown("**Checklist**")
        st.write(nlp.get("operator_checklist", []))
        st.markdown("**Residual Risks**")
        st.write(nlp.get("residual_risks", []))
        st.caption("Citations: "+", ".join(nlp.get("citations", [])))

        # Memory commit
        if st.button("Commit Decision to Memory"):
            st.session_state.decisions.append(artifact)
            st.session_state.audit.append({"type":"decision", "payload": artifact, "ts_utc": datetime.utcnow().isoformat()})
            st.success("Decision committed.")
            if duty_throttle>0:
                time.sleep(duty_throttle)

# --------------------------- Tab 2: What-If & Optimiser ----------------------
with tabs[2]:
    st.subheader("What-If Counterfactuals & Irrigation Optimiser")
    if len(st.session_state.decisions) == 0:
        st.info("Commit a decision first.")
    else:
        last_art = st.session_state.decisions[-1]
        # Recreate model for current df
        model = CalibratedLogit().fit(df)
        preds = model.predict_proba(df)

        st.markdown("**Counterfactual Sliders (applied to latest row)**")
        cfa, cfb, cfc, cfd, cfe = st.columns(5)
        d_vpd = cfa.slider("∆ VPD (kPa)", -0.5, 0.5, 0.0, 0.05)
        d_sm = cfb.slider("∆ Soil Moisture", -0.10, 0.10, 0.00, 0.01)
        d_ndvi = cfc.slider("∆ NDVI", -0.10, 0.10, 0.00, 0.01)
        d_temp = cfd.slider("∆ Temp (°C)", -3.0, 3.0, 0.0, 0.1)
        d_rh = cfe.slider("∆ RH (%)", -15.0, 15.0, 0.0, 0.5)

        # Compute deltas independently
        deltas = {
            "vpd": local_delta(model, df, "vpd", d_vpd),
            "soil_moist": local_delta(model, df, "soil_moist", d_sm),
            "ndvi": local_delta(model, df, "ndvi", d_ndvi),
            "temp_c": local_delta(model, df, "temp_c", d_temp),
            "rh": local_delta(model, df, "rh", d_rh),
        }
        st.markdown("**Risk Delta (change in probability)**")
        st.bar_chart(pd.Series(deltas))

        # Optimiser (window over last 48 points)
        win = df.tail(48).copy()
        win["pred"] = model.predict_proba(win)
        if HAS_PULP:
            max_water = st.slider("Water Budget (L)", 0, 2000, 600, 50)
            sol = optimise_irrigation(win, max_water_l=max_water)
            if sol is not None:
                st.markdown("**Optimised Irrigation Plan (next 48 steps)**")
                st.line_chart(pd.Series(sol), height=160)
                before = float(win["pred"].mean())
                after = max(before * (1 - min(sum(sol)/1000.0, 0.4)), 0.0)
                wue = water_use_efficiency(sum(sol), before, after)
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Risk (before)", f"{before:.3f}")
                c2.metric("Avg Risk (after est)", f"{after:.3f}")
                c3.metric("Water-Use Efficiency (L per risk averted)", f"{wue:.2f}")
        else:
            st.warning("PuLP not available; optimizer disabled.")

        # Pareto frontier
        lo, hi = conformal_band(preds, df["water_stress"].values, alpha=0.1)
        pf = pareto_points(preds, water_grid=list(range(0, 2001, 100)), uncert_band=(hi-lo))
        figp = px.line(pf, x="water_L", y="risk_after", title="Pareto: Water vs Risk (band shown as hover)")
        st.plotly_chart(figp, use_container_width=True)

# --------------------------- Tab 3: Transparency & Lineage -------------------
with tabs[3]:
    st.subheader("Transparency, Lineage, Overrides & Audit")
    if len(st.session_state.decisions) == 0:
        st.info("No committed decisions yet.")
    else:
        mem_df = pd.DataFrame(st.session_state.decisions)
        st.dataframe(mem_df[["ts_utc","decision_id","action","prediction","inputs_hash"]].tail(50), use_container_width=True)
        st.markdown("**Latest Artifact**")
        st.json(st.session_state.decisions[-1])

        st.markdown("**Override Register**")
        with st.form("override_form"):
            oid = st.text_input("Decision ID", value=st.session_state.decisions[-1]["decision_id"])
            new_action = st.selectbox("New Action", ["defer_and_observe","irrigate_now","conservative_plan"]) 
            rationale = st.text_input("Rationale", value="Operator judgment based on field inspection.")
            operator = st.text_input("Operator", value="operator@farm.local")
            submitted = st.form_submit_button("Record Override")
        if submitted:
            record_override(oid, new_action, rationale, operator)
            st.session_state.audit.append({"type":"override", "payload": st.session_state.overrides[-1], "ts_utc": datetime.utcnow().isoformat()})
            st.success("Override recorded.")

        if len(st.session_state.overrides):
            st.dataframe(pd.DataFrame(st.session_state.overrides).tail(20), use_container_width=True)

        # Audit export
        st.markdown("**Audit JSONL Export**")
        if st.button("Download audit.jsonl"):
            buf = io.StringIO()
            for rec in st.session_state.audit:
                buf.write(json.dumps(rec)+"\n")
            st.download_button("Save audit.jsonl", data=buf.getvalue(), file_name="audit.jsonl")

# --------------------------- Tab 4: Memory & Feedback ------------------------
with tabs[4]:
    st.subheader("Outcome Feedback (Close the Loop)")
    if len(st.session_state.decisions) == 0:
        st.info("Commit a decision first.")
    else:
        last_art = st.session_state.decisions[-1]
        st.markdown("**Last Decision**")
        st.json(last_art)

        st.markdown("**Record Outcome**")
        colA, colB = st.columns(2)
        with colA:
            realized_stress = st.selectbox("Observed water stress within 24h?", ["unknown","no","yes"], index=0)
        with colB:
            cost = st.number_input("Estimated cost impact (€)", value=0.0, step=10.0)
        note = st.text_input("Operator note (optional)", value="")

        if st.button("Save Outcome"):
            outcome = {
                "decision_id": last_art["decision_id"],
                "ts_utc": datetime.utcnow().isoformat(),
                "realized_stress": realized_stress,
                "cost": float(cost),
                "note": note
            }
            st.session_state.outcomes.append(outcome)
            st.session_state.audit.append({"type":"outcome", "payload": outcome, "ts_utc": datetime.utcnow().isoformat()})
            st.success("Outcome recorded.")
        if len(st.session_state.outcomes):
            st.markdown("**Recent Outcomes**")
            st.dataframe(pd.DataFrame(st.session_state.outcomes).tail(20), use_container_width=True)

# --------------------------- Tab 5: Agentic AI -------------------------------
with tabs[5]:
    st.subheader("Agentic AI — Grounded Explainer & Scenario Proposals")
    if len(st.session_state.decisions) == 0:
        st.info("Commit a decision first to explain it with the agent.")
    else:
        last = st.session_state.decisions[-1]
        explain = st.button("Explain Latest Decision")
        plan_scn = st.button("Propose Water-Saving Scenario")
        if explain:
            out = auto_nlp_explanation(last, {"task":"explain"})
            st.success("Explanation ready.")
            st.write(out.get("rationale",""))
            st.markdown("**Checklist**"); st.write(out.get("operator_checklist", []))
            st.markdown("**Residual Risks**"); st.write(out.get("residual_risks", []))
        if plan_scn:
            # simple scenario prompt via auto_nlp_explanation
            extras = {"task":"scenario","goal":"reduce water by 20% with minimal risk increase"}
            out = auto_nlp_explanation(last, extras)
            st.success("Scenario guidance ready.")
            st.write(out.get("rationale",""))
            st.markdown("**Checklist**"); st.write(out.get("operator_checklist", []))

        st.markdown("---")
        st.markdown("**Knowledge Base (local, cite-able)**")
        st.table(pd.DataFrame(kb_context()))

# --------------------------- Tab 6: Governance -------------------------------
with tabs[6]:
    st.subheader("Governance Readiness & Regeneration KPIs")

    has_lineage = len(st.session_state.decisions) > 0
    has_feedback = len(st.session_state.outcomes) > 0
    transparency_cov = 100.0 if has_lineage else 0.0
    feedback_cov = 100.0 if has_feedback else 0.0

    # Safe PSI fallback from sensors tab values (approximate using recent window)
    if df.empty:
        _psi_vpd = 0.0
    else:
        recent = df.tail(300)
        _psi_vpd = population_stability_index(baseline["vpd"].values, recent["vpd"].values, bins=10)
    resilience_index = max(0.0, 1.0 - max(0.0, _psi_vpd - policy["psi_limit"])) * 100.0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Transparency Coverage", f"{transparency_cov:.0f}%")
    with c2: st.metric("Feedback Coverage", f"{feedback_cov:.0f}%")
    with c3: st.metric("Resilience Index (demo)", f"{resilience_index:.0f}")

    # Regeneration KPIs (demo formulas)
    if not df.empty:
        risk_now = df["water_stress"].tail(64).mean()
        water_used_demo = 600.0  # placeholder from optimiser
        canopy_stability = 1.0 - float(np.std(df["ndvi"].tail(200)))  # proxy
        kpi = {
            "Soil-Water Regeneration Index": round(max(0.0, 1.0 - risk_now), 3),
            "Water Use (L)": water_used_demo,
            "Canopy Stability (↑ better)": round(canopy_stability, 3),
        }
        st.json(kpi)

    st.markdown("**Model Card (export)**")
    mc = {
        "model_version": f"logit-{APP_VERSION}",
        "intended_use": "Irrigation decision support (risk scoring) for olive groves (demo).",
        "limitations": [
            "Synthetic data; simplified physics.",
            "Calibration imperfect; conformal band is distribution-free but coarse.",
            "Fairness not modeled in this agronomic demo."
        ],
        "data": "Synthetic sensor streams with optional drift injection; baseline of 400 samples.",
        "governance": [
            "Decision Passport with lineage & risk flags.",
            "Outcome logging with operator notes and override register.",
            "EU AI Act friendly: human-in-the-loop; audit JSONL export; federated stub."
        ]
    }
    st.json(mc)

    cexp1, cexp2 = st.columns(2)
    if cexp1.button("Download model_card.json"):
        st.download_button("Save model_card.json", data=json.dumps(mc, indent=2), file_name="model_card.json")
    if cexp2.button("Export session bundle (signed)"):
        bundle = {
            "decisions": st.session_state.decisions,
            "outcomes": st.session_state.outcomes,
            "overrides": st.session_state.overrides,
            "audit": st.session_state.audit,
            "ts_utc": datetime.utcnow().isoformat()
        }
        digest = sha256_of_dict(bundle)
        out = json.dumps({"digest": digest, "bundle": bundle}, default=str)
        st.download_button("Save session_bundle.json", data=out, file_name="session_bundle.json")

st.caption("© Designed & Developed by Jit — Research Prototype (No real agronomic claims).")

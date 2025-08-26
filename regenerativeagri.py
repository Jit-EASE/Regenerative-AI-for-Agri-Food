# Re-generating the Streamlit app, requirements, and adding a README for Jit.

# regenerative_ai_streamlit.py
# -----------------------------------------------------------------------------
# Regenerative AI Dashboard (Olive Grove Scenario) — Synthetic Sensors + Agentic AI
# Designed & Developed by Jit (research prototype)
# -----------------------------------------------------------------------------
# Features
# - Synthetic sensor streams (soil moisture, temp, RH, NDVI, VPD)
# - Batch simulation with drift injection to test resilience
# - OLS model (statsmodels) + uncertainty & calibration diagnostics
# - Decision Artifact ("passport") with lineage, bounds, and risk checks
# - Memory layer: episodic log of decisions & outcomes in-session
# - Agentic AI: OpenAI-backed explainer using local policy context
# - Transparency tabs: lineage, drift, calibration, and governance readiness
# -----------------------------------------------------------------------------

import os, json, uuid, time, hashlib, math, textwrap
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Modeling & stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, brier_score_loss
import statsmodels.api as sm

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# Optional OpenAI (agentic AI explainer). Uses new SDK interface.
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False

# --------------------------- App Config --------------------------------------
st.set_page_config(
    page_title="Regenerative AI Dashboard — Olive Grove (Prototype)",
    page_icon="🫒",
    layout="wide"
)

APP_VERSION = "v0.4"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------- Utilities ---------------------------------------
def sha256_of_dict(d: Dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_vpd_celsius(temp_c: float, rh_pct: float) -> float:
    # Simplified VPD approximation (kPa) for demo purposes
    # es (saturation vapor pressure) in kPa (Tetens)
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    vpd = max(es - ea, 0.0)
    return float(vpd)

def simulate_batch(n:int, seed:int, drift:float=0.0) -> pd.DataFrame:
    """
    Generate synthetic olive-grove sensor data. Drift shifts means & adds noise.
    """
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
    # Higher VPD + lower soil moisture increases stress.
    logit = -2.0 + 4.0*(vpd - 0.9) - 5.0*(soil_moist - 0.25) - 1.5*(ndvi - 0.7) + rng.normal(0, 0.6, n)
    p_stress = 1/(1 + np.exp(-logit))
    # Binary outcome sampled for supervision
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

    # Add a timestamp for lineage
    now = datetime.utcnow().isoformat()
    df["ts_utc"] = now
    return df

def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins:int=10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    # Avoid degenerate cases
    if len(ref) < 2 or len(cur) < 2:
        return 0.0

    # Bin edges on reference
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, qs)
    # Deduplicate edges if necessary
    edges = np.unique(edges)

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    psi = np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps)))
    return float(psi)

def fit_ols_prob(df: pd.DataFrame, target_col="water_stress") -> Dict:
    """
    Fit OLS to predict water_stress ~ features as a proxy risk score.
    (In practice, a logistic model would be more appropriate; OLS is used
     to keep the demo simple and show p-values, R^2 updates, etc.)
    """
    features = ["vpd", "soil_moist", "ndvi", "temp_c", "rh"]
    X = df[features].copy()
    y = df[target_col].astype(float)

    # Add constant for statsmodels
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    preds = model.predict(X_sm)
    # Clamp to [0,1] as a naive probability-like score
    preds = np.clip(preds, 0, 1)

    # Diagnostics
    r2 = r2_score(y, preds)
    try:
        brier = brier_score_loss(y, preds)
    except Exception:
        brier = float("nan")

    # Bootstrap prediction interval for the last row
    last = X_sm.iloc[[-1]].values
    rng = np.random.default_rng(42)
    boot_preds = []
    for _ in range(200):
        # Simple bootstrap of residuals
        resid = (y - preds)
        resampled = rng.choice(resid, size=len(resid), replace=True)
        boot_y = preds + resampled
        try:
            bm = sm.OLS(boot_y, X_sm).fit()
            bp = float(np.clip(bm.predict(last)[0], 0, 1))
            boot_preds.append(bp)
        except Exception:
            continue
    if boot_preds:
        lo, hi = float(np.quantile(boot_preds, 0.05)), float(np.quantile(boot_preds, 0.95))
    else:
        lo, hi = float("nan"), float("nan")

    return {
        "model": model,
        "features": features,
        "preds": preds,
        "r2": float(r2),
        "brier": float(brier),
        "last_pi": (lo, hi),
        "last_pred": float(preds[-1]),
        "summary": str(model.summary())
    }

def make_decision_artifact(latest_row: pd.Series, model_info: Dict, lineage: Dict, drift_flags: Dict, policy: Dict) -> Dict:
    probability = float(model_info["last_pred"])
    lo, hi = model_info["last_pi"]
    action, threshold = ("irrigate_now", policy.get("stress_threshold", 0.6))
    fallback_used = False

    # Uncertainty gating
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) > policy.get("max_bandwidth", 0.5):
        action = "defer_and_observe"
        fallback_used = True

    # Drift gating
    if drift_flags.get("psi_vpd", 0.0) > policy.get("psi_limit", 0.2):
        action = "conservative_plan"
        fallback_used = True

    # Threshold gating
    if probability < threshold:
        action = "defer_and_observe"

    inputs = latest_row[["temp_c","rh","soil_moist","ndvi","vpd"]].to_dict()
    artifact = {
        "decision_id": str(uuid.uuid4()),
        "ts_utc": datetime.utcnow().isoformat(),
        "action": action,
        "prediction": probability,
        "confidence_bounds": [lo, hi],
        "model_version": f"ols-{APP_VERSION}",
        "inputs_hash": sha256_of_dict(inputs),
        "inputs": inputs,
        "explanation": "Primary drivers: VPD↑ increases stress; Soil moisture↓ increases stress; NDVI↓ indicates canopy stress.",
        "lineage": lineage,
        "risk_checks": {
            "drift": drift_flags,
            "uncertainty_band": (hi - lo if (np.isfinite(hi) and np.isfinite(lo)) else None),
        },
        "policy": policy,
        "fallback_used": fallback_used
    }
    return artifact

def kb_context() -> List[Dict[str,str]]:
    """
    Local, cite-able snippets for the agent to use.
    """
    return [
        {"id":"policy_irrigation", "title":"Irrigation Policy v1",
         "text":"Trigger irrigation when modeled water-stress risk > 0.6 and uncertainty band < 0.5, unless VPD drift PSI > 0.2."},
        {"id":"agronomy_vpd", "title":"Agronomy Note — VPD",
         "text":"Higher VPD implies stronger evaporative demand; leaves lose water faster, increasing irrigation need if soil water is low."},
        {"id":"governance_eu_ai", "title":"Governance — EU AI Act (demo)",
         "text":"For operational decision support, maintain lineage, human oversight, and model cards. Record overrides and rationale."}
    ]

def agentic_ai_explain(artifact: Dict, history: List[Dict], kb: List[Dict]) -> Dict:
    """
    Calls OpenAI (if available) to produce a structured explanation.
    """
    if not HAS_OPENAI:
        return {"error":"OpenAI SDK not available in this environment."}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"error":"Set OPENAI_API_KEY environment variable to enable agentic AI."}

    try:
        client = OpenAI(api_key=api_key)
        # Build compact, grounded prompt
        kb_block = "\n".join([f"[{x['id']}] {x['title']}: {x['text']}" for x in kb])
        history_slim = history[-5:] if history else []
        messages = [
            {"role":"system","content":textwrap.dedent("""\
                You are a Regenerative AI operator assistant. Explain decisions using the provided knowledge base (KB).
                RULES:
                - Only make claims grounded in the KB or the supplied artifact.
                - Cite KB items in square brackets like [policy_irrigation].
                - If uncertain, say what data would reduce uncertainty.
                - Provide a short action checklist for operators.
            """)},
            {"role":"user","content":json.dumps({
                "artifact": artifact,
                "recent_decisions": history_slim,
                "kb": kb_block
            }, ensure_ascii=False)}
        ]

        # Expect a concise JSON response
        schema_hint = textwrap.dedent("""
        Respond as JSON with keys:
        - "rationale": str
        - "citations": [str]
        - "operator_checklist": [str]
        - "residual_risks": [str]
        """)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages + [{"role":"system","content":schema_hint}],
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        content = completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"Agent error: {e}"}

# --------------------------- Session State -----------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "baseline" not in st.session_state:
    st.session_state.baseline = simulate_batch(400, seed=123, drift=0.0)
if "decisions" not in st.session_state:
    st.session_state.decisions = []  # episodic memory of artifacts
if "outcomes" not in st.session_state:
    st.session_state.outcomes = []   # feedback memory

# --------------------------- Sidebar -----------------------------------------
with st.sidebar:
    st.markdown("## 🫒 Regenerative AI — Controls")
    st.caption(f"Version: {APP_VERSION} · Designed & Developed by Jit")
    seed = st.number_input("Random seed", value=42, step=1)
    batch_size = st.slider("Batch size", min_value=25, max_value=500, value=120, step=5)
    drift = st.slider("Drift injection (0–1)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    stress_threshold = st.slider("Stress threshold", 0.2, 0.9, 0.6, 0.01)
    psi_limit = st.slider("PSI drift limit", 0.05, 0.8, 0.2, 0.01)
    max_bandwidth = st.slider("Max uncertainty bandwidth", 0.1, 1.0, 0.5, 0.01)
    st.divider()
    generate = st.button("Generate New Batch")
    st.caption("Note: Agentic AI uses OpenAI (set OPENAI_API_KEY). No data leaves your machine unless you enable it.")

policy = {
    "stress_threshold": float(stress_threshold),
    "psi_limit": float(psi_limit),
    "max_bandwidth": float(max_bandwidth),
}

# --------------------------- Main Layout -------------------------------------
st.title("🫒 Regenerative AI Dashboard — Olive Grove (Prototype)")
st.write("**Principle:** If we build AI like extraction, we scale fragility. Build it like regeneration and we scale resilience. This demo shows feedback loops, transparency, and graceful degradation.")

if generate:
    new_df = simulate_batch(batch_size, seed=int(seed), drift=float(drift))
    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

df = st.session_state.df.copy()
baseline = st.session_state.baseline.copy()

tabs = st.tabs(["Live Sensors", "Model & Decision", "Transparency & Lineage", "Memory & Feedback", "Agentic AI", "Governance"])

# --------------------------- Tab: Live Sensors -------------------------------
with tabs[0]:
    st.subheader("Live Sensors & Drift")
    if df.empty:
        st.info("Click **Generate New Batch** to simulate sensors.")
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

        st.markdown("##### Drift Monitors (PSI vs Baseline)")
        latest = recent
        psi_vpd = population_stability_index(baseline["vpd"].values, latest["vpd"].values, bins=10)
        psi_sm = population_stability_index(baseline["soil_moist"].values, latest["soil_moist"].values, bins=10)
        psi_ndvi = population_stability_index(baseline["ndvi"].values, latest["ndvi"].values, bins=10)

        st.metric("PSI: VPD", f"{psi_vpd:.3f}", help=">0.2 suggests material drift")
        st.metric("PSI: Soil Moisture", f"{psi_sm:.3f}")
        st.metric("PSI: NDVI", f"{psi_ndvi:.3f}")

# --------------------------- Tab: Model & Decision ---------------------------
with tabs[1]:
    st.subheader("Risk Model, Calibration & Decision")
    if df.empty:
        st.info("Generate a batch first.")
    else:
        model_info = fit_ols_prob(df)
        r2 = model_info["r2"]
        brier = model_info["brier"]
        lo, hi = model_info["last_pi"]
        last_pred = model_info["last_pred"]

        # Calibration (reliability) curve proxy
        tmp = df.copy()
        tmp["pred"] = model_info["preds"]
        tmp["bin"] = pd.qcut(tmp["pred"], q=10, duplicates="drop")
        calib = tmp.groupby("bin").agg(obs=("water_stress","mean"), pred=("pred","mean")).dropna().reset_index()

        c1, c2 = st.columns([1,1])
        with c1:
            st.metric("R² (risk score vs outcome)", f"{r2:.3f}")
            st.metric("Brier score (lower is better)", f"{brier:.3f}")
            fig4 = px.scatter(calib, x="pred", y="obs", title="Calibration Curve (binned)")
            fig4.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect"))
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            st.markdown("**Model Summary (OLS)**")
            st.code(model_info["summary"][:2000])

        # Decision Artifact
        drift_flags = {"psi_vpd": float(psi_vpd), "psi_soil_moist": float(psi_sm), "psi_ndvi": float(psi_ndvi)}
        lineage = {
            "data_ts": df["ts_utc"].iloc[-1],
            "batch_size": int(len(df)),
            "generator_seed": int(seed),
            "app_version": APP_VERSION
        }
        artifact = make_decision_artifact(df.iloc[-1], model_info, lineage, drift_flags, policy)

        st.markdown("#### Decision Artifact (Passport)")
        st.json(artifact)

        # Store in episodic memory
        if st.button("Commit Decision to Memory"):
            st.session_state.decisions.append(artifact)
            st.success("Decision committed.")

# --------------------------- Tab: Transparency & Lineage ---------------------
with tabs[2]:
    st.subheader("Transparency, Lineage & Logs")
    if len(st.session_state.decisions) == 0:
        st.info("No committed decisions yet.")
    else:
        mem_df = pd.DataFrame(st.session_state.decisions)
        st.dataframe(mem_df[["ts_utc","decision_id","action","prediction","inputs_hash"]].tail(50), use_container_width=True)
        st.markdown("**Latest Artifact**")
        st.json(st.session_state.decisions[-1])

        # Simple lineage visual (textual)
        st.markdown("**Lineage Snapshot**")
        ls = st.session_state.decisions[-1].get("lineage", {})
        st.write(pd.DataFrame([ls]))

# --------------------------- Tab: Memory & Feedback --------------------------
with tabs[3]:
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
            st.success("Outcome recorded.")
        if len(st.session_state.outcomes):
            st.markdown("**Recent Outcomes**")
            st.dataframe(pd.DataFrame(st.session_state.outcomes).tail(20), use_container_width=True)

# --------------------------- Tab: Agentic AI ---------------------------------
with tabs[4]:
    st.subheader("Agentic AI — Grounded Explainer")
    if len(st.session_state.decisions) == 0:
        st.info("Commit a decision first to explain it with the agent.")
    else:
        kb = kb_context()
        res = {}
        if st.button("Ask Agent to Explain Latest Decision"):
            res = agentic_ai_explain(st.session_state.decisions[-1], st.session_state.decisions, kb)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Agent response received.")
        if res:
            if "error" not in res:
                st.markdown("**Rationale**")
                st.write(res.get("rationale",""))
                st.markdown("**Citations**")
                st.write(res.get("citations", []))
                st.markdown("**Operator Checklist**")
                st.write(res.get("operator_checklist", []))
                st.markdown("**Residual Risks**")
                st.write(res.get("residual_risks", []))

        st.markdown("---")
        st.markdown("**Knowledge Base (local, cite-able)**")
        st.table(pd.DataFrame(kb))

# --------------------------- Tab: Governance ---------------------------------
with tabs[5]:
    st.subheader("Governance Readiness")
    # Simple scorecard to indicate coverage (demo only)
    has_lineage = len(st.session_state.decisions) > 0
    has_feedback = len(st.session_state.outcomes) > 0
    transparency_cov = 100.0 if has_lineage else 0.0
    feedback_cov = 100.0 if has_feedback else 0.0
    # Ensure psi_vpd is defined (fall back to 0 if Live Sensors hasn't run)
    try:
        _psi_vpd = float(psi_vpd)
    except Exception:
        _psi_vpd = 0.0
    resilience_index = max(0.0, 1.0 - max(0.0, _psi_vpd - policy["psi_limit"])) * 100.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Transparency Coverage", f"{transparency_cov:.0f}%")
    with c2:
        st.metric("Feedback Coverage", f"{feedback_cov:.0f}%")
    with c3:
        st.metric("Resilience Index (demo)", f"{resilience_index:.0f}")

    st.markdown("**Model Card (auto-generated excerpt)**")
    mc = {
        "model_version": f"ols-{APP_VERSION}",
        "intended_use": "Irrigation decision support (risk scoring) for olive groves (demo).",
        "limitations": [
            "Synthetic data; OLS used as proxy for logistic.",
            "Calibration imperfect; uncertainty estimated via residual bootstrap.",
            "No fairness attributes in the demo dataset."
        ],
        "data": "Synthetic sensor streams with optional drift injection; baseline of 400 samples.",
        "governance": [
            "Decision Passport with lineage & risk flags.",
            "Outcome logging with operator notes.",
            "EU AI Act friendly: human-in-the-loop possible; overrides should be recorded."
        ]
    }
    st.json(mc)

st.caption("© Designed & Developed by Jit — Research Prototype (No real agronomic claims).")


import streamlit as st
import sys, os
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import numpy as np

# Make the local "app" folder importable regardless of where you run Streamlit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import from local modules directly (no "app." prefix)
from drift import null_rates, ks_numeric_drift, psi_categorical, outliers_z, severity_score
from context import extract_pdf_text, ocr_image_text, find_signals
from explain import build_explanation

# =============================
# Utilities for Threshold/Cost
# =============================
def _metrics_at_threshold(y_true, y_proba, thresh=0.5):
    """
    Compute precision, recall, f1, auc, confusion matrix (tp/fp/tn/fn)
    for binary labels (0/1) at a given threshold on probabilities/scores.
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
    )
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= float(thresh)).astype(int)

    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def _expected_cost(cm: dict, c_fp=1.0, c_fn=5.0):
    """
    Expected cost = c_fp * FP + c_fn * FN (simple, editable).
    """
    return float(c_fp) * float(cm.get("fp", 0)) + float(c_fn) * float(cm.get("fn", 0))

def _rerun():
    # Backwards compatibility with older Streamlit
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Multimodal Data Drift Detector", layout="wide")
st.title("Multimodal Data Drift Detector")

st.write(
    """Compare **Baseline** vs **Current** data to detect drift and anomalies,
then ingest **release notes (PDF/text)** and **dashboard screenshot (image)**
to explain **why** things changed.

> New: Optional **Cost & Threshold Sandbox** (upload labels+scores OR use a numeric column) + **reproducible config/report downloads**.
"""
)

# -----------------------------
# Session state defaults
# -----------------------------
if "analysis_active" not in st.session_state:
    st.session_state.analysis_active = False
if "analysis" not in st.session_state:
    st.session_state.analysis = {}      # will store results + narrative
if "chat_feed" not in st.session_state:
    st.session_state.chat_feed = []     # list of {ts, role, text}
if "sandbox" not in st.session_state:
    st.session_state.sandbox = {}       # store threshold/cost sandbox results

CHAT_LOG_PATH = Path("simulated_chat.jsonl")

# -----------------------------
# Simulated Chat helpers
# -----------------------------
def post_to_sim_chat(text: str, role: str = "assistant"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = {"ts": ts, "role": role, "text": text}
    st.session_state.chat_feed.append(msg)
    try:
        with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _supports_chat_message() -> bool:
    return hasattr(st, "chat_message")

def render_sim_chat(feed):
    st.subheader("Simulated Chat")
    st.caption("No external chat required. Messages are also saved to simulated_chat.jsonl.")
    for m in feed[-200:]:
        role = m.get("role", "assistant")
        header = f"[{m.get('ts','')}] {role}"
        if _supports_chat_message():
            with st.chat_message(role):
                st.markdown(f"{header}\n\n{m.get('text','')}")
        else:
            st.markdown(f"**{header}**\n\n{m.get('text','')}")

# -----------------------------
# Sidebar: data + context input
# -----------------------------
with st.sidebar:
    st.header("Data")
    use_samples = st.toggle("Use sample data", value=True)
    if use_samples:
        base = pd.read_csv("sample_data/transactions_baseline.csv")
        cur  = pd.read_csv("sample_data/transactions_current.csv")
    else:
        b = st.file_uploader("Baseline CSV", type=["csv"], key="b")
        c = st.file_uploader("Current CSV", type=["csv"], key="c")
        base = pd.read_csv(b) if b else None
        cur  = pd.read_csv(c) if c else None

    st.header("Context (optional)")
    if st.toggle("Use sample context", value=True):
        # Gracefully load sample PDF first, then TXT, else skip
        context_blob = None   # tuple: (name, bytes)
        screenshot_blob = None

        if os.path.exists("sample_data/release_notes.pdf"):
            context_blob = ("release_notes.pdf", open("sample_data/release_notes.pdf", "rb").read())
        elif os.path.exists("sample_data/release_notes.txt"):
            context_blob = ("release_notes.txt", open("sample_data/release_notes.txt", "rb").read())

        if os.path.exists("sample_data/dashboard_eur.png"):
            screenshot_blob = ("dashboard_eur.png", open("sample_data/dashboard_eur.png", "rb").read())
    else:
        pdf_up = st.file_uploader("Release notes (PDF or txt)", type=["pdf", "txt"], key="pdf")
        img_up = st.file_uploader("Dashboard screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"], key="img")
        context_blob = (pdf_up.name, pdf_up.read()) if pdf_up else None
        screenshot_blob = (img_up.name, img_up.read()) if img_up else None

# -----------------------------
# Guard: need two datasets
# -----------------------------
if base is None or cur is None:
    st.info("Load data to continue.")
    st.stop()

# -----------------------------
# Preview
# -----------------------------
st.subheader("1) Preview")
c1, c2 = st.columns(2)
with c1:
    st.write("Baseline")
    st.dataframe(base.head(10))
with c2:
    st.write("Current")
    st.dataframe(cur.head(10))

# -----------------------------
# Run or re-display analysis
# -----------------------------
st.subheader("2) Drift & Anomaly Checks")

# Button to compute and persist results
if st.button("Run Analysis", type="primary", key="run_analysis_btn"):
    # Prepare columns
    num_cols = [c for c in cur.columns if pd.api.types.is_numeric_dtype(cur[c])]
    cat_cols = [c for c in cur.columns if c not in num_cols and cur[c].nunique() <= 50]

    # Compute checks
    nulls_cur = null_rates(cur)
    drift_num = ks_numeric_drift(base, cur, num_cols)
    drift_cat = psi_categorical(base, cur, cat_cols)
    out_cur   = outliers_z(cur, num_cols, z=3.0)

    # Severity
    sev, reasons = severity_score(nulls_cur, drift_num, drift_cat, out_cur)

    # --- Context ingestion now; store extracted text so it persists ---
    texts = []
    extracted_release = ""
    ocr_text = ""

    if context_blob:
        name, data = context_blob
        if name.lower().endswith(".pdf"):
            tmp = "tmp_release.pdf"
            with open(tmp, "wb") as f:
                f.write(data)
            extracted_release = extract_pdf_text(tmp) or ""
        else:
            extracted_release = data.decode("utf-8", errors="ignore") or ""
        texts.append(extracted_release)

    screenshot_path = None
    if screenshot_blob:
        name, data = screenshot_blob
        screenshot_path = "tmp_dash.png"
        with open(screenshot_path, "wb") as f:
            f.write(data)
        ocr_text = ocr_image_text(screenshot_path) or ""
        texts.append(ocr_text)

    signals = find_signals(texts)

    # Narrative
    narrative = build_explanation(sev, reasons, drift_num, drift_cat, signals)

    # Persist everything so replies don't collapse the view
    st.session_state.analysis_active = True
    st.session_state.analysis = {
        "sev": sev, "reasons": reasons,
        "nulls_cur": nulls_cur, "drift_num": drift_num, "drift_cat": drift_cat, "out_cur": out_cur,
        "extracted_release": extracted_release, "ocr_text": ocr_text, "screenshot_path": screenshot_path,
        "signals": signals, "narrative": narrative
    }
    _rerun()

# -----------------------------
# Display persisted analysis (if any)
# -----------------------------
if st.session_state.analysis_active and st.session_state.analysis:
    a = st.session_state.analysis

    st.markdown(f"**Overall Status:** {a['sev'].upper()}")
    if a["reasons"]:
        st.write("Why:")
        for r in a["reasons"]:
            st.write("- ", r)

    with st.expander("Null % by column"):
        st.dataframe(
            pd.DataFrame.from_dict(a["nulls_cur"], orient="index", columns=["null_ratio"])
              .sort_values("null_ratio", ascending=False)
        )

    with st.expander("Numeric drift (KS & mean change)"):
        st.dataframe(pd.DataFrame(a["drift_num"]).T)

    with st.expander("Categorical drift (PSI)"):
        st.dataframe(pd.DataFrame({k: v['psi'] for k, v in a["drift_cat"].items()}, index=["psi"]).T)

    with st.expander("Outliers (z>3)"):
        st.json({k: v['count'] for k, v in a["out_cur"].items()})

    # -------------------------
    # Context: show what we extracted
    # -------------------------
    st.subheader("3) Context: Release Notes & Dashboard Screenshot")
    if a["extracted_release"]:
        st.write("Extracted from release notes:")
        txt = a["extracted_release"]
        st.code((txt[:1000] + ("..." if len(txt) > 1000 else "")) or "(no text detected)")
    if a["screenshot_path"]:
        st.image(a["screenshot_path"], caption="Dashboard screenshot")
    if a["ocr_text"]:
        st.write("OCR from screenshot:")
        st.code((a["ocr_text"][:500] + ("..." if len(a["ocr_text"]) > 500 else "")) or "(no text detected)")

    st.write("Detected signals:", ", ".join([k for k, v in a["signals"].items() if v]) or "(none)")

    # -------------------------
    # Explanation
    # -------------------------
    st.subheader("4) Explanation & Suggested Actions")
    st.code(a["narrative"])

    # -------------------------
    # Simulated Chat (alert + replies) — persists across reruns
    # -------------------------
    with st.expander("Simulated Chat (demo without Slack/Discord)", expanded=True):

        # Send the current narrative as an "assistant" alert
        send_alert = st.button("Send alert to Simulated Chat", key="send_alert_btn")
        if send_alert:
            status = a["sev"].upper() if isinstance(a["sev"], str) else str(a["sev"])
            alert_text = f"Data Drift Alert - {status}\n\n{a['narrative']}"
            post_to_sim_chat(alert_text, role="assistant")
            _rerun()

        # Reply form (user message)
        with st.form("sim_reply_form", clear_on_submit=True):
            user_msg = st.text_input("Type a reply (simulated teammate):", key="sim_reply_input")
            submitted = st.form_submit_button("Send reply")
            if submitted and user_msg.strip():
                post_to_sim_chat(user_msg.strip(), role="user")
                _rerun()

        # Render feed
        render_sim_chat(st.session_state.chat_feed)

        # Utilities
        colA, colB = st.columns(2)
        with colA:
            if st.button("Clear simulated chat", key="clear_chat_btn"):
                st.session_state.chat_feed = []
                try:
                    if CHAT_LOG_PATH.exists():
                        CHAT_LOG_PATH.unlink()
                except Exception:
                    pass
                _rerun()
        with colB:
            if CHAT_LOG_PATH.exists():
                st.download_button(
                    "Download chat log (JSONL)",
                    data=CHAT_LOG_PATH.read_text(encoding="utf-8"),
                    file_name="simulated_chat.jsonl",
                    mime="application/x-ndjson",
                    key="dl_chat_btn",
                )
else:
    st.info("Click Run Analysis to compute drift and ingest context.")

# ===========================================================
# 5) (NEW) Cost & Threshold Sandbox + Reproducibility
# ===========================================================
st.markdown("---")
st.subheader("5) Optional: Cost & Threshold Sandbox (with Reproducibility)")

st.write(
    """Use this to **optimize a decision threshold** and **evaluate cost trade-offs**.
You can either:
1) Upload a small CSV with columns **`y_true` (0/1)** and **`y_proba`** (probability/score), **or**
2) Select a numeric column from **Current** to create a quick **pseudo-score** (min-max scaled) plus provide/derive a `y_true` column if you have one.
"""
)

tab1, tab2 = st.tabs(["Upload labels+scores CSV", "Use Current data (pseudo-score)"])

with tab1:
    upl = st.file_uploader("Upload CSV with y_true (0/1) and y_proba columns", type=["csv"], key="upl_scores")
    if upl is not None:
        df_scores = pd.read_csv(upl)
        y_col = st.selectbox("Label column (0/1)", options=list(df_scores.columns), index=0)
        p_col = st.selectbox("Probability/Score column", options=list(df_scores.columns), index=min(1, len(df_scores.columns)-1))
        c1, c2, c3 = st.columns(3)
        with c1:
            thresh = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)
        with c2:
            c_fn = st.number_input("Cost of False Negative", value=5.0, step=0.5)
        with c3:
            c_fp = st.number_input("Cost of False Positive", value=1.0, step=0.5)

        if st.button("Evaluate threshold on uploaded scores", key="eval_uploaded"):
            y_true = df_scores[y_col].values
            y_proba = df_scores[p_col].values
            m = _metrics_at_threshold(y_true, y_proba, thresh=thresh)
            cost = _expected_cost(m, c_fp=c_fp, c_fn=c_fn)

            # quick sweep suggestion
            ts = np.linspace(0.05, 0.95, 19)
            best_t, best_cost = None, float("inf")
            for t in ts:
                mm = _metrics_at_threshold(y_true, y_proba, thresh=t)
                cc = _expected_cost(mm, c_fp=c_fp, c_fn=c_fn)
                if cc < best_cost:
                    best_cost, best_t = cc, t

            st.json({**m, "expected_cost": cost})
            st.success(f"Suggested threshold to minimize expected cost: **{best_t:.2f}** (≈ {best_cost:.3f})")

            # Save to session for reproducibility
            st.session_state.sandbox = {
                "mode": "uploaded_scores",
                "threshold": float(thresh),
                "costs": {"FP": float(c_fp), "FN": float(c_fn)},
                "metrics_at_threshold": m,
                "expected_cost": float(cost),
                "suggested_threshold": float(best_t),
                "suggested_cost": float(best_cost),
                "y_col": y_col,
                "p_col": p_col,
            }

with tab2:
    st.write("Build a pseudo-score from a numeric column of **Current** (min-max scaled). If you also have a binary label column in Current, select it to compute metrics; otherwise, you’ll only see the score distribution.")
    num_cols_cur = [c for c in cur.columns if pd.api.types.is_numeric_dtype(cur[c])]
    if len(num_cols_cur) == 0:
        st.info("No numeric columns detected in Current.")
    else:
        score_col = st.selectbox("Numeric column to use as pseudo-score", options=num_cols_cur)
        label_candidates = [c for c in cur.columns if cur[c].dropna().nunique() == 2]
        label_col = st.selectbox("Optional label column (0/1) in Current", options=["(none)"] + label_candidates, index=0)

        c1, c2, c3 = st.columns(3)
        with c1:
            thresh2 = st.slider("Decision threshold (on pseudo-score)", 0.05, 0.95, 0.5, 0.01, key="thresh2")
        with c2:
            c_fn2 = st.number_input("Cost of False Negative", value=5.0, step=0.5, key="cfn2")
        with c3:
            c_fp2 = st.number_input("Cost of False Positive", value=1.0, step=0.5, key="cfp2")

        # Build pseudo score
        s = cur[score_col].astype(float)
        smin, smax = np.nanmin(s), np.nanmax(s)
        if smax > smin:
            y_proba2 = (s - smin) / (smax - smin)
        else:
            y_proba2 = np.zeros_like(s, dtype=float)

        if label_col != "(none)":
            y_true2 = cur[label_col].astype(int).values
            if st.button("Evaluate threshold on pseudo-score", key="eval_pseudo"):
                m2 = _metrics_at_threshold(y_true2, y_proba2, thresh=thresh2)
                cost2 = _expected_cost(m2, c_fp=c_fp2, c_fn=c_fn2)

                # quick sweep suggestion
                ts2 = np.linspace(0.05, 0.95, 19)
                best_t2, best_cost2 = None, float("inf")
                for t in ts2:
                    mm2 = _metrics_at_threshold(y_true2, y_proba2, thresh=t)
                    cc2 = _expected_cost(mm2, c_fp=c_fp2, c_fn=c_fn2)
                    if cc2 < best_cost2:
                        best_cost2, best_t2 = cc2, t

                st.json({**m2, "expected_cost": cost2})
                st.success(f"Suggested threshold (pseudo-score): **{best_t2:.2f}** (≈ {best_cost2:.3f})")

                st.session_state.sandbox = {
                    "mode": "pseudo_score",
                    "threshold": float(thresh2),
                    "costs": {"FP": float(c_fp2), "FN": float(c_fn2)},
                    "metrics_at_threshold": m2,
                    "expected_cost": float(cost2),
                    "suggested_threshold": float(best_t2),
                    "suggested_cost": float(best_cost2),
                    "score_col": score_col,
                    "label_col": label_col,
                }
        else:
            st.info("No label selected — showing only the constructed pseudo-score distribution.")
            # Display basic distribution
            st.write(
                pd.Series(y_proba2).describe().to_frame("pseudo_score_stats")
            )

# -----------------------------
# Reproducibility: Save report + config
# -----------------------------
st.markdown("### Reproducibility Downloads")

# Build a concise report of the full drift run + (optional) sandbox
report = {
    "timestamp": datetime.now().isoformat(),
    "drift_run": st.session_state.analysis if st.session_state.analysis_active else "(not run)",
    "sandbox": st.session_state.sandbox or "(empty)",
}

config = {
    "data_config": {
        "use_samples": bool(use_samples),
        "baseline_path": "sample_data/transactions_baseline.csv" if use_samples else "(uploaded)",
        "current_path":  "sample_data/transactions_current.csv"  if use_samples else "(uploaded)",
    },
    "context_config": {
        "use_sample_context": True if ('release_notes.pdf' in str(context_blob or []) or 'release_notes.txt' in str(context_blob or [])) else False
    },
    "sandbox_config": st.session_state.sandbox or "(empty)",
}

st.download_button(
    "⬇️ Download report.json",
    data=json.dumps(report, indent=2).encode("utf-8"),
    file_name="report.json",
    mime="application/json",
)

st.download_button(
    "⬇️ Download config.json",
    data=json.dumps(config, indent=2).encode("utf-8"),
    file_name="config.json",
    mime="application/json",
)

st.caption(
    "Stretch features added: Threshold slider, cost-sensitive metrics, and reproducibility (report/config). "
    "Upload a labels+scores CSV for best results, or use a numeric column from Current to simulate a score."
)

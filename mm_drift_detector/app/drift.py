import pandas as pd
import numpy as np
from scipy import stats

def null_rates(df: pd.DataFrame):
    n = len(df)
    return {c: float(df[c].isna().sum())/n for c in df.columns}

def ks_numeric_drift(baseline: pd.DataFrame, current: pd.DataFrame, cols):
    res = {}
    for c in cols:
        a = pd.to_numeric(baseline[c], errors='coerce').dropna()
        b = pd.to_numeric(current[c], errors='coerce').dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        stat, p = stats.ks_2samp(a, b)
        mean_change = (b.mean() - a.mean()) / (abs(a.mean()) + 1e-9)
        res[c] = {"ks": float(stat), "p": float(p), "mean_change": float(mean_change)}
    return res

def psi_categorical(baseline: pd.DataFrame, current: pd.DataFrame, cols):
    res = {}
    for c in cols:
        a = baseline[c].fillna("__NA__")
        b = current[c].fillna("__NA__")
        va = a.value_counts(normalize=True)
        vb = b.value_counts(normalize=True)
        keys = sorted(set(va.index) | set(vb.index))
        psi = 0.0
        details = []
        for k in keys:
            pa = max(va.get(k,0.0), 1e-6)
            pb = max(vb.get(k,0.0), 1e-6)
            psi_part = (pb - pa) * np.log(pb/pa)
            psi += psi_part
            details.append({"category": str(k), "base": float(pa), "cur": float(pb)})
        res[c] = {"psi": float(psi), "details": details}
    return res

def outliers_z(df: pd.DataFrame, numeric_cols, z=3.0):
    res = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(s) < 30:
            continue
        zs = np.abs((s - s.mean()) / (s.std(ddof=0) + 1e-9))
        cnt = int((zs > z).sum())
        res[c] = {"count": int(cnt)}
    return res

def severity_score(nulls, drift_num, drift_cat, outliers, thresholds=None):
    th = thresholds or {
        "null_warn": 0.05, "null_crit": 0.15,
        "ks_warn": 0.15,  "ks_crit": 0.25,
        "psi_warn": 0.1,  "psi_crit": 0.25,
        "out_warn": 10,   "out_crit": 50
    }
    sev = "green"; reasons = []
    high_nulls = [c for c,r in nulls.items() if r >= th["null_crit"]]
    med_nulls  = [c for c,r in nulls.items() if th["null_warn"] <= r < th["null_crit"]]
    if high_nulls: sev="red"; reasons.append(f"High nulls: {', '.join(high_nulls)}")
    elif med_nulls: sev="yellow"; reasons.append(f"Moderate nulls: {', '.join(med_nulls)}")
    high_ks = [c for c,v in drift_num.items() if v.get("ks",0) >= th["ks_crit"]]
    med_ks  = [c for c,v in drift_num.items() if th["ks_warn"] <= v.get("ks",0) < th["ks_crit"]]
    if high_ks: sev="red"; reasons.append(f"Strong numeric drift: {', '.join(high_ks)}")
    elif med_ks and sev!="red": sev="yellow"; reasons.append(f"Possible numeric drift: {', '.join(med_ks)}")
    high_psi = [c for c,v in drift_cat.items() if v.get("psi",0) >= th["psi_crit"]]
    med_psi  = [c for c,v in drift_cat.items() if th["psi_warn"] <= v.get("psi",0) < th["psi_crit"]]
    if high_psi: sev="red"; reasons.append(f"Strong categorical drift: {', '.join(high_psi)}")
    elif med_psi and sev!="red": sev="yellow"; reasons.append(f"Possible categorical drift: {', '.join(med_psi)}")
    high_out = [c for c,v in outliers.items() if v.get("count",0) >= th["out_crit"]]
    med_out  = [c for c,v in outliers.items() if th["out_warn"] <= v.get("count",0) < th["out_crit"]]
    if high_out: sev="red"; reasons.append(f"Many outliers: {', '.join(high_out)}")
    elif med_out and sev!="red": sev="yellow"; reasons.append(f"Some outliers: {', '.join(med_out)}")
    return sev, reasons

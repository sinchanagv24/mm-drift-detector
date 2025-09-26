TEMPLATE = """\
Overall status: {sev}

Key reasons:
{reasons_bullets}

Top numeric drift:
{num_drift}

Top categorical drift:
{cat_drift}

Context signals:
{signals}

Hypothesis:
{hypothesis}

Recommended actions:
{actions}
"""

def format_drift_num(drift):
    if not drift:
        return "- None"
    items = sorted(drift.items(), key=lambda x: x[1].get("ks",0), reverse=True)[:3]
    return "\n".join([f"- {c}: KS={v['ks']:.2f}, meanΔ={v['mean_change']*100:.1f}%" for c,v in items])

def format_drift_cat(drift):
    if not drift:
        return "- None"
    items = sorted(drift.items(), key=lambda x: x[1].get("psi",0), reverse=True)[:3]
    return "\n".join([f"- {c}: PSI={v['psi']:.2f}" for c,v in items])

def build_explanation(sev, reasons, drift_numeric, drift_categorical, signals):
    reasons_bullets = "\n".join([f"- {r}" for r in reasons]) if reasons else "- No major issues"
    num_drift = format_drift_num(drift_numeric)
    cat_drift = format_drift_cat(drift_categorical)

    hyp = []
    if 'amount' in drift_numeric:
        ks = drift_numeric['amount'].get('ks',0)
        mean_chg = drift_numeric['amount'].get('mean_change',0)
        if ks > 0.15 and abs(mean_chg) > 0.05 and (signals.get('currency_eur') or signals.get('conversion_note')):
            direction = "increase" if mean_chg > 0 else "decrease"
            hyp.append(f"Detected {direction} in 'amount' with KS={ks:.2f}. Release/dashboard text mentions currency (EUR) / conversion. Likely currency switch without proper conversion.")
    if signals.get('api_v2'):
        hyp.append("API v2 mentioned — upstream schema/semantics could have changed.")
    if not hyp:
        hyp.append("Investigate upstream changes in recent commits/releases; correlate drifted columns to ETL steps and source APIs.")

    actions = [
        "Confirm expected currency and apply consistent conversion before aggregation.",
        "Backfill impacted metrics for the affected window if needed.",
        "Add a data contract test for 'amount' units and acceptable drift bounds.",
        "Notify dashboard owners to annotate currency change."
    ]
    return TEMPLATE.format(
        sev=sev.upper(),
        reasons_bullets=reasons_bullets,
        num_drift=num_drift,
        cat_drift=cat_drift,
        signals=", ".join([k for k,v in signals.items() if v]) or "None detected",
        hypothesis=" ".join(hyp),
        actions="\n".join([f"- {a}" for a in actions])
    )

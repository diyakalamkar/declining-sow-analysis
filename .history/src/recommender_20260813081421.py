import pandas as pd

OFFER_MAP = {
    "Silent Attrition (Card Active, Unused)": "Urgent win-back: direct outreach + reactivation bonus (e.g. one-time 5% cashback on next purchase) — this cohort abandoned the card without closing it, so standard engagement triggers won't reach them; needs proactive contact",
    "High-Spend Leakers": "Category-expansion challenge: bonus 4% cashback on Apparel/Travel for 60 days if XYZ_Inc SoW rises 15pp",
    "Cash/UPI Migrators": "Instant redemption pilot (remove 1-month statement lag) + fee-free EMI nudge",
    "Wallet Cannibalized": "Cross-link ABC card as default funding source in XYZ Wallet checkout + rewards messaging",
    "New & Unengaged": "First-90-days activation bonus + category-specific onboarding nudge",
    "Declining Loyalists": "Win-back: boosted rate for 90 days + proactive outreach (highest CLV at risk)",
}

def run(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    feats["recommended_offer"] = feats["micro_segment_name"].map(OFFER_MAP).fillna("Standard engagement nurture")
    feats["recoverable_spend"] = feats["total_lifetime_spend"] * (1 - feats["avg_SoW"])
    feats["priority_score"] = feats["sow_risk_score"] * feats["recoverable_spend"]
    return feats.sort_values("priority_score", ascending=False)
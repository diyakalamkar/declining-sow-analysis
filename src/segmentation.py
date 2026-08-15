import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

CLUSTER_COLS = ["avg_SoW", "sow_trend", "recency_days",
                 "txn_count", "tenure_months", "credit_utilization_proxy",
                 "pct_Grocery", "pct_Electronics", "pct_Apparel", "pct_Travel",
                 "pctpay_Other Bank Credit Card", "pctpay_Cash/UPI", "pctpay_XYZ Wallet"]

SEGMENT_NAMES = {
    0: "High-Spend Leakers",
    1: "Cash/UPI Migrators",
    2: "Wallet Cannibalized",
    3: "New & Unengaged",
    4: "Declining Loyalists",
}

def sow_tier(row):
    if row["avg_SoW"] >= 0.6: return "High SoW - Loyal"
    if row["avg_SoW"] >= 0.3: return "Moderate SoW - Growable"
    if row["avg_SoW"] > 0:    return "Low SoW - At Risk"
    return "Dormant/Never Used"

def run(feats: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feats = feats.copy()
    feats["sow_tier"] = feats.apply(sow_tier, axis=1)

    cols = [c for c in CLUSTER_COLS if c in feats.columns]
    target = feats[feats["sow_tier"].isin(["Low SoW - At Risk", "Moderate SoW - Growable"])].copy()

    X_raw = target[cols].fillna(0).copy()
    for skewed_col in ["total_lifetime_spend", "recency_days", "credit_utilization_proxy"]:
        if skewed_col in X_raw.columns:
            X_raw[skewed_col] = np.log1p(X_raw[skewed_col].clip(lower=0))

    for c in X_raw.columns:
        lo, hi = X_raw[c].quantile([0.01, 0.99])
        X_raw[c] = X_raw[c].clip(lo, hi)

    X = RobustScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    target["micro_segment"] = km.fit_predict(X)

    order = target.groupby("micro_segment")["avg_SoW"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    target["micro_segment"] = target["micro_segment"].map(remap)
    target["micro_segment_name"] = target["micro_segment"].map(SEGMENT_NAMES).fillna("Unclassified")

    feats = feats.merge(
        target[["Customer_ID", "micro_segment", "micro_segment_name"]],
        on="Customer_ID", how="left"
    )
    feats["micro_segment_name"] = feats["micro_segment_name"].fillna("Not Targeted (High/Dormant SoW)")

    # silent attrition overrides whatever cluster they landed in — it's the priority label
    silent_mask = feats["is_silent_attrition"] == 1
    feats.loc[silent_mask, "micro_segment_name"] = "Silent Attrition (Card Active, Unused)"

    # diagnostic print, safe to keep — helps you sanity check cluster balance each run
    print("\nSegment sizes:")
    print(feats["micro_segment_name"].value_counts())

    return feats
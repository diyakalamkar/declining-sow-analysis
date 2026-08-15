import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

CLUSTER_COLS = ["avg_SoW", "sow_trend", "total_lifetime_spend", "recency_days",
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

def run(feats: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    feats = feats.copy()
    feats["sow_tier"] = feats.apply(sow_tier, axis=1)

    cols = [c for c in CLUSTER_COLS if c in feats.columns]
    target = feats[feats["sow_tier"].isin(["Low SoW - At Risk", "Moderate SoW - Growable"])].copy()

    X_raw = target[cols].fillna(0).copy()
    # log-transform heavy-tailed spend/recency columns to stop them dominating distance
    for skewed_col in ["total_lifetime_spend", "recency_days", "credit_utilization_proxy"]:
        if skewed_col in X_raw.columns:
            X_raw[skewed_col] = np.log1p(X_raw[skewed_col].clip(lower=0))

    X = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    target["micro_segment"] = km.fit_predict(X)

# def run(feats: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
#     feats = feats.copy()
#     feats["sow_tier"] = feats.apply(sow_tier, axis=1)

#     cols = [c for c in CLUSTER_COLS if c in feats.columns]
#     target = feats[feats["sow_tier"].isin(["Low SoW - At Risk", "Moderate SoW - Growable"])].copy()

#     X = StandardScaler().fit_transform(target[cols].fillna(0))
#     # km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
#     # target["micro_segment"] = km.labels_

#     # gpt
#     km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
#     target["micro_segment"] = km.fit_predict(X)

    # order clusters by avg_SoW ascending so naming is roughly consistent run-to-run
    order = target.groupby("micro_segment")["avg_SoW"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    target["micro_segment"] = target["micro_segment"].map(remap)
    target["micro_segment_name"] = target["micro_segment"].map(SEGMENT_NAMES).fillna("Unclassified")

    feats = feats.merge(
        target[["Customer_ID", "micro_segment", "micro_segment_name"]],
        on="Customer_ID", how="left"
    )
    feats["micro_segment_name"] = feats["micro_segment_name"].fillna("Not Targeted (High/Dormant SoW)")
    return feats
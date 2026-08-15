import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# FEATURE_COLS = ["avg_SoW", "sow_trend", "total_lifetime_spend", "recency_days",
#                  "txn_count", "tenure_months", "credit_utilization_proxy",
#                  "Age", "Credit_Card_Limit", "Credit_Card_APR",
#                  "pct_Grocery", "pct_Electronics", "pct_Apparel", "pct_Travel"]

# FEATURE_COLS = ["avg_SoW", "early_SoW", "total_lifetime_spend", "recency_days",
#                  "txn_count", "tenure_months", "credit_utilization_proxy",
#                  "Age", "Credit_Card_Limit", "Credit_Card_APR",
#                  "pct_Grocery", "pct_Electronics", "pct_Apparel", "pct_Travel"]

FEATURE_COLS = ["early_SoW", "total_lifetime_spend", "recency_days",
                 "txn_count", "tenure_months", "credit_utilization_proxy",
                 "Age", "Credit_Card_Limit", "Credit_Card_APR",
                 "pct_Grocery", "pct_Electronics", "pct_Apparel", "pct_Travel",
                 "pctpay_Other Bank Credit Card", "pctpay_Cash/UPI", "pctpay_XYZ Wallet"]

def run(feats: pd.DataFrame):
    feats = feats.copy()
    feats["target_decline"] = (feats["sow_trend"] < -0.10).astype(int)

    cols = [c for c in FEATURE_COLS if c in feats.columns]
    X = pd.get_dummies(feats[cols].fillna(0), drop_first=True)
    y = feats["target_decline"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="auc"
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Model AUC: {auc:.3f}")

    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10))

    feats["sow_risk_score"] = model.predict_proba(X)[:, 1]
    return feats, model
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURE_COLS = ["early_SoW", "total_lifetime_spend", "recency_days",
                 "txn_count", "tenure_months", "credit_utilization_proxy",
                 "Age", "Credit_Card_Limit", "Credit_Card_APR",
                 "pct_Grocery", "pct_Electronics", "pct_Apparel", "pct_Travel"]

MIN_EARLY_SOW = 0.05   # exclude customers who couldn't have "declined" in any meaningful sense
RELATIVE_DROP_THRESHOLD = -0.30   # 30% relative decline in SoW vs their own early baseline

def run(feats: pd.DataFrame):
    feats = feats.copy()

    # relative decline instead of absolute — removes the mechanical early_SoW ceiling effect
    feats["relative_sow_change"] = np.where(
        feats["early_SoW"] > 0,
        (feats["recent_SoW"] - feats["early_SoW"]) / feats["early_SoW"],
        np.nan
    )
    feats["target_decline"] = (feats["relative_sow_change"] < RELATIVE_DROP_THRESHOLD).astype(int)

    # train only on customers where decline was actually possible to observe
    train_pop = feats[feats["early_SoW"] >= MIN_EARLY_SOW].copy()
    print(f"Training population (early_SoW >= {MIN_EARLY_SOW}): {len(train_pop)} of {len(feats)}")
    print(f"Decline rate in training population: {train_pop['target_decline'].mean():.1%}")

    cols = [c for c in FEATURE_COLS if c in feats.columns]
    X_train_pop = pd.get_dummies(train_pop[cols].fillna(0), drop_first=True)
    y_train_pop = train_pop["target_decline"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_pop, y_train_pop, test_size=0.2, stratify=y_train_pop, random_state=42
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

    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importances.head(10))

    # score EVERY customer (not just training population) for the final output
    X_all = pd.get_dummies(feats[cols].fillna(0), drop_first=True)
    X_all = X_all.reindex(columns=X_train.columns, fill_value=0)  # align columns
    feats["sow_risk_score"] = model.predict_proba(X_all)[:, 1]

    return feats, model
import pandas as pd
feats = pd.read_csv("outputs/customer_action_list.csv")
sow = pd.read_parquet("data/processed/sow_monthly.parquet")

print("Total active customers analyzed:", len(feats))
print("Portfolio avg SoW:", feats["avg_SoW"].mean())
print("SoW tier breakdown:\n", feats["sow_tier"].value_counts())
print("\nSegment breakdown:\n", feats["micro_segment_name"].value_counts())
print("\nSilent attrition count:", (feats["micro_segment_name"]=="Silent Attrition (Card Active, Unused)").sum())
print("Silent attrition recoverable spend:", feats.loc[feats["micro_segment_name"]=="Silent Attrition (Card Active, Unused)", "recoverable_spend"].sum())
print("\nTotal recoverable spend, all segments:", feats["recoverable_spend"].sum())
print("Top 500 priority customers' combined recoverable spend:", feats.sort_values("priority_score", ascending=False).head(500)["recoverable_spend"].sum())
print("\nModel AUC (paste from your last clean run, no leakage):", "___")
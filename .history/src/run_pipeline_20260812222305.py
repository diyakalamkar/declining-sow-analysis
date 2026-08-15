from src.data_loader import load_raw
from src.clean import clean_customers, clean_transactions
from src.sow_engine import compute_sow_monthly
from src.features import build_customer_features
from src import segmentation, risk_model, recommender

print("Loading...")
cat, pay, cust, txn = load_raw()

print("Cleaning...")
cust_clean = clean_customers(cust)
txn_clean = clean_transactions(txn, cat, pay)

print("Computing SoW...")
sow = compute_sow_monthly(txn_clean, cust_clean)
sow.to_parquet("data/processed/sow_monthly.parquet")

print("Building features...")
feats = build_customer_features(txn_clean, cust_clean, sow)

print("Segmenting...")
feats = segmentation.run(feats)

print("Scoring risk...")
feats, model = risk_model.run(feats)

print("Generating recommendations...")
final = recommender.run(feats)

final.to_csv("outputs/customer_action_list.csv", index=False)
print(f"Done. {len(final)} customers scored. Top segment counts:")
print(final["micro_segment_name"].value_counts())
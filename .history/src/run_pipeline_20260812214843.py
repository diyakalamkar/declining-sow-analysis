from src.data_loader import load_raw
from src.clean import clean_customers, clean_transactions
from src.sow_engine import compute_sow_monthly
from src.features import build_customer_features
# ... segmentation, risk_model, recommender imports

cat, pay, cust, txn = load_raw()
cust = clean_customers(cust)
txn = clean_transactions(txn, cat, pay)
sow = compute_sow_monthly(txn, cust)
feats = build_customer_features(txn, cust, sow)
# -> segmentation.run(feats) -> risk_model.run(feats) -> recommender.run(feats)
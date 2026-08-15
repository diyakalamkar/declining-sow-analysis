import pandas as pd
from src.data_loader import load_raw
from src.clean import clean_customers, clean_transactions

cat, pay, cust, txn = load_raw()
cust_clean = clean_customers(cust)
txn = clean_transactions(txn, cat, pay)

# A) Is the drop concentrated in one category, or broad-based?
before = txn[(txn.year_month >= "2025-05") & (txn.year_month <= "2025-07") & (txn.Payment_Code == 3)]
after  = txn[(txn.year_month >= "2025-08") & (txn.year_month <= "2025-10") & (txn.Payment_Code == 3)]

print("ABC-card spend by category, 3mo BEFORE cliff:")
print(before.groupby("Category")["Net_Amount"].sum().sort_values(ascending=False))
print("\nABC-card spend by category, 3mo AFTER cliff:")
print(after.groupby("Category")["Net_Amount"].sum().sort_values(ascending=False))

# B) Where did the diverted spend go? Non-ABC payment methods, same period, at XYZ Inc
before_other = txn[(txn.year_month >= "2025-05") & (txn.year_month <= "2025-07") & (txn.Payment_Code != 3)]
after_other  = txn[(txn.year_month >= "2025-08") & (txn.year_month <= "2025-10") & (txn.Payment_Code != 3)]

print("\nNon-ABC spend by payment method, BEFORE:")
print(before_other.groupby("Payment_Method")["Net_Amount"].sum())
print("\nNon-ABC spend by payment method, AFTER:")
print(after_other.groupby("Payment_Method")["Net_Amount"].sum())

# C) Broad-based dip, or did a subset of customers stop using the ABC card entirely?
cust_abc_before = txn[(txn.year_month >= "2025-05") & (txn.year_month <= "2025-07") & (txn.Payment_Code == 3)] \
    .groupby("Customer_ID")["Net_Amount"].sum()
cust_abc_after = txn[(txn.year_month >= "2025-08") & (txn.year_month <= "2025-10") & (txn.Payment_Code == 3)] \
    .groupby("Customer_ID")["Net_Amount"].sum()

went_to_zero = cust_abc_before.index.difference(cust_abc_after.index)
print(f"\nCustomers who used ABC card before but NOT after: {len(went_to_zero)} of {len(cust_abc_before)}")
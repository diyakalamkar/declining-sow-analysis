import pandas as pd
sow = pd.read_parquet("data/processed/sow_monthly.parquet")

monthly = sow.groupby("year_month").agg(
    n_active_customers=("Customer_ID", "nunique"),
    total_spend=("total_spend", "sum"),
    abc_spend=("abc_spend", "sum")
).reset_index()
monthly["SoW"] = monthly["abc_spend"] / monthly["total_spend"]
print(monthly.to_string())
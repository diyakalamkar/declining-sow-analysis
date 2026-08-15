import pandas as pd
import numpy as np
from . import config as cfg

def clean_dates(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c], format=cfg.DATE_FMT, errors="coerce")
    return df

def clean_customers(cust):
    cust = cust.drop_duplicates("Customer_ID").copy()
    cust = clean_dates(cust, ["Credit_Card_Open_Date", "Credit_Card_Close_Date"])
    cust["is_card_active"] = cust["Credit_Card_Close_Date"].isna()
    # sanity: open date must exist
    cust = cust[cust["Credit_Card_Open_Date"].notna()]
    return cust

def clean_transactions(txn, cat_map, pay_map):
    txn = clean_dates(txn.copy(), ["Transaction_Date"])
    txn = txn.dropna(subset=["Transaction_Date", "Customer_ID", "Transaction_Amount"])

    # point 7: Number_of_Transactions = 0 for returns
    txn["Number_of_Transactions"] = np.where(
        txn["Transaction_Type"].eq("Return"), 0, txn["Number_of_Transactions"]
    )
    txn["Number_of_Transactions"] = txn["Number_of_Transactions"].fillna(0).astype(int)

    # signed net amount: returns subtract
    txn["Net_Amount"] = np.where(
        txn["Transaction_Type"].eq("Return"),
        -txn["Transaction_Amount"].abs(),
        txn["Transaction_Amount"].abs()
    )

    txn = txn.merge(cat_map, on="Category_Code", how="left")
    

    # fiscal year/month (Aug–Jul)
    txn["fiscal_month"] = txn["Transaction_Date"].dt.month.apply(lambda m: m - 7 if m >= 8 else m + 5)
    txn["fiscal_year"] = txn["Transaction_Date"].dt.year.where(
        txn["Transaction_Date"].dt.month >= 8, txn["Transaction_Date"].dt.year - 1
    )
    txn["year_month"] = txn["Transaction_Date"].dt.to_period("M")
    return txn
import pandas as pd
import numpy as np
from . import config as cfg

def clean_dates(df, cols):
    for c in cols:
        parsed = pd.to_datetime(df[c], format=cfg.DATE_FMT, errors="coerce")
        # fallback for any stray non-conforming rows (e.g. Excel-reformatted cells)
        still_bad = parsed.isna() & df[c].notna()
        if still_bad.any():
            parsed.loc[still_bad] = pd.to_datetime(df.loc[still_bad, c], errors="coerce", dayfirst=True)
        df[c] = parsed
    return df

def clean_customers(cust):
    cust = cust.drop_duplicates(cfg.COL_CUSTOMER_ID).copy()
    print("Before date clean:", len(cust))

    cust = clean_dates(cust, [cfg.COL_OPEN_DATE, cfg.COL_CLOSE_DATE])
    print("Open date nulls after parse:", cust[cfg.COL_OPEN_DATE].isna().sum())
    print("Close date nulls after parse (expected: most, since most cards are still open):",
          cust[cfg.COL_CLOSE_DATE].isna().sum())

    cust["is_card_active"] = cust[cfg.COL_CLOSE_DATE].isna()

    cust = cust[cust[cfg.COL_OPEN_DATE].notna()]
    print("After filtering null open-dates:", len(cust))

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
    txn = txn.merge(pay_map, on="Payment_Code", how="left")

    print("Null Category after merge (should be ~0):", txn["Category"].isna().sum())
    print("Null Payment_Method after merge (should be ~0):", txn["Payment_Method"].isna().sum())

    # fiscal year/month (Aug–Jul)
    txn["fiscal_month"] = txn["Transaction_Date"].dt.month.apply(lambda m: m - 7 if m >= 8 else m + 5)
    txn["fiscal_year"] = txn["Transaction_Date"].dt.year.where(
        txn["Transaction_Date"].dt.month >= 8, txn["Transaction_Date"].dt.year - 1
    )
    txn["year_month"] = txn["Transaction_Date"].dt.to_period("M")
    return txn
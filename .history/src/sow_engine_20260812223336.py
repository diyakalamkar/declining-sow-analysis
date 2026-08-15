import pandas as pd
import numpy as np
from . import config as cfg

def build_active_month_index(cust, as_of_date):
    """One row per customer per fiscal month the card was active."""
    rows = []
    for _, r in cust.iterrows():
        end = r[cfg.COL_CLOSE_DATE] if pd.notna(r[cfg.COL_CLOSE_DATE]) else as_of_date
        months = pd.period_range(r[cfg.COL_OPEN_DATE], end, freq="M")
        rows.append(pd.DataFrame({"Customer_ID": r["Customer_ID"], "year_month": months}))
    return pd.concat(rows, ignore_index=True)

def compute_sow_monthly(txn, cust):
    as_of = txn["Transaction_Date"].max()
    active_idx = build_active_month_index(cust, as_of)

    # total net spend at XYZ Inc, all payment methods
    total = txn.groupby(["Customer_ID", "year_month"])["Net_Amount"].sum().rename("total_spend")

    # spend specifically on ABC co-branded card
    abc = (txn[txn["Payment_Code"] == cfg.ABC_PAYMENT_CODE]
           .groupby(["Customer_ID", "year_month"])["Net_Amount"].sum()
           .rename("abc_spend"))

    sow = active_idx.merge(total, on=["Customer_ID", "year_month"], how="left") \
                     .merge(abc, on=["Customer_ID", "year_month"], how="left")
    sow[["total_spend", "abc_spend"]] = sow[["total_spend", "abc_spend"]].fillna(0)

    # SoW undefined when total_spend<=0 (no spend that month) -> NaN, exclude from averages
    sow["SoW"] = np.where(sow["total_spend"] > 0, sow["abc_spend"] / sow["total_spend"], np.nan)
    return sow
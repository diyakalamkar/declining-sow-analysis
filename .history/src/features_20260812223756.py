import pandas as pd
import numpy as np

def build_customer_features(txn, cust, sow):
    as_of = txn["Transaction_Date"].max()

    cust_sow = sow.groupby("Customer_ID").agg(
        avg_SoW=("SoW", "mean"),
        sow_std=("SoW", "std"),
        months_active=("year_month", "nunique")
    ).reset_index()

    recent = sow.sort_values("year_month").groupby("Customer_ID").tail(3)
    recent_sow = recent.groupby("Customer_ID")["SoW"].mean().rename("recent_SoW").reset_index()

    early = sow.sort_values("year_month").groupby("Customer_ID").head(3)
    early_sow = early.groupby("Customer_ID")["SoW"].mean().rename("early_SoW").reset_index()

    feats = cust_sow.merge(recent_sow, on="Customer_ID").merge(early_sow, on="Customer_ID")
    feats["sow_trend"] = feats["recent_SoW"] - feats["early_SoW"]

    total_spend = txn.groupby("Customer_ID")["Net_Amount"].sum().rename("total_lifetime_spend")
    freq = txn[txn.Transaction_Type == "Sale"].groupby("Customer_ID")["Transaction_ID"].count().rename("txn_count")
    recency = (as_of - txn.groupby("Customer_ID")["Transaction_Date"].max()).dt.days.rename("recency_days")

    cat_mix = (txn.groupby(["Customer_ID", "Category"])["Net_Amount"].sum().unstack(fill_value=0))
    cat_mix = cat_mix.div(cat_mix.sum(axis=1).replace(0, np.nan), axis=0).add_prefix("pct_")

    pay_mix = (txn.groupby(["Customer_ID", "Payment_Method"])["Net_Amount"].sum().unstack(fill_value=0))
    pay_mix = pay_mix.div(pay_mix.sum(axis=1).replace(0, np.nan), axis=0).add_prefix("pctpay_")

    feats = (feats
             .merge(total_spend, on="Customer_ID")
             .merge(freq, on="Customer_ID")
             .merge(recency, on="Customer_ID")
             .merge(cat_mix, on="Customer_ID")
             .merge(pay_mix, on="Customer_ID")
             .merge(cust[["Customer_ID","Age","Gender","Membership_Type","Credit_Card_Open_Date",
              "Credit_Card_Closed_Date","Credit_Card_Limit","Credit_Card_APR",
              "is_card_active"]], on="Customer_ID"))

    feats["tenure_months"] = ((as_of - feats["Credit_Card_Open_Date"]).dt.days / 30).round(1)
    feats["credit_utilization_proxy"] = feats["total_lifetime_spend"] / feats["Credit_Card_Limit"]

    # only fillna on numeric columns; leave datetime/object columns alone
    num_cols = feats.select_dtypes(include="number").columns
    feats[num_cols] = feats[num_cols].fillna(0)
    return feats
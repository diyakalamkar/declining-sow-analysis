import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
sow = pd.read_parquet("data/processed/sow_monthly.parquet")
feats = pd.read_csv("outputs/customer_action_list.csv")

st.title("XYZ–ABC Co-Brand Card: Share of Wallet Command Center")

c1, c2, c3 = st.columns(3)
c1.metric("Portfolio Avg SoW", f"{feats['avg_SoW'].mean():.1%}")
c2.metric("Customers At Risk", f"{(feats['sow_tier'] == 'Low SoW - At Risk').sum():,}")
c3.metric("Total Recoverable Spend", f"₹{feats['recoverable_spend'].sum():,.0f}")

# --- fix: convert Period -> string, guard divide-by-zero ---
trend = sow.groupby("year_month").apply(
    lambda d: d.abc_spend.sum() / d.total_spend.sum() if d.total_spend.sum() > 0 else np.nan
).reset_index(name="Portfolio_SoW")
trend = trend.dropna(subset=["Portfolio_SoW"])
trend["year_month"] = trend["year_month"].astype(str)

st.plotly_chart(px.line(trend, x="year_month", y="Portfolio_SoW", title="Portfolio SoW Trend"))

# st.plotly_chart(px.scatter(feats, x="recency_days", y="avg_SoW", color="micro_segment_name",
#                             size="total_lifetime_spend", title="Segment Map"))
feats["size_for_plot"] = feats["total_lifetime_spend"].clip(lower=0)
st.plotly_chart(px.scatter(feats, x="recency_days", y="avg_SoW", color="micro_segment_name",
                            size="size_for_plot", title="Segment Map",
                            hover_data=["Customer_ID", "total_lifetime_spend"]))

st.dataframe(feats.sort_values("priority_score", ascending=False).head(50))
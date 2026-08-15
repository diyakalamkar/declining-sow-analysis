import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
sow = pd.read_parquet("data/processed/sow_monthly.parquet")
feats = pd.read_csv("outputs/customer_action_list.csv")

st.title("XYZ–ABC Co-Brand Card: Share of Wallet Command Center")

c1, c2, c3 = st.columns(3)
c1.metric("Portfolio Avg SoW", f"{feats['avg_SoW'].mean():.1%}")
c2.metric("Customers At Risk", f"{(feats['sow_tier']=='Low SoW – At Risk').sum():,}")
c3.metric("Total Recoverable Spend", f"₹{feats['recoverable_spend'].sum():,.0f}")

trend = sow.groupby("year_month").apply(lambda d: d.abc_spend.sum()/d.total_spend.sum()).reset_index(name="Portfolio_SoW")
st.plotly_chart(px.line(trend, x="year_month", y="Portfolio_SoW", title="Portfolio SoW Trend"))

st.plotly_chart(px.scatter(feats, x="recency_days", y="avg_SoW", color="micro_segment_name",
                            size="total_lifetime_spend", title="Segment Map"))

st.dataframe(feats.sort_values("priority_score", ascending=False).head(50))
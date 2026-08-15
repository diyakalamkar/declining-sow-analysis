import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="SoW Command Center", layout="wide", page_icon="💳")

st.markdown("""
<style>
    .main { background-color: #f7f9fc; }
    div[data-testid="stMetric"] {
        background-color: white; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #eef0f3;
    }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1a1f36; }
    h1 { color: #1a1f36; font-weight: 800; }
    h3 { color: #3b3f4a; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    sow = pd.read_parquet("data/processed/sow_monthly.parquet")
    feats = pd.read_csv("outputs/customer_action_list.csv")
    return sow, feats

sow, feats = load_data()

st.title("💳 XYZ–ABC Co-Brand Card: Share of Wallet Command Center")
st.caption("ABC Ltd. Analytics & Strategy · Customer & Credit Card Analytics Hackathon 2026")

# ---- KPI row ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portfolio Avg SoW", f"{feats['avg_SoW'].mean():.1%}")
k2.metric("Customers At Risk", f"{(feats['sow_tier'] == 'Low SoW - At Risk').sum():,}")
k3.metric("Silent Attrition Cohort", f"{(feats['micro_segment_name']=='Silent Attrition (Card Active, Unused)').sum():,}")
k4.metric("Total Recoverable Spend", f"₹{feats['recoverable_spend'].sum()/1e7:,.1f} Cr")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📉 Portfolio Trend", "🧩 Segments", "🎯 Priority Targets", "🔍 Customer Lookup"])

with tab1:
    trend = sow.groupby("year_month").apply(
        lambda d: d.abc_spend.sum() / d.total_spend.sum() if d.total_spend.sum() > 0 else np.nan
    ).reset_index(name="Portfolio_SoW")
    trend = trend.dropna(subset=["Portfolio_SoW"])
    trend["year_month"] = trend["year_month"].dt.to_timestamp()   # Period -> real datetime, not string

    fig = px.line(trend, x="year_month", y="Portfolio_SoW", markers=True,
                  title="Portfolio SoW Trend (Aug 2024 – Jul 2026)")
    fig.update_traces(line_color="#4C6FFF", line_width=3)
    fig.add_vline(x="2025-08", line_dash="dash", line_color="#E5484D",
              annotation_text="SoW Cliff — Aug 2025", annotation_position="top")
    fig.update_layout(yaxis_tickformat=".0%", plot_bgcolor="white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Finding:** SoW held steady ~42% through Jul 2025, then dropped to ~24% in a single month "
        "and never recovered. Diagnosis shows spend fell proportionally across *every* category "
        "and diffused evenly across *every* competing payment method — no single competitor or "
        "category drove it. Instead, ~13% of prior ABC-card users stopped using the card entirely "
        "while keeping it open ('silent attrition'), invisible to standard active-card metrics."
    )

with tab2:
    seg_counts = feats["micro_segment_name"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig_pie = px.pie(seg_counts, names="Segment", values="Customers", hole=0.45)
        fig_pie.update_layout(height=420, showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        plot_df = feats[feats["micro_segment_name"] != "Not Targeted (High/Dormant SoW)"].copy()
        plot_df["size_for_plot"] = plot_df["total_lifetime_spend"].clip(lower=0)
        fig_scatter = px.scatter(plot_df, x="recency_days", y="avg_SoW", color="micro_segment_name",
                                  size="size_for_plot", opacity=0.65,
                                  title="Actionable Segments — Recency vs SoW",
                                  labels={"recency_days": "Days Since Last Purchase", "avg_SoW": "Avg SoW"})
        fig_scatter.update_layout(yaxis_tickformat=".0%", plot_bgcolor="white", height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Segment Profiles")
    profile_cols = ["avg_SoW", "sow_trend", "total_lifetime_spend", "recency_days", "tenure_months"]
    profile_cols = [c for c in profile_cols if c in feats.columns]
    st.dataframe(
        feats.groupby("micro_segment_name")[profile_cols].mean().round(2),
        use_container_width=True
    )

with tab3:
    st.subheader("Top Priority Customers (Risk × Recoverable Spend)")
    top_n = st.slider("Show top N customers", 20, 500, 100, step=20)
    display_cols = ["Customer_ID", "micro_segment_name", "sow_risk_score", "avg_SoW",
                     "recoverable_spend", "priority_score", "recommended_offer"]
    display_cols = [c for c in display_cols if c in feats.columns]
    top_df = feats.sort_values("priority_score", ascending=False).head(top_n)[display_cols]
    st.dataframe(
        top_df.style.format({
            "sow_risk_score": "{:.2f}", "avg_SoW": "{:.1%}",
            "recoverable_spend": "₹{:,.0f}", "priority_score": "{:,.0f}"
        }),
        use_container_width=True, height=500
    )
    st.download_button("Download full action list (CSV)",
                        feats.sort_values("priority_score", ascending=False).to_csv(index=False),
                        "customer_action_list.csv", "text/csv")

with tab4:
    st.subheader("Look up a customer")
    cid = st.selectbox("Customer ID", feats["Customer_ID"].astype(str).tolist())
    row = feats[feats["Customer_ID"].astype(str) == cid].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Segment", row.get("micro_segment_name", "—"))
    c2.metric("Avg SoW", f"{row.get('avg_SoW', 0):.1%}")
    c3.metric("Risk Score", f"{row.get('sow_risk_score', 0):.2f}")
    st.write(f"**Recommended offer:** {row.get('recommended_offer', '—')}")
    st.write(f"**Recoverable spend:** ₹{row.get('recoverable_spend', 0):,.0f}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# st.set_page_config(layout="wide")
# sow = pd.read_parquet("data/processed/sow_monthly.parquet")
# feats = pd.read_csv("outputs/customer_action_list.csv")

# st.title("XYZ–ABC Co-Brand Card: Share of Wallet Command Center")

# c1, c2, c3 = st.columns(3)
# c1.metric("Portfolio Avg SoW", f"{feats['avg_SoW'].mean():.1%}")
# c2.metric("Customers At Risk", f"{(feats['sow_tier'] == 'Low SoW - At Risk').sum():,}")
# c3.metric("Total Recoverable Spend", f"₹{feats['recoverable_spend'].sum():,.0f}")

# # --- fix: convert Period -> string, guard divide-by-zero ---
# trend = sow.groupby("year_month").apply(
#     lambda d: d.abc_spend.sum() / d.total_spend.sum() if d.total_spend.sum() > 0 else np.nan
# ).reset_index(name="Portfolio_SoW")
# trend = trend.dropna(subset=["Portfolio_SoW"])
# trend["year_month"] = trend["year_month"].astype(str)

# st.plotly_chart(px.line(trend, x="year_month", y="Portfolio_SoW", title="Portfolio SoW Trend"))


# feats["size_for_plot"] = feats["total_lifetime_spend"].clip(lower=0)

# # st.plotly_chart(px.scatter(feats, x="recency_days", y="avg_SoW", color="micro_segment_name",
# #                             size="size_for_plot", title="Segment Map",
# #                             hover_data=["Customer_ID", "total_lifetime_spend"]))

# st.plotly_chart(px.scatter(
#     feats[feats["micro_segment_name"] != "Not Targeted (High/Dormant SoW)"],
#     x="recency_days", y="avg_SoW", color="micro_segment_name",
#     size="size_for_plot", title="Segment Map (Actionable Segments Only)",
#     opacity=0.6
# ), use_container_width=True)

# st.dataframe(feats.sort_values("priority_score", ascending=False).head(50))
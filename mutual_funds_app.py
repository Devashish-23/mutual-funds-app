import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Mutual Fund Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .block-container { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

RISK_MAP = {1: "Low", 2: "Low-Moderate", 3: "Moderate", 4: "Mod-High", 5: "High", 6: "Very High"}

@st.cache_data
def load_data():
    df = pd.read_csv("comprehensive_mutual_funds_data.csv", dtype=str)

    numeric_cols = [
        "min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr", "fund_age_yr",
        "sortino", "alpha", "sd", "beta", "sharpe", "risk_level", "rating",
        "returns_1yr", "returns_3yr", "returns_5yr"
    ]
    for col in numeric_cols:
        if col in df.columns:
            cleaned = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
            # If multiple dots exist, keep only the first one
            def fix_dots(s):
                parts = s.split(".")
                if len(parts) > 2:
                    return parts[0] + "." + "".join(parts[1:])
                return s
            cleaned = cleaned.apply(fix_dots)
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    df["risk_level"] = df["risk_level"].fillna(0).astype(int)
    df["risk_label"] = df["risk_level"].map(RISK_MAP)
    return df

df = load_data()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("🔍 Filters")

    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Category", categories)
    filtered = df if sel_cat == "All" else df[df["category"] == sel_cat]

    sub_cats = ["All"] + sorted(filtered["sub_category"].dropna().unique().tolist())
    sel_sub = st.selectbox("Sub-Category", sub_cats)
    if sel_sub != "All":
        filtered = filtered[filtered["sub_category"] == sel_sub]

    amcs = ["All"] + sorted(filtered["amc_name"].dropna().unique().tolist())
    sel_amc = st.selectbox("AMC", amcs)
    if sel_amc != "All":
        filtered = filtered[filtered["amc_name"] == sel_amc]

    risk_opts = ["All"] + [RISK_MAP[i] for i in sorted(RISK_MAP)]
    sel_risk = st.selectbox("Risk Level", risk_opts)
    if sel_risk != "All":
        filtered = filtered[filtered["risk_label"] == sel_risk]

    rating_min, rating_max = int(df["rating"].min()), int(df["rating"].max())
    sel_rating = st.slider("Min Star Rating", rating_min, rating_max, rating_min)
    filtered = filtered[filtered["rating"] >= sel_rating]

    exp_min = float(df["expense_ratio"].min())
    exp_max = float(df["expense_ratio"].max())
    sel_exp = st.slider("Max Expense Ratio (%)", exp_min, exp_max, exp_max, step=0.05)
    filtered = filtered[filtered["expense_ratio"] <= sel_exp]

    st.markdown(f"**{len(filtered)} funds** match your filters")

st.title("📈 Mutual Fund Dashboard")
st.caption(f"Dataset: {len(df)} funds | Showing: {len(filtered)} after filters")

def fmt(val, decimals=1, suffix="%"):
    return f"{val:.{decimals}f}{suffix}" if (len(filtered) > 0 and pd.notna(val)) else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg 1Y Return", fmt(filtered['returns_1yr'].mean()))
k2.metric("Avg 3Y Return", fmt(filtered['returns_3yr'].mean()))
k3.metric("Avg 5Y Return", fmt(filtered['returns_5yr'].mean()))
k4.metric("Avg Expense Ratio", fmt(filtered['expense_ratio'].mean(), decimals=2))
k5.metric("Avg Sharpe", fmt(filtered['sharpe'].mean(), suffix=""))

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📉 Returns", "⚖️ Risk & Metrics", "🏆 Top Funds", "🔎 Fund Explorer"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        cat_counts = filtered["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, names="Category", values="Count",
                     title="Fund Distribution by Category",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_counts = filtered["risk_label"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        risk_order = list(RISK_MAP.values())
        risk_counts["Risk"] = pd.Categorical(risk_counts["Risk"], categories=risk_order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk")
        fig2 = px.bar(risk_counts, x="Risk", y="Count", title="Risk Level Distribution",
                      color="Risk", text="Count")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        amc_top = filtered.groupby("amc_name").size().nlargest(12).reset_index()
        amc_top.columns = ["AMC", "Funds"]
        fig3 = px.bar(amc_top, x="Funds", y="AMC", orientation="h",
                      title="Top 12 AMCs by Fund Count", color="Funds",
                      color_continuous_scale="Blues")
        fig3.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        rating_dist = filtered["rating"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Rating", "Count"]
        fig4 = px.bar(rating_dist, x="Rating", y="Count", title="Star Rating Distribution",
                      color="Rating", color_continuous_scale="Oranges", text="Count")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        avg_ret = (
            filtered.groupby("category")[["returns_1yr", "returns_3yr", "returns_5yr"]]
            .mean().reset_index()
            .melt(id_vars="category", var_name="Period", value_name="Return (%)")
        )
        avg_ret["Period"] = avg_ret["Period"].str.replace("returns_", "").str.replace("yr", "Y")
        fig5 = px.bar(avg_ret, x="category", y="Return (%)", color="Period",
                      barmode="group", title="Avg Returns by Category & Period",
                      color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"])
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        fig6 = px.box(filtered[filtered["returns_1yr"].notna()], x="category", y="returns_1yr",
                      title="1Y Return Distribution by Category",
                      color="category", points=False)
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig7 = px.scatter(
            filtered.dropna(subset=["returns_3yr", "returns_5yr"]),
            x="returns_3yr", y="returns_5yr", color="category",
            hover_name="scheme_name", title="3Y vs 5Y Returns",
            labels={"returns_3yr": "3Y Return (%)", "returns_5yr": "5Y Return (%)"},
            opacity=0.7
        )
        st.plotly_chart(fig7, use_container_width=True)

    with c4:
        fig8 = px.histogram(
            filtered.dropna(subset=["returns_1yr"]),
            x="returns_1yr", nbins=40, color="category",
            title="1Y Return Distribution (Histogram)",
            labels={"returns_1yr": "1Y Return (%)"},
            barmode="overlay", opacity=0.65
        )
        st.plotly_chart(fig8, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig9 = px.scatter(
            filtered.dropna(subset=["sharpe", "returns_3yr"]),
            x="sharpe", y="returns_3yr", color="category",
            size="fund_size_cr", size_max=22,
            hover_name="scheme_name",
            title="Sharpe Ratio vs 3Y Return (bubble = fund size)",
            labels={"sharpe": "Sharpe Ratio", "returns_3yr": "3Y Return (%)"},
            opacity=0.75
        )
        st.plotly_chart(fig9, use_container_width=True)

    with c2:
        fig10 = px.scatter(
            filtered.dropna(subset=["alpha", "beta"]),
            x="beta", y="alpha", color="category",
            hover_name="scheme_name",
            title="Alpha vs Beta by Category",
            opacity=0.7
        )
        fig10.add_hline(y=0, line_dash="dash", line_color="gray")
        fig10.add_vline(x=1, line_dash="dash", line_color="gray")
        st.plotly_chart(fig10, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        avg_metrics = filtered.groupby("category")[["sharpe", "sortino", "alpha", "sd"]].mean().reset_index()
        fig11 = px.bar(
            avg_metrics.melt(id_vars="category", var_name="Metric", value_name="Value"),
            x="category", y="Value", color="Metric", barmode="group",
            title="Avg Risk Metrics by Category"
        )
        st.plotly_chart(fig11, use_container_width=True)

    with c4:
        num_cols = ["expense_ratio", "sharpe", "alpha", "beta", "sd", "sortino",
                    "returns_1yr", "returns_3yr", "returns_5yr"]
        corr = filtered[num_cols].corr()
        fig12 = px.imshow(corr, text_auto=".2f", aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig12, use_container_width=True)

with tab4:
    sort_col = st.selectbox("Rank by", ["returns_1yr", "returns_3yr", "returns_5yr", "sharpe", "alpha"],
                            format_func=lambda x: x.replace("_", " ").title())
    top_n = st.slider("Show top N funds", 5, 30, 15)

    top_funds = (
        filtered.dropna(subset=[sort_col])
        .nlargest(top_n, sort_col)[
            ["scheme_name", "amc_name", "category", "rating",
             "returns_1yr", "returns_3yr", "returns_5yr",
             "expense_ratio", "sharpe", "risk_label"]
        ]
    )

    fig13 = px.bar(
        top_funds, x=sort_col, y="scheme_name", orientation="h",
        color="category", title=f"Top {top_n} Funds by {sort_col.replace('_', ' ').title()}",
        height=max(400, top_n * 30)
    )
    fig13.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig13, use_container_width=True)

    st.dataframe(
        top_funds.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "returns_1yr": st.column_config.NumberColumn("1Y Return (%)", format="%.1f"),
            "returns_3yr": st.column_config.NumberColumn("3Y Return (%)", format="%.1f"),
            "returns_5yr": st.column_config.NumberColumn("5Y Return (%)", format="%.1f"),
            "expense_ratio": st.column_config.NumberColumn("Expense Ratio (%)", format="%.2f"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "rating": st.column_config.NumberColumn("⭐ Rating"),
        }
    )

with tab5:
    search = st.text_input("🔍 Search fund name", placeholder="e.g. HDFC, Mirae, SBI...")
    results = filtered[filtered["scheme_name"].str.contains(search, case=False, na=False)] if search else filtered

    st.caption(f"Showing {len(results)} funds")
    st.dataframe(
        results[[
            "scheme_name", "amc_name", "category", "sub_category",
            "returns_1yr", "returns_3yr", "returns_5yr",
            "expense_ratio", "sharpe", "alpha", "beta", "sd",
            "fund_size_cr", "fund_age_yr", "risk_label", "rating"
        ]].reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            "returns_1yr": st.column_config.NumberColumn("1Y Ret %", format="%.1f"),
            "returns_3yr": st.column_config.NumberColumn("3Y Ret %", format="%.1f"),
            "returns_5yr": st.column_config.NumberColumn("5Y Ret %", format="%.1f"),
            "expense_ratio": st.column_config.NumberColumn("Exp Ratio", format="%.2f"),
            "fund_size_cr": st.column_config.NumberColumn("Fund Size (Cr)"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "alpha": st.column_config.NumberColumn("Alpha", format="%.2f"),
            "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
            "rating": st.column_config.NumberColumn("⭐"),
        }
    )

    if len(results) > 0:
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=results.to_csv(index=False).encode(),
            file_name="filtered_mutual_funds.csv",
            mime="text/csv"
        )

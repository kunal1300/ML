import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────── PAGE CONFIG ────────────────────────────────
st.set_page_config(
    page_title="Indian Roads EDA",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── CUSTOM CSS ─────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h2 { color: #e63946; font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #9ca3af; font-size: 0.85rem; margin: 4px 0 0 0; }
    .section-header {
        background: linear-gradient(90deg, #e63946 0%, #c1121f 100%);
        color: white; padding: 10px 20px; border-radius: 8px;
        font-size: 1.05rem; font-weight: 600; margin-bottom: 14px;
    }
    .insight-box {
        background: #1a1d2e; border-left: 4px solid #e63946;
        border-radius: 0 8px 8px 0; padding: 14px 18px;
        margin: 12px 0; font-size: 0.9rem; color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────── THEME ──────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
    font=dict(color="#d1d5db", family="Inter, sans-serif"),
    title_font=dict(size=14, color="#ffffff"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="#252840", bordercolor="#2e3250", borderwidth=1),
    xaxis=dict(gridcolor="#2e3250", linecolor="#2e3250", zerolinecolor="#2e3250"),
    yaxis=dict(gridcolor="#2e3250", linecolor="#2e3250", zerolinecolor="#2e3250"),
)
RED=  "#e63946"; BLUE="#4361ee"; TEAL="#2ec4b6"
ORG=  "#ff9f1c"; GRN= "#2dc653"; PURP="#7209b7"

def T(fig, **kw):
    fig.update_layout(**BASE_LAYOUT, **kw)
    return fig

def section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"<div class='insight-box'>💡 {text}</div>", unsafe_allow_html=True)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.markdown(f"<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>",
                     unsafe_allow_html=True)

# ─── pandas-3 safe helpers ───────────────────────────────────────────────────
def vc(series, name_col, count_col="Count"):
    d = series.value_counts().reset_index()
    d.columns = [name_col, count_col]
    return d

def gbs(df, by, count_col="Count"):
    return df.groupby(by, observed=True).size().reset_index(name=count_col)

# ─────────────────────────────── DATA LOAD ──────────────────────────────────
@st.cache_data
def load_data(f):
    df = pd.read_csv(f)
    df["festival"]   = df["festival"].fillna("No Festival")
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    return df

# ─────────────────────────────── SIDEBAR ────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛣️ Indian Roads EDA")
    st.markdown("---")
    uploaded = st.file_uploader("Upload `indian_roads_dataset.csv`", type=["csv"])
    if uploaded is None:
        st.info("Please upload the dataset CSV to begin.")
        st.stop()

    df = load_data(uploaded)
    st.success(f"✅ {len(df):,} records loaded")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview", "⏱️ Time Patterns", "🌦️ Weather & Environment",
        "🛤️ Road Infrastructure", "💥 Causes & Severity",
        "📍 Geographic Hotspots", "📊 Correlations & Risk",
    ])
    st.markdown("---")
    st.markdown("**Filters**")
    sel_city = st.selectbox("City",     ["All"] + sorted(df["city"].dropna().unique().tolist()))
    sel_sev  = st.selectbox("Severity", ["All"] + sorted(df["accident_severity"].dropna().unique().tolist()))
    dff = df.copy()
    if sel_city != "All": dff = dff[dff["city"] == sel_city]
    if sel_sev  != "All": dff = dff[dff["accident_severity"] == sel_sev]
    st.markdown(f"*Showing **{len(dff):,}** records*")

# ═══════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛣️ Indian Roads — Accident EDA Dashboard")
    st.markdown("Explore hidden patterns in road accidents across India. Use the sidebar to navigate and filter.")
    st.markdown("---")

    metric_row([
        ("Total Accidents",  f"{len(dff):,}"),
        ("Fatal Accidents",  f"{int((dff['accident_severity']=='fatal').sum()):,}"),
        ("Total Casualties", f"{int(dff['casualties'].sum()):,}"),
        ("Cities Covered",   str(dff["city"].nunique())),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Accidents by City")
        d = vc(dff["city"], "City", "Accidents")
        fig = px.bar(d, x="Accidents", y="City", orientation="h",
                     color="Accidents", color_continuous_scale=[BLUE, RED])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("Severity Breakdown")
        d = vc(dff["accident_severity"], "Severity", "Count")
        fig = px.pie(d, names="Severity", values="Count", hole=0.45,
                     color="Severity",
                     color_discrete_map={"minor": GRN, "major": ORG, "fatal": RED})
        T(fig); st.plotly_chart(fig, use_container_width=True)

    section("Monthly Accident Trend")
    mo = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
    d = dff.groupby("month_name", observed=True).size().reindex(mo).fillna(0).reset_index()
    d.columns = ["Month", "Accidents"]
    fig = px.area(d, x="Month", y="Accidents",
                  color_discrete_sequence=[RED], line_shape="spline")
    fig.update_traces(fill="tozeroy", fillcolor="rgba(230,57,70,0.15)", line_width=2.5)
    T(fig); st.plotly_chart(fig, use_container_width=True)
    insight("Navigate via the sidebar to explore time, weather, infrastructure, causes, geography, and risk patterns.")

# ═══════════════════════════════════════════════════════════════════════════
# ⏱️ TIME PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Time Patterns":
    st.title("⏱️ Time-Based Accident Patterns")

    hourly   = gbs(dff, "hour", "Accidents")
    peak_h   = int(hourly.loc[hourly["Accidents"].idxmax(), "hour"])
    safe_h   = int(hourly.loc[hourly["Accidents"].idxmin(), "hour"])
    day_ord  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily    = dff.groupby("day_of_week", observed=True).size().reindex(day_ord).fillna(0).reset_index()
    daily.columns = ["Day", "Accidents"]
    worst_d  = daily.loc[daily["Accidents"].idxmax(), "Day"]
    peak_pct = int(dff["is_peak_hour"].mean() * 100)

    metric_row([("Peak Hour", f"{peak_h}:00"), ("Safest Hour", f"{safe_h}:00"),
                ("Worst Day", worst_d),         ("Peak-Hour %", f"{peak_pct}%")])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Accidents by Hour")
        fig = go.Figure(go.Scatter(x=hourly["hour"], y=hourly["Accidents"],
            mode="lines+markers", line=dict(color=RED, width=3),
            marker=dict(size=7, color=RED, line=dict(color="white", width=1))))
        T(fig); fig.update_xaxes(title="Hour", tickvals=list(range(0,24,2)))
        fig.update_yaxes(title="Accidents"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("Peak vs Non-Peak")
        pk  = dff["is_peak_hour"].map({0:"Non-Peak", 1:"Peak"})
        d   = vc(pk, "Type", "Count")
        fig = px.pie(d, names="Type", values="Count", hole=0.45,
                     color="Type", color_discrete_map={"Peak":RED,"Non-Peak":BLUE})
        T(fig); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("Weekday vs Weekend")
        wk  = dff["is_weekend"].map({0:"Weekday", 1:"Weekend"})
        d   = vc(wk, "Type", "Count")
        fig = px.bar(d, x="Type", y="Count", color="Type",
                     color_discrete_map={"Weekday":TEAL,"Weekend":ORG})
        T(fig, showlegend=False); fig.update_xaxes(title="")
        fig.update_yaxes(title="Accidents"); st.plotly_chart(fig, use_container_width=True)
    with c4:
        section("Accidents by Day of Week")
        fig = px.bar(daily, x="Day", y="Accidents",
                     color="Accidents", color_continuous_scale=[BLUE, RED])
        T(fig, coloraxis_showscale=False); fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    insight(f"Hour {peak_h}:00 is the most dangerous, {safe_h}:00 is the safest. {worst_d} records the most weekly accidents.")

# ═══════════════════════════════════════════════════════════════════════════
# 🌦️ WEATHER & ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌦️ Weather & Environment":
    st.title("🌦️ Weather & Environmental Factors")

    w_df  = vc(dff["weather"],         "Weather",    "Count")
    v_df  = vc(dff["visibility"],      "Visibility", "Count")
    den_df= vc(dff["traffic_density"], "Density",    "Count")
    dff2  = dff.copy()
    dff2["temp_bin"] = pd.cut(dff2["temperature"], bins=5)
    t_df  = dff2.groupby("temp_bin", observed=True).size().reset_index(name="Accidents")
    t_df["Range"] = t_df["temp_bin"].astype(str)

    metric_row([("Most Accidents in", w_df.iloc[0]["Weather"]),
                ("Top Visibility",    v_df.iloc[0]["Visibility"]),
                ("Traffic Densities", str(dff["traffic_density"].nunique())),
                ("Avg Temperature",   f"{dff['temperature'].mean():.1f}°C")])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("By Weather Condition")
        fig = px.bar(w_df, x="Count", y="Weather", orientation="h",
                     color="Count", color_continuous_scale=[BLUE, RED])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("By Visibility")
        fig = px.bar(v_df, x="Count", y="Visibility", orientation="h",
                     color="Count", color_continuous_scale=[TEAL, PURP])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("By Temperature Range")
        fig = px.bar(t_df, x="Range", y="Accidents",
                     color="Accidents", color_continuous_scale=[GRN, ORG, RED])
        T(fig, coloraxis_showscale=False); fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        section("By Traffic Density")
        fig = px.bar(den_df, x="Count", y="Density", orientation="h",
                     color="Count", color_continuous_scale=[GRN, RED])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)

    insight(f"{w_df.iloc[0]['Weather']} weather sees the most accidents. Poor visibility and high traffic density together amplify risk significantly.")

# ═══════════════════════════════════════════════════════════════════════════
# 🛤️ ROAD INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🛤️ Road Infrastructure":
    st.title("🛤️ Road Type & Infrastructure Analysis")

    road_df  = vc(dff["road_type"], "Road Type", "Count")
    lanes_df = gbs(dff, "lanes", "Accidents")
    sig_df   = vc(dff["traffic_signal"].map({0:"No Signal", 1:"Signal Present"}), "Signal", "Count")
    rs_df    = gbs(dff, ["road_type","accident_severity"], "Count")

    no_sig_n = int(sig_df[sig_df["Signal"] == "No Signal"]["Count"].sum())
    top_lane = int(lanes_df.loc[lanes_df["Accidents"].idxmax(), "lanes"])

    metric_row([("Most Dangerous Road",  road_df.iloc[0]["Road Type"]),
                ("Highest Accident Lane", f"{top_lane}-lane"),
                ("Without Signal",        f"{no_sig_n:,}"),
                ("Road Types",            str(dff["road_type"].nunique()))])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Accidents by Road Type")
        fig = px.bar(road_df, x="Road Type", y="Count", color="Road Type",
                     color_discrete_sequence=[RED, ORG, GRN, BLUE])
        T(fig, showlegend=False); fig.update_xaxes(title=""); fig.update_yaxes(title="Accidents")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("Accidents by Number of Lanes")
        fig = go.Figure(go.Scatter(x=lanes_df["lanes"], y=lanes_df["Accidents"],
            mode="lines+markers", line=dict(color=BLUE, width=3),
            marker=dict(size=9, color=BLUE, symbol="square")))
        T(fig); fig.update_xaxes(title="Lanes"); fig.update_yaxes(title="Accidents")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("Traffic Signal Presence")
        fig = px.pie(sig_df, names="Signal", values="Count", hole=0.4,
                     color="Signal",
                     color_discrete_map={"No Signal":RED,"Signal Present":GRN})
        T(fig); st.plotly_chart(fig, use_container_width=True)
    with c4:
        section("Severity by Road Type")
        fig = px.bar(rs_df, x="road_type", y="Count", color="accident_severity",
                     barmode="group",
                     color_discrete_map={"minor":GRN,"major":ORG,"fatal":RED},
                     category_orders={"accident_severity":["minor","major","fatal"]})
        T(fig); fig.update_xaxes(title="Road Type"); fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)

    insight(f"{road_df.iloc[0]['Road Type']} roads report the most accidents. {no_sig_n:,} accidents occurred at locations without traffic signals.")

# ═══════════════════════════════════════════════════════════════════════════
# 💥 CAUSES & SEVERITY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💥 Causes & Severity":
    st.title("💥 Accident Causes & Severity Analysis")

    cause_df = vc(dff["cause"], "Cause", "Count")
    sev_ord  = ["minor","major","fatal"]
    sev_df   = dff.groupby("accident_severity", observed=True).size().reindex(sev_ord).fillna(0).reset_index()
    sev_df.columns = ["Severity", "Count"]
    sev_cas  = dff.groupby("accident_severity", observed=True)["casualties"].mean().reindex(sev_ord).fillna(0).reset_index()
    sev_cas.columns = ["Severity", "Avg Casualties"]
    cs_df    = gbs(dff, ["cause","accident_severity"], "Count")

    top_cause = cause_df.iloc[0]["Cause"]
    fatal_pct = round((dff["accident_severity"] == "fatal").mean() * 100, 1)
    avg_cas   = round(dff["casualties"].mean(), 2)

    metric_row([("Top Cause",              top_cause),
                ("Fatal Accident %",        f"{fatal_pct}%"),
                ("Avg Casualties/Accident", str(avg_cas)),
                ("Unique Causes",           str(dff["cause"].nunique()))])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Top Accident Causes")
        fig = px.bar(cause_df.head(10), x="Count", y="Cause", orientation="h",
                     color="Count", color_continuous_scale=[ORG, RED])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        section("Severity Distribution")
        fig = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                     color_discrete_map={"minor":GRN,"major":ORG,"fatal":RED})
        T(fig, showlegend=False)
        for _, r in sev_df.iterrows():
            fig.add_annotation(x=r["Severity"], y=r["Count"], text=str(int(r["Count"])),
                               showarrow=False, yshift=8, font=dict(color="white", size=12))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("Cause vs Severity Heatmap")
        pivot = cs_df.pivot(index="cause", columns="accident_severity", values="Count").fillna(0)
        fig   = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto", text_auto=True)
        T(fig); st.plotly_chart(fig, use_container_width=True)
    with c4:
        section("Average Casualties by Severity")
        fig = px.bar(sev_cas, x="Severity", y="Avg Casualties", color="Severity",
                     color_discrete_map={"minor":GRN,"major":ORG,"fatal":RED})
        T(fig, showlegend=False)
        for _, r in sev_cas.iterrows():
            fig.add_annotation(x=r["Severity"], y=r["Avg Casualties"],
                               text=f"{r['Avg Casualties']:.2f}", showarrow=False,
                               yshift=8, font=dict(color="white", size=12))
        st.plotly_chart(fig, use_container_width=True)

    insight(f"'{top_cause}' is the leading cause. Fatal accidents = {fatal_pct}% of total, with {avg_cas} average casualties each.")

# ═══════════════════════════════════════════════════════════════════════════
# 📍 GEOGRAPHIC HOTSPOTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📍 Geographic Hotspots":
    st.title("📍 Geographic Hotspots")

    city_df  = vc(dff["city"],  "City",  "Accidents")
    city_cas = dff.groupby("city", observed=True)["casualties"].sum().sort_values().reset_index()
    city_cas.columns = ["City", "Casualties"]
    city_sev = gbs(dff, ["city","accident_severity"], "Count")

    metric_row([
        ("Most Accident-Prone City", city_df.iloc[0]["City"]),
        ("Highest Casualties City",  city_cas.iloc[-1]["City"]),
        ("Cities in Dataset",        str(dff["city"].nunique())),
        ("States in Dataset",        str(dff["state"].nunique()) if "state" in dff.columns else "N/A"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Accidents by City")
        fig = px.bar(city_df, x="Accidents", y="City", orientation="h",
                     color="Accidents", color_continuous_scale=[BLUE, RED])
        T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "state" in dff.columns:
            section("Accidents by State")
            state_df = vc(dff["state"], "State", "Accidents")
            fig = px.bar(state_df, x="Accidents", y="State", orientation="h",
                         color="Accidents", color_continuous_scale=[TEAL, PURP])
            T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)

    section("Total Casualties by City")
    fig = px.bar(city_cas, x="Casualties", y="City", orientation="h",
                 color="Casualties", color_continuous_scale=[ORG, RED])
    T(fig, coloraxis_showscale=False); st.plotly_chart(fig, use_container_width=True)

    section("Severity by City (Stacked)")
    fig = px.bar(city_sev, x="city", y="Count", color="accident_severity",
                 barmode="stack",
                 color_discrete_map={"minor":GRN,"major":ORG,"fatal":RED},
                 category_orders={"accident_severity":["minor","major","fatal"]})
    T(fig); fig.update_xaxes(title="City", tickangle=-30); fig.update_yaxes(title="Accidents")
    st.plotly_chart(fig, use_container_width=True)

    insight(f"{city_df.iloc[0]['City']} has the most accidents; {city_cas.iloc[-1]['City']} has the highest total casualties — priority areas for intervention.")

# ═══════════════════════════════════════════════════════════════════════════
# 📊 CORRELATIONS & RISK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Correlations & Risk":
    st.title("📊 Risk Factors & Correlations")

    num_cols = [c for c in ["hour","lanes","traffic_signal","temperature",
                             "vehicles_involved","casualties","is_peak_hour",
                             "is_weekend","risk_score"] if c in dff.columns]
    num_df   = dff[num_cols].dropna()
    corr     = num_df.corr()
    has_risk = "risk_score" in corr.columns

    avg_risk = round(dff["risk_score"].mean(), 2) if has_risk else "N/A"
    top_pos  = corr["risk_score"].drop("risk_score").idxmax() if has_risk else "N/A"

    metric_row([("Avg Risk Score",     str(avg_risk)),
                ("Top Positive Factor", str(top_pos)),
                ("Features Analysed",   str(len(num_cols))),
                ("Data Points",         f"{len(num_df):,}")])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        section("Correlation Matrix")
        fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
        T(fig, height=500); st.plotly_chart(fig, use_container_width=True)
    with c2:
        if has_risk:
            section("Factors vs Risk Score")
            rc    = corr["risk_score"].drop("risk_score").sort_values()
            colors= [GRN if v > 0 else RED for v in rc.values]
            fig   = go.Figure(go.Bar(x=rc.values, y=rc.index, orientation="h",
                                     marker_color=colors))
            T(fig, height=500)
            fig.add_vline(x=0, line_width=1, line_color="white")
            fig.update_xaxes(title="Correlation"); st.plotly_chart(fig, use_container_width=True)

    if has_risk and "casualties" in dff.columns:
        section("Risk Score vs Casualties")
        samp = dff.sample(min(2000, len(dff)), random_state=42)
        fig  = px.scatter(samp, x="risk_score", y="casualties", color="accident_severity",
                          color_discrete_map={"minor":GRN,"major":ORG,"fatal":RED}, opacity=0.5)
        T(fig); fig.update_xaxes(title="Risk Score"); fig.update_yaxes(title="Casualties")
        st.plotly_chart(fig, use_container_width=True)

    insight(f"'{top_pos}' shows the strongest positive correlation with risk score.")
    st.markdown("---")
    if st.checkbox("📄 Show Raw Data (first 100 rows)"):
        st.dataframe(dff.head(100), use_container_width=True)

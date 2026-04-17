from pathlib import Path
from html import escape

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DATA_PATH = BASE_DIR / "data" / "indian_roads_dataset.csv"
DOWNLOADS_DATA_PATH = Path.home() / "Downloads" / "indian_roads_dataset.csv"

SEVERITY_ORDER = ["minor", "major", "fatal"]
SEVERITY_COLORS = ["#2a9d8f", "#f4a261", "#e63946"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CHART_TEXT = "#17211b"
CHART_MUTED = "#394840"
CHART_GRID = "#d9e2dc"
CHART_PANEL = "#ffffff"
HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957"
    "?auto=format&fit=crop&w=1800&q=80"
)


st.set_page_config(page_title="Indian Roads Dashboard", layout="wide")


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --page-bg: #f7f9f7;
                --panel: #ffffff;
                --ink: #17211b;
                --muted: #60706a;
                --line: #d9e2dc;
                --teal: #227c70;
                --green: #2f8f46;
                --amber: #d9901f;
                --red: #c93f37;
                --violet: #6a5acd;
            }}

            .stApp {{
                background:
                    linear-gradient(180deg, rgba(247, 249, 247, 0.96), rgba(239, 245, 241, 0.98));
                color: var(--ink);
            }}

            .block-container {{
                max-width: 1440px;
                padding-top: 1.25rem;
                padding-bottom: 3rem;
            }}

            section[data-testid="stSidebar"] {{
                background: #eef4f0;
                border-right: 1px solid var(--line);
            }}

            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {{
                color: var(--ink);
            }}

            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp label,
            div[data-testid="stWidgetLabel"] p {{
                color: var(--ink);
            }}

            div[data-baseweb="select"] > div {{
                background: #ffffff;
                border-color: #c9d7cf;
            }}

            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input {{
                color: var(--ink);
            }}

            div[data-testid="stMetric"] {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 10px 28px rgba(23, 33, 27, 0.07);
            }}

            .hero {{
                min-height: 250px;
                display: flex;
                align-items: flex-end;
                margin-bottom: 1.25rem;
                padding: 2rem;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.34);
                background-image:
                    linear-gradient(90deg, rgba(12, 18, 15, 0.90), rgba(12, 18, 15, 0.52)),
                    url("{HERO_IMAGE_URL}");
                background-size: cover;
                background-position: center;
                box-shadow: 0 22px 60px rgba(23, 33, 27, 0.18);
            }}

            .hero-content {{
                max-width: 820px;
            }}

            .eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.42rem 0.72rem;
                margin-bottom: 0.85rem;
                border: 1px solid rgba(255, 255, 255, 0.36);
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.16);
                color: #ffffff;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0;
                text-transform: uppercase;
            }}

            .hero h1 {{
                margin: 0;
                color: #ffffff;
                font-size: 2.45rem;
                line-height: 1.1;
                letter-spacing: 0;
            }}

            .hero p {{
                max-width: 780px;
                margin: 0.75rem 0 0;
                color: rgba(255, 255, 255, 0.86);
                font-size: 1rem;
                line-height: 1.6;
            }}

            .source-pill {{
                display: inline-flex;
                max-width: 100%;
                margin-top: 1rem;
                padding: 0.45rem 0.7rem;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.14);
                color: rgba(255, 255, 255, 0.88);
                font-size: 0.84rem;
                overflow-wrap: anywhere;
            }}

            .kpi-card {{
                min-height: 132px;
                padding: 1rem;
                border-radius: 8px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-top: 4px solid var(--accent);
                box-shadow: 0 12px 32px rgba(23, 33, 27, 0.08);
            }}

            .kpi-label {{
                color: var(--muted);
                font-size: 0.83rem;
                font-weight: 700;
                letter-spacing: 0;
                text-transform: uppercase;
            }}

            .kpi-value {{
                margin-top: 0.55rem;
                color: var(--ink);
                font-size: 2rem;
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: 0;
            }}

            .kpi-note {{
                margin-top: 0.55rem;
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.35;
            }}

            .insight-strip {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin: 0.4rem 0 1.15rem;
            }}

            .insight-chip {{
                padding: 0.55rem 0.72rem;
                border-radius: 8px;
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.82);
                color: var(--ink);
                font-size: 0.9rem;
                box-shadow: 0 8px 20px rgba(23, 33, 27, 0.05);
            }}

            .insight-chip strong {{
                color: var(--teal);
            }}

            h2, h3 {{
                letter-spacing: 0;
            }}

            div[data-testid="stAltairChart"] {{
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.75rem;
                background: var(--panel);
                box-shadow: 0 12px 32px rgba(23, 33, 27, 0.07);
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
            }}

            div[data-testid="stDownloadButton"] button,
            div[data-testid="stBaseButton-secondary"] {{
                border-radius: 8px;
            }}

            @media (max-width: 780px) {{
                .hero {{
                    min-height: 290px;
                    padding: 1.25rem;
                }}

                .hero h1 {{
                    font-size: 1.85rem;
                }}

                .kpi-value {{
                    font-size: 1.65rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section(source_label: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-content">
                <div class="eyebrow">Road Safety Intelligence</div>
                <h1>Indian Roads Accident Dashboard</h1>
                <p>
                    Monitor accident patterns across cities, severity, causes, weather,
                    traffic density, time of day, and geolocation.
                </p>
                <div class="source-pill">Data source: {escape(source_label)}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent: {accent};">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            <div class="kpi-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def polish_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(background=CHART_PANEL)
        .configure(background=CHART_PANEL)
        .configure_axis(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            gridColor=CHART_GRID,
            domainColor=CHART_GRID,
            tickColor=CHART_GRID,
            labelFontSize=13,
            titleFontSize=14,
            labelFontWeight=600,
            titleFontWeight=700,
        )
        .configure_legend(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            orient="bottom",
            symbolStrokeWidth=0,
            labelFontSize=12,
            titleFontSize=13,
            titleFontWeight=700,
        )
        .configure_header(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            labelFontSize=13,
            titleFontSize=14,
        )
        .configure_title(
            color=CHART_TEXT,
            fontSize=16,
            fontWeight=700,
        )
        .configure_view(strokeWidth=0)
    )


@st.cache_data
def load_default_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def find_default_data_path() -> Path | None:
    if PROJECT_DATA_PATH.exists():
        return PROJECT_DATA_PATH
    if DOWNLOADS_DATA_PATH.exists():
        return DOWNLOADS_DATA_PATH
    return None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def yes_no_label(series: pd.Series, yes_label: str, no_label: str) -> pd.Series:
    values = series.fillna(0).astype(str).str.strip().str.lower()
    mask = values.isin({"1", "true", "yes", "y"})
    return mask.map({True: yes_label, False: no_label})


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)

    numeric_columns = [
        "accident_id",
        "latitude",
        "longitude",
        "hour",
        "is_weekend",
        "lanes",
        "traffic_signal",
        "temperature",
        "vehicles_involved",
        "casualties",
        "is_peak_hour",
        "risk_score",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df["month_name"] = df["date"].dt.month_name()

    if "festival" in df.columns:
        df["festival"] = df["festival"].fillna("None").replace("", "None")

    if "accident_severity" in df.columns:
        df["accident_severity"] = df["accident_severity"].astype(str).str.lower()
        df["severity_score"] = df["accident_severity"].map({"minor": 1, "major": 2, "fatal": 3})

    if "risk_score" in df.columns:
        df["risk_band"] = pd.cut(
            df["risk_score"],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.00],
            labels=["Low", "Moderate", "High", "Critical"],
        )

    if "is_weekend" in df.columns:
        df["week_type"] = yes_no_label(df["is_weekend"], "Weekend", "Weekday")

    if "traffic_signal" in df.columns:
        df["signal_status"] = yes_no_label(df["traffic_signal"], "Signal", "No signal")

    if "is_peak_hour" in df.columns:
        df["peak_status"] = yes_no_label(df["is_peak_hour"], "Peak hour", "Off peak")

    return df


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def apply_multiselect_filter(
    df: pd.DataFrame,
    column: str,
    label: str,
    default_limit: int | None = None,
) -> pd.DataFrame:
    values = option_values(df, column)
    if not values:
        return df

    default_values = values if default_limit is None else values[:default_limit]
    selected = st.sidebar.multiselect(label, values, default=default_values)
    if selected:
        return df[df[column].astype(str).isin(selected)]
    return df.iloc[0:0]


def count_rows(df: pd.DataFrame) -> int:
    return int(len(df))


def build_summary(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    summary = df.groupby(group_column, dropna=False).agg(
        accidents=("accident_id", "count") if "accident_id" in df.columns else (group_column, "size"),
        casualties=("casualties", "sum") if "casualties" in df.columns else (group_column, "size"),
        avg_risk=("risk_score", "mean") if "risk_score" in df.columns else (group_column, "size"),
    )
    return summary.reset_index()


uploaded_file = st.sidebar.file_uploader("Upload Indian roads CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    raw_data = read_uploaded_file(uploaded_file)
    source_label = uploaded_file.name
else:
    default_path = find_default_data_path()
    if default_path is None:
        st.error("Add `data/indian_roads_dataset.csv` or upload a CSV/Excel file from the sidebar.")
        st.stop()
    raw_data = load_default_data(str(default_path))
    source_label = str(default_path)

data = prepare_data(raw_data)

inject_styles()
hero_section(source_label)

st.sidebar.header("Filters")
filtered = data.copy()

if "date" in filtered.columns and filtered["date"].notna().any():
    min_date = filtered["date"].min().date()
    max_date = filtered["date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
        filtered = filtered[
            (filtered["date"].dt.date >= start_date)
            & (filtered["date"].dt.date <= end_date)
        ]

filtered = apply_multiselect_filter(filtered, "state", "State")
filtered = apply_multiselect_filter(filtered, "city", "City")
filtered = apply_multiselect_filter(filtered, "accident_severity", "Severity")
filtered = apply_multiselect_filter(filtered, "road_type", "Road type")
filtered = apply_multiselect_filter(filtered, "weather", "Weather")
filtered = apply_multiselect_filter(filtered, "cause", "Cause")
filtered = apply_multiselect_filter(filtered, "traffic_density", "Traffic density")
filtered = apply_multiselect_filter(filtered, "week_type", "Week type")
filtered = apply_multiselect_filter(filtered, "peak_status", "Peak status")

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_accidents = count_rows(filtered)
total_casualties = int(filtered["casualties"].sum()) if "casualties" in filtered.columns else 0
fatal_accidents = (
    int(filtered["accident_severity"].eq("fatal").sum())
    if "accident_severity" in filtered.columns
    else 0
)
avg_risk = filtered["risk_score"].mean() if "risk_score" in filtered.columns else None
peak_share = (
    filtered["is_peak_hour"].mean() * 100
    if "is_peak_hour" in filtered.columns and filtered["is_peak_hour"].notna().any()
    else None
)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
with metric_1:
    kpi_card("Accidents", f"{total_accidents:,}", "Filtered records", "#227c70")
with metric_2:
    kpi_card("Casualties", f"{total_casualties:,}", "Total people affected", "#c93f37")
with metric_3:
    kpi_card("Fatal accidents", f"{fatal_accidents:,}", "Highest severity cases", "#7f1d1d")
with metric_4:
    kpi_card("Average risk", f"{avg_risk:.2f}" if avg_risk is not None else "N/A", "Risk score mean", "#d9901f")
with metric_5:
    kpi_card("Peak-hour share", f"{peak_share:.1f}%" if peak_share is not None else "N/A", "Accidents in peak hours", "#6a5acd")

top_city = filtered["city"].mode().iat[0] if "city" in filtered.columns and not filtered["city"].mode().empty else "N/A"
top_cause = filtered["cause"].mode().iat[0] if "cause" in filtered.columns and not filtered["cause"].mode().empty else "N/A"
top_weather = (
    filtered["weather"].mode().iat[0]
    if "weather" in filtered.columns and not filtered["weather"].mode().empty
    else "N/A"
)

st.markdown(
    f"""
    <div class="insight-strip">
        <div class="insight-chip">Top city: <strong>{escape(str(top_city).title())}</strong></div>
        <div class="insight-chip">Top cause: <strong>{escape(str(top_cause).title())}</strong></div>
        <div class="insight-chip">Common weather: <strong>{escape(str(top_weather).title())}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

trend_col, severity_col = st.columns((1.35, 1))

with trend_col:
    st.subheader("Monthly Trend")
    if "month" in filtered.columns:
        monthly = filtered.dropna(subset=["month"]).groupby("month").agg(
            accidents=("accident_id", "count") if "accident_id" in filtered.columns else ("month", "size"),
            casualties=("casualties", "sum") if "casualties" in filtered.columns else ("month", "size"),
            avg_risk=("risk_score", "mean") if "risk_score" in filtered.columns else ("month", "size"),
        )
        monthly = monthly.reset_index()

        trend_metric = st.selectbox(
            "Trend metric",
            ["Accidents", "Casualties", "Average risk"],
            key="trend_metric",
        )
        y_column = {
            "Accidents": "accidents",
            "Casualties": "casualties",
            "Average risk": "avg_risk",
        }[trend_metric]

        trend_base = alt.Chart(monthly).encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y(f"{y_column}:Q", title=trend_metric),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip(f"{y_column}:Q", title=trend_metric, format=",.2f"),
            ],
        )
        trend_chart = (
            alt.layer(
                trend_base.mark_area(color="#2a9d8f", opacity=0.18),
                trend_base.mark_line(color="#227c70", strokeWidth=3),
                trend_base.mark_point(
                    color="#ffffff",
                    filled=True,
                    size=86,
                    stroke="#227c70",
                    strokeWidth=2,
                ),
            )
            .properties(height=330)
        )
        st.altair_chart(polish_chart(trend_chart), width="stretch")
    else:
        st.info("A `date` column is needed for the monthly trend.")

with severity_col:
    st.subheader("Severity Mix")
    if "accident_severity" in filtered.columns:
        severity_counts = (
            filtered["accident_severity"]
            .value_counts()
            .reindex(SEVERITY_ORDER)
            .dropna()
            .reset_index()
        )
        severity_counts.columns = ["severity", "accidents"]
        severity_domain_max = max(float(severity_counts["accidents"].max()) * 1.16, 1)

        severity_base = alt.Chart(severity_counts).encode(
            x=alt.X("severity:N", sort=SEVERITY_ORDER, title="Severity"),
            y=alt.Y(
                "accidents:Q",
                title="Accidents",
                scale=alt.Scale(domain=[0, severity_domain_max]),
            ),
            tooltip=[
                alt.Tooltip("severity:N", title="Severity"),
                alt.Tooltip("accidents:Q", title="Accidents", format=","),
            ],
        )
        severity_chart = (
            alt.layer(
                severity_base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    color=alt.Color(
                        "severity:N",
                        scale=alt.Scale(domain=SEVERITY_ORDER, range=SEVERITY_COLORS),
                        legend=None,
                    ),
                ),
                severity_base.mark_text(
                    dy=-8,
                    color=CHART_TEXT,
                    fontSize=13,
                    fontWeight=700,
                ).encode(text=alt.Text("accidents:Q", format=",")),
            )
            .properties(height=330)
        )
        st.altair_chart(polish_chart(severity_chart), width="stretch")
    else:
        st.info("An `accident_severity` column is needed for this chart.")

city_col, cause_col = st.columns(2)

with city_col:
    st.subheader("City Hotspots")
    if "city" in filtered.columns:
        rank_metric = st.selectbox(
            "Rank cities by",
            ["Accidents", "Casualties", "Average risk"],
            key="city_rank_metric",
        )
        metric_column = {
            "Accidents": "accidents",
            "Casualties": "casualties",
            "Average risk": "avg_risk",
        }[rank_metric]

        city_summary = build_summary(filtered, "city").sort_values(metric_column, ascending=False).head(10)
        city_domain_max = max(float(city_summary[metric_column].max()) * 1.18, 1)
        city_label_format = ".2f" if metric_column == "avg_risk" else ","
        city_base = alt.Chart(city_summary).encode(
            x=alt.X(
                f"{metric_column}:Q",
                title=rank_metric,
                scale=alt.Scale(domain=[0, city_domain_max]),
            ),
            y=alt.Y("city:N", sort="-x", title="City"),
            tooltip=[
                alt.Tooltip("city:N", title="City"),
                alt.Tooltip("accidents:Q", title="Accidents", format=","),
                alt.Tooltip("casualties:Q", title="Casualties", format=","),
                alt.Tooltip("avg_risk:Q", title="Average risk", format=".2f"),
            ],
        )
        city_chart = (
            alt.layer(
                city_base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    color=alt.Color(
                        "avg_risk:Q",
                        title="Avg risk",
                        scale=alt.Scale(range=["#2a9d8f", "#f4a261", "#e63946"]),
                    )
                ),
                city_base.mark_text(
                    align="left",
                    dx=6,
                    color=CHART_TEXT,
                    fontSize=12,
                    fontWeight=700,
                ).encode(text=alt.Text(f"{metric_column}:Q", format=city_label_format)),
            )
            .properties(height=360)
        )
        st.altair_chart(polish_chart(city_chart), width="stretch")
    else:
        st.info("A `city` column is needed for hotspot ranking.")

with cause_col:
    st.subheader("Causes By Severity")
    if {"cause", "accident_severity"}.issubset(filtered.columns):
        cause_severity = (
            filtered.groupby(["cause", "accident_severity"])
            .size()
            .reset_index(name="accidents")
        )
        top_causes = (
            filtered["cause"]
            .value_counts()
            .head(8)
            .index
            .tolist()
        )
        cause_severity = cause_severity[cause_severity["cause"].isin(top_causes)]
        cause_totals = cause_severity.groupby("cause", as_index=False)["accidents"].sum()
        cause_domain_max = max(float(cause_totals["accidents"].max()) * 1.18, 1)

        cause_base = alt.Chart(cause_severity).encode(
            x=alt.X(
                "accidents:Q",
                title="Accidents",
                stack="zero",
                scale=alt.Scale(domain=[0, cause_domain_max]),
            ),
            y=alt.Y("cause:N", sort="-x", title="Cause"),
            color=alt.Color(
                "accident_severity:N",
                sort=SEVERITY_ORDER,
                scale=alt.Scale(domain=SEVERITY_ORDER, range=SEVERITY_COLORS),
                title="Severity",
            ),
            tooltip=[
                alt.Tooltip("cause:N", title="Cause"),
                alt.Tooltip("accident_severity:N", title="Severity"),
                alt.Tooltip("accidents:Q", title="Accidents", format=","),
            ],
        )
        cause_chart = (
            alt.layer(
                cause_base.mark_bar(
                    cornerRadiusTopRight=3,
                    cornerRadiusBottomRight=3,
                    stroke="#ffffff",
                    strokeWidth=1,
                ),
                alt.Chart(cause_totals)
                .mark_text(
                    align="left",
                    dx=6,
                    color=CHART_TEXT,
                    fontSize=12,
                    fontWeight=700,
                )
                .encode(
                    x=alt.X(
                        "accidents:Q",
                        scale=alt.Scale(domain=[0, cause_domain_max]),
                    ),
                    y=alt.Y("cause:N", sort="-x"),
                    text=alt.Text("accidents:Q", format=","),
                ),
            )
            .properties(height=360)
            .resolve_scale(color="shared")
        )
        st.altair_chart(polish_chart(cause_chart), width="stretch")
    else:
        st.info("Cause and severity columns are needed for this chart.")

heatmap_col, road_col = st.columns((1.2, 1))

with heatmap_col:
    st.subheader("Risk By Day And Hour")
    if {"day_of_week", "hour", "risk_score"}.issubset(filtered.columns):
        heatmap = (
            filtered.dropna(subset=["day_of_week", "hour", "risk_score"])
            .groupby(["day_of_week", "hour"], as_index=False)["risk_score"]
            .mean()
        )
        heatmap_base = alt.Chart(heatmap).encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("day_of_week:N", sort=DAY_ORDER, title="Day"),
            tooltip=[
                alt.Tooltip("day_of_week:N", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("risk_score:Q", title="Average risk", format=".2f"),
            ],
        )
        heatmap_chart = (
            alt.layer(
                heatmap_base.mark_rect(stroke="#ffffff", strokeWidth=1).encode(
                    color=alt.Color(
                        "risk_score:Q",
                        title="Average risk",
                        scale=alt.Scale(scheme="yelloworangered"),
                    ),
                ),
                heatmap_base.mark_text(fontSize=10, fontWeight=700).encode(
                    text=alt.Text("risk_score:Q", format=".2f"),
                    color=alt.condition(
                        alt.datum.risk_score > 0.56,
                        alt.value("#ffffff"),
                        alt.value(CHART_TEXT),
                    ),
                ),
            )
            .properties(height=330)
        )
        st.altair_chart(polish_chart(heatmap_chart), width="stretch")
    else:
        st.info("Day, hour, and risk score columns are needed for the heatmap.")

with road_col:
    st.subheader("Road And Weather Profile")
    profile_dimension = st.selectbox(
        "Group accidents by",
        ["road_type", "weather", "visibility", "traffic_density", "signal_status", "risk_band"],
        format_func=lambda value: value.replace("_", " ").title(),
    )

    if profile_dimension in filtered.columns:
        profile = (
            filtered[profile_dimension]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        profile.columns = [profile_dimension, "accidents"]
        profile_domain_max = max(float(profile["accidents"].max()) * 1.16, 1)
        profile_base = alt.Chart(profile).encode(
            x=alt.X(
                f"{profile_dimension}:N",
                sort="-y",
                title=profile_dimension.replace("_", " ").title(),
            ),
            y=alt.Y(
                "accidents:Q",
                title="Accidents",
                scale=alt.Scale(domain=[0, profile_domain_max]),
            ),
            tooltip=[
                alt.Tooltip(f"{profile_dimension}:N", title=profile_dimension.replace("_", " ").title()),
                alt.Tooltip("accidents:Q", title="Accidents", format=","),
            ],
        )
        profile_chart = (
            alt.layer(
                profile_base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    color=alt.Color(
                        f"{profile_dimension}:N",
                        legend=None,
                        scale=alt.Scale(scheme="set2"),
                    ),
                ),
                profile_base.mark_text(
                    dy=-8,
                    color=CHART_TEXT,
                    fontSize=12,
                    fontWeight=700,
                ).encode(text=alt.Text("accidents:Q", format=",")),
            )
            .properties(height=330)
        )
        st.altair_chart(polish_chart(profile_chart), width="stretch")
    else:
        st.info(f"`{profile_dimension}` is not available in this data.")

st.subheader("Accident Map")
if {"latitude", "longitude"}.issubset(filtered.columns):
    map_data = filtered.dropna(subset=["latitude", "longitude"]).rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    if len(map_data) > 5000:
        map_data = map_data.sample(5000, random_state=42)

    st.map(map_data[["lat", "lon"]], zoom=4, width="stretch")
    st.caption(f"Showing {len(map_data):,} mapped records.")
else:
    st.info("Latitude and longitude columns are needed for the map.")

st.subheader("Filtered Records")
display_columns = [
    column
    for column in [
        "accident_id",
        "date",
        "time",
        "city",
        "state",
        "road_type",
        "weather",
        "traffic_density",
        "cause",
        "accident_severity",
        "vehicles_involved",
        "casualties",
        "risk_score",
    ]
    if column in filtered.columns
]
st.dataframe(filtered[display_columns], width="stretch", hide_index=True)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data",
    data=csv_data,
    file_name="filtered_indian_roads_data.csv",
    mime="text/csv",
)

with st.expander("Column Summary"):
    st.dataframe(filtered.describe(include="all").astype(str), width="stretch")

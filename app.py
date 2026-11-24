from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))
from metrics import evaluate_forecast, make_fourier_terms

DATA_DIR = ROOT_DIR / "data" / "processed"
RES_DIR = ROOT_DIR / "results"

st.set_page_config(
    page_title="Grid-RVF | Cross-Grid Energy Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# UNIFIED SAGE-MINT ENTERPRISE THEME & LOCKED FULL-WIDTH TOP HEADER
# -------------------------------------------------------------------------
THEME = {
    "canvas": "#DCEEE6",       # Soft Sage-Green consistent across all 4 tabs
    "surface": "#FFFFFF",      # Crisp white elevated cards, widgets & charts
    "navy": "#0F2922",         # Deep forest-slate primary text
    "slate": "#2C4A3E",        # Secondary label text
    "muted": "#527164",        # Muted captions & gridlines
    "border": "#BFD9CE",       # Crisp sage-tinted card borders
    "blue": "#0284C7",         # California / primary series accent
    "emerald": "#047857",      # Pakistan / primary energy accent
    "amber": "#D97706",        # Solar series accent
    "crimson": "#DC2626",      # Net thermal / trend accent
    "indigo": "#4F46E5",       # Benchmark accent
}

st.markdown(
    f"""
    <style>
    /* Apply Soft Sage-Green ONLY to main viewport & all tab panels (never to inputs/widgets) */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    div[data-baseweb="tab-panel"] {{
        background-color: {THEME["canvas"]} !important;
    }}

    /* Keep all interactive widgets, inputs, dropdowns, and dataframes crisp white */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    [data-testid="stDataFrame"],
    [data-testid="stExpander"] {{
        background-color: {THEME["surface"]} !important;
        color: {THEME["navy"]} !important;
    }}

    /* Hide native Streamlit top chrome & dock sidebar expand button cleanly BELOW the top bar */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        pointer-events: none;
    }}
    [data-testid="stSidebarCollapsedControl"] {{
        position: fixed !important;
        top: 6.1rem !important;
        left: 0.75rem !important;
        pointer-events: auto !important;
        z-index: 1003 !important;
        background-color: {THEME["surface"]} !important;
        border: 1px solid {THEME["border"]} !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 6px rgba(15, 41, 34, 0.12);
    }}

    /* Permanently Locked Full-Width Top Header Bar (Never Moves When Sidebar Toggles) */
    .locked-topbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 5.55rem;
        z-index: 1002;
        background: linear-gradient(135deg, #0B2520 0%, #133A32 100%);
        border-bottom: 3px solid {THEME["emerald"]};
        padding: 0.75rem 2.5rem;
        box-shadow: 0 4px 18px rgba(11, 37, 32, 0.18);
        display: flex;
        align-items: center;
    }}
    .topbar-inner {{
        width: 100%;
        max-width: 1380px;
        margin: 0 auto;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
    }}

    /* Left Filter Sidebar Docked Cleanly BELOW the Fixed Top Header Bar */
    section[data-testid="stSidebar"] {{
        top: 5.55rem !important;
        height: calc(100vh - 5.55rem) !important;
        background-color: {THEME["surface"]} !important;
        border-right: 1px solid {THEME["border"]};
        z-index: 1001 !important;
    }}
    section[data-testid="stSidebar"][aria-expanded="false"] {{
        width: 0px !important;
        min-width: 0px !important;
        max-width: 0px !important;
        border-right: none !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem !important;
        height: 100% !important;
        overflow-y: auto !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        background-color: {THEME["surface"]} !important;
    }}

    /* Symmetrically Balanced Main Workspace (Centers When Closed, Shifts Right When Sidebar Opens) */
    [data-testid="stMain"] {{
        width: 100% !important;
        flex: 1 1 auto !important;
    }}
    [data-testid="stMainBlockContainer"],
    .block-container {{
        width: 100% !important;
        max-width: 1380px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 6.75rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }}

    /* Ensure all tab panels stretch evenly across the balanced container */
    div[data-baseweb="tab-panel"] {{
        width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }}

    .banner-kicker {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #34D399;
        margin-bottom: 0.12rem;
    }}
    .banner-title {{
        font-size: 1.18rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.25;
    }}
    .banner-sub {{
        font-size: 0.77rem;
        color: #CBE3DA;
        margin-top: 0.12rem;
    }}
    .banner-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }}
    .pill {{
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #F8FAFC;
        padding: 0.28rem 0.65rem;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
    }}
    .pill strong {{
        color: #34D399;
    }}

    /* Clean White KPI Cards */
    div[data-testid="stMetric"] {{
        background-color: {THEME["surface"]} !important;
        border: 1px solid {THEME["border"]};
        border-top: 3px solid {THEME["emerald"]};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 6px rgba(15, 41, 34, 0.06);
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {THEME["muted"]} !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {THEME["navy"]} !important;
        font-size: 1.48rem !important;
        font-weight: 700 !important;
    }}

    /* Consistent Centered Navigation Tabs Bar */
    div[data-baseweb="tab-list"] {{
        background-color: {THEME["surface"]} !important;
        border: 1px solid {THEME["border"]};
        border-radius: 8px;
        padding: 0.28rem 0.45rem;
        gap: 0.3rem;
        margin-bottom: 1rem;
        justify-content: center;
        box-shadow: 0 1px 4px rgba(15, 41, 34, 0.06);
    }}
    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {THEME["muted"]} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        padding: 0.48rem 0.95rem !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: #DCEEE6 !important;
        color: {THEME["emerald"]} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {THEME["emerald"]} !important;
    }}

    /* White Card Header Strip for Sections */
    .card-header {{
        background-color: {THEME["surface"]} !important;
        border: 1px solid {THEME["border"]};
        border-left: 4px solid {THEME["emerald"]};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(15, 41, 34, 0.05);
    }}
    .card-header-title {{
        font-size: 0.94rem;
        font-weight: 700;
        color: {THEME["navy"]};
    }}
    .card-header-sub {{
        font-size: 0.79rem;
        color: {THEME["muted"]};
        margin-top: 0.1rem;
    }}

    /* Sidebar Stat Rows */
    .gov-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid {THEME["border"]};
        font-size: 0.82rem;
    }}
    .gov-label {{ color: {THEME["muted"]}; font-weight: 500; }}
    .gov-val {{ color: {THEME["navy"]}; font-weight: 700; }}

    /* Responsive adjustments on narrower screens */
    @media (max-width: 1100px) {{
        .locked-topbar {{
            height: auto;
            min-height: 5.55rem;
            padding: 0.75rem 1.25rem;
        }}
        .block-container,
        [data-testid="stMainBlockContainer"] {{
            padding-top: 8.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig: go.Figure, title: str = "", height: int = 430, show_legend: bool = True) -> go.Figure:
    """Consistent white-card chart styling with generous top spacing so title & legend never collide."""
    top_margin = 78 if (title and show_legend) else (48 if title else 28)
    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=14, color=THEME["navy"], family="Segoe UI, Inter, sans-serif"),
        ),
        paper_bgcolor=THEME["surface"],
        plot_bgcolor=THEME["surface"],
        font=dict(color=THEME["slate"], family="Segoe UI, Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=24, r=24, t=top_margin, b=28),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=11, color=THEME["slate"]),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EBF3EF",
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"], size=11),
        title_font=dict(color=THEME["slate"], size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EBF3EF",
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["muted"], size=11),
        title_font=dict(color=THEME["slate"], size=11),
    )
    return fig


# -------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed hourly telemetry...")
def load_datasets():
    cal = pd.read_parquet(DATA_DIR / "cal_hourly_2021_2026.parquet")
    cal["Time"] = pd.to_datetime(cal["Local_time"])
    cal["Hour_of_Day"] = cal["Local_Hour"]
    cal["Thermal_MW"] = cal["Gas_MW"]

    pak = pd.read_parquet(DATA_DIR / "pak_hourly_fy2025.parquet")
    pak["Time"] = pd.to_datetime(pak["Timestamp"])
    pak["Hour_of_Day"] = pak["Hour"]
    pak["Thermal_MW"] = pak["RLNG_MW"]
    return cal, pak


@st.cache_data
def load_saved_results():
    files = {
        "mstl": RES_DIR / "mstl_seasonal_strength.csv",
        "stationarity": RES_DIR / "stationarity_tests.csv",
        "granger": RES_DIR / "granger_causality_results.csv",
        "cal_bench": RES_DIR / "cal_forecast_benchmark.csv",
        "pak_bench": RES_DIR / "pak_forecast_validation.csv",
        "cross_grid": RES_DIR / "cross_grid_comparison.csv",
        "cal_lb": RES_DIR / "cal_ljung_box.csv",
        "pak_lb": RES_DIR / "pak_ljung_box.csv",
    }
    out = {}
    for k, p in files.items():
        if p.exists():
            out[k] = pd.read_csv(p, index_col=0 if k.endswith("_lb") else None)
    return out


def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    y = df["Demand_MW"]
    for lag in [1, 2, 3, 24, 25, 48, 72, 168, 336]:
        feat[f"lag_{lag}"] = y.shift(lag)
    feat["roll24_mean"] = y.shift(1).rolling(24).mean()
    feat["roll24_std"] = y.shift(1).rolling(24).std()
    feat["roll168_mean"] = y.shift(1).rolling(168).mean()
    feat["hour_sin"] = np.sin(2 * np.pi * df["Hour_of_Day"] / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * df["Hour_of_Day"] / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
    feat["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    return feat


@st.cache_resource(show_spinner="Calibrating gradient-boosted forecasting models...")
def train_xgb_engines(cal: pd.DataFrame, pak: pd.DataFrame):
    models = {}
    for key, df, cutoff in [
        ("CAL", cal, cal["Year"] <= 2025),
        ("PAK", pak, pak["Time"] < "2025-04-01 00:00:00"),
    ]:
        X_all = build_xgb_features(df)
        valid = cutoff & X_all.notnull().all(axis=1)
        m = XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_all.loc[valid], df.loc[valid, "Demand_MW"])
        models[key] = m
    return models


def recursive_xgb_predict(model: XGBRegressor, hist_df: pd.DataFrame, fut_df: pd.DataFrame, horizon: int) -> np.ndarray:
    buf = hist_df[["Hour_of_Day", "DayOfWeek", "Month", "Demand_MW"]].iloc[-400:].copy()
    preds = []
    for step in range(horizon):
        nxt = fut_df.iloc[[step]][["Hour_of_Day", "DayOfWeek", "Month"]].copy()
        nxt["Demand_MW"] = np.nan
        buf = pd.concat([buf, nxt], ignore_index=True)
        X_step = build_xgb_features(buf).iloc[[-1]]
        y_hat = float(model.predict(X_step)[0])
        buf.iloc[-1, buf.columns.get_loc("Demand_MW")] = y_hat
        preds.append(y_hat)
    return np.asarray(preds, dtype=float)


# -------------------------------------------------------------------------
# LOAD DATA & SIDEBAR CONTROLS
# -------------------------------------------------------------------------
cal_df, pak_df = load_datasets()
saved_res = load_saved_results()
xgb_engines = train_xgb_engines(cal_df, pak_df)

st.sidebar.markdown("#### GRID & TEMPORAL FILTERS")
grid_choice = st.sidebar.radio(
    "Balancing Authority",
    ["California ISO (EIA-930 | 2021–2026)", "Pakistan National Grid (ISMO | FY2025)"],
    index=0,
)

is_cal = grid_choice.startswith("California")
active_df = cal_df if is_cal else pak_df
grid_code = "CAL" if is_cal else "PAK"
grid_short = "California ISO (CAL)" if is_cal else "Pakistan National Grid (ISMO)"
thermal_label = "Natural Gas Peaking" if is_cal else "RLNG Thermal Peaking"

min_date = active_df["Time"].min().date()
max_date = active_df["Time"].max().date()
default_start = pd.Timestamp("2025-01-01").date() if is_cal else min_date

date_range = st.sidebar.date_input(
    "Observation Date Range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

season_filter = st.sidebar.selectbox(
    "Climatic Regime",
    ["Full Annual Cycle", "Summer Cooling (Jun–Aug)", "Winter Load (Dec–Feb)", "Shoulder Months (Mar–May, Sep–Nov)"],
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23)
    filt_df = active_df[(active_df["Time"] >= start_dt) & (active_df["Time"] <= end_dt)].copy()
else:
    filt_df = active_df.copy()

if season_filter == "Summer Cooling (Jun–Aug)":
    filt_df = filt_df[filt_df["Month"].isin([6, 7, 8])]
elif season_filter == "Winter Load (Dec–Feb)":
    filt_df = filt_df[filt_df["Month"].isin([12, 1, 2])]
elif season_filter.startswith("Shoulder"):
    filt_df = filt_df[~filt_df["Month"].isin([6, 7, 8, 12, 1, 2])]

st.sidebar.markdown("---")
st.sidebar.markdown("#### DATASET SUMMARY")
st.sidebar.markdown(
    f"""
    <div class="gov-row"><span class="gov-label">Filtered Sample</span><span class="gov-val">{len(filt_df):,} hrs</span></div>
    <div class="gov-row"><span class="gov-label">Total Archive</span><span class="gov-val">{len(active_df):,} hrs</span></div>
    <div class="gov-row"><span class="gov-label">Archive Mean Load</span><span class="gov-val">{active_df['Demand_MW'].mean():,.0f} MW</span></div>
    <div class="gov-row"><span class="gov-label">Validation Split</span><span class="gov-val">Out-of-Sample</span></div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# LOCKED FULL-WIDTH EXECUTIVE HEADER BAR (STATIC POSITION)
# -------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="locked-topbar">
        <div class="topbar-inner">
            <div>
                <div class="banner-kicker">GRID-RVF // CROSS-GRID POWER SYSTEMS ANALYTICS</div>
                <div class="banner-title">Multi-Seasonal Renewable Variability &amp; Thermal Peaking Intelligence</div>
                <div class="banner-sub">Comparative Time-Series Study: U.S. EIA-930 California (2021–2026) &amp; Pakistan ISMO (FY2025)</div>
            </div>
            <div class="banner-pills">
                <div class="pill">GRID: <strong>{grid_short}</strong></div>
                <div class="pill">SLICE: <strong>{len(filt_df):,} HRS</strong></div>
                <div class="pill">MODELS: <strong>MSTL | FOURIER-ARIMA | XGBOOST</strong></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if len(filt_df) < 48:
    st.warning("Selected filter contains fewer than 48 hours. Please widen the date range or climatic regime.")
    st.stop()

# -------------------------------------------------------------------------
# EXECUTIVE KPI RIBBON
# -------------------------------------------------------------------------
hr_prof = filt_df.groupby("Hour_of_Day")[["Demand_MW", "Solar_MW", "Thermal_MW", "Net_Thermal_Load_MW"]].mean()
noon_dem = hr_prof.loc[12, "Demand_MW"] if 12 in hr_prof.index else filt_df["Demand_MW"].mean()
eve_dem = hr_prof.loc[20, "Demand_MW"] if 20 in hr_prof.index else filt_df["Demand_MW"].max()
evening_ramp = eve_dem - noon_dem
thermal_ramp = (
    (hr_prof.loc[20, "Thermal_MW"] - hr_prof.loc[12, "Thermal_MW"])
    if (12 in hr_prof.index and 20 in hr_prof.index)
    else 0.0
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Peak System Demand", f"{filt_df['Demand_MW'].max():,.0f} MW", f"Mean Load: {filt_df['Demand_MW'].mean():,.0f} MW")
k2.metric("Maximum Solar Dispatch", f"{filt_df['Solar_MW'].max():,.0f} MW", f"12:00 PM Avg: {hr_prof.loc[12, 'Solar_MW']:,.0f} MW")
k3.metric("Evening Demand Surge (12h–20h)", f"+{evening_ramp:,.0f} MW", f"20:00 Peak Avg: {eve_dem:,.0f} MW")
k4.metric(f"{thermal_label} Ramp", f"+{thermal_ramp:,.0f} MW", f"Peak Dispatch: {filt_df['Thermal_MW'].max():,.0f} MW")

st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# MAIN NAVIGATION TABS (CONSISTENT SAGE-GREEN CANVAS ACROSS ALL TABS)
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "01 | Duck Curve & Grid Telemetry",
    "02 | MSTL Seasonal Decomposition",
    "03 | Stationarity & Granger Causality",
    "04 | Live Forecasting Engine & Benchmarks",
])

# =========================================================================
# TAB 1: DUCK CURVE & TELEMETRY
# =========================================================================
with tab1:
    c_left, c_right = st.columns([1.15, 1.0])

    with c_left:
        hr_all = filt_df.groupby("Hour_of_Day")[
            ["Demand_MW", "Net_Thermal_Load_MW", "Solar_MW", "Wind_MW", "Thermal_MW"]
        ].mean().reset_index()

        fig_duck = go.Figure()
        fig_duck.add_trace(go.Scatter(
            x=hr_all["Hour_of_Day"], y=hr_all["Solar_MW"],
            name="Solar Generation", fill="tozeroy",
            line=dict(color=THEME["amber"], width=1.6),
            fillcolor="rgba(217, 119, 6, 0.16)",
        ))
        fig_duck.add_trace(go.Scatter(
            x=hr_all["Hour_of_Day"], y=hr_all["Demand_MW"],
            name="Total Grid Demand", line=dict(color=THEME["navy"], width=3),
        ))
        fig_duck.add_trace(go.Scatter(
            x=hr_all["Hour_of_Day"], y=hr_all["Net_Thermal_Load_MW"],
            name="Net Thermal Load", line=dict(color=THEME["crimson"], width=2.4, dash="dash"),
        ))
        fig_duck.add_trace(go.Scatter(
            x=hr_all["Hour_of_Day"], y=hr_all["Thermal_MW"],
            name=thermal_label, line=dict(color=THEME["blue"], width=2.4),
        ))
        style_figure(fig_duck, title=f"Intraday Duck Curve & Evening Thermal Ramp ({grid_code})", height=440)
        fig_duck.update_xaxes(title_text="Hour of Day (00:00 – 23:00)", dtick=2)
        fig_duck.update_yaxes(title_text="Average Power (MW)")
        st.plotly_chart(fig_duck, use_container_width=True)

    with c_right:
        cal_norm = cal_df.groupby("Hour_of_Day")["Demand_MW"].mean()
        cal_norm = (cal_norm / cal_norm.mean()) * 100
        pak_norm = pak_df.groupby("Hour_of_Day")["Demand_MW"].mean()
        pak_norm = (pak_norm / pak_norm.mean()) * 100

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=cal_norm.index, y=cal_norm.values, name="California ISO (2021–2026)",
            line=dict(color=THEME["blue"], width=3),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=pak_norm.index, y=pak_norm.values, name="Pakistan National Grid (FY2025)",
            line=dict(color=THEME["emerald"], width=3),
        ))
        fig_cmp.add_hline(
            y=100, line_dash="dot", line_color=THEME["muted"],
            annotation_text="Daily Mean Baseline (100%)", annotation_font_color=THEME["muted"],
        )
        style_figure(fig_cmp, title="Normalized Cross-Grid Intraday Demand (% of Daily Mean)", height=440)
        fig_cmp.update_xaxes(title_text="Hour of Day (00:00 – 23:00)", dtick=2)
        fig_cmp.update_yaxes(title_text="Normalized Load Index (%)")
        st.plotly_chart(fig_cmp, use_container_width=True)

    ts_sample = filt_df.iloc[::2] if len(filt_df) > 12000 else filt_df
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_sample["Time"], y=ts_sample["Demand_MW"], name="Total Demand (MW)",
        line=dict(color=THEME["navy"], width=1.3),
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_sample["Time"], y=ts_sample["Thermal_MW"], name=thermal_label,
        line=dict(color=THEME["blue"], width=1.2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_sample["Time"], y=ts_sample["Solar_MW"], name="Solar Generation (MW)",
        line=dict(color=THEME["amber"], width=1.1),
    ))
    style_figure(fig_ts, title="Hourly Grid Dispatch & Load Trajectory (Selected Window)", height=350)
    fig_ts.update_xaxes(title_text="Timestamp")
    fig_ts.update_yaxes(title_text="Power (MW)")
    st.plotly_chart(fig_ts, use_container_width=True)

# =========================================================================
# TAB 2: MSTL SEASONAL DECOMPOSITION
# =========================================================================
with tab2:
    st.markdown(
        """
        <div class="card-header">
            <div class="card-header-title">Multiple Seasonal-Trend Decomposition (MSTL)</div>
            <div class="card-header-sub">Decomposing hourly load into structural trend, intraday (s = 24), intra-week (s = 168), and annual (s = 8,760) cycles.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "mstl" in saved_res:
        st.dataframe(saved_res["mstl"], use_container_width=True, hide_index=True)

    mstl_window = st.slider(
        "Decomposition Window Length (Trailing Days)",
        min_value=14, max_value=180, value=60, step=7,
    )
    sub_mstl = filt_df.iloc[-(mstl_window * 24) :].copy()
    mstl_fit = MSTL(sub_mstl["Demand_MW"].values, periods=[24, 168], iterate=2).fit()

    fig_mstl = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            "Observed Demand (MW)",
            "Trend Component",
            "Daily Seasonality (s = 24 Hours)",
            "Weekly Seasonality (s = 168 Hours)",
            "Irregular Remainder",
        ),
    )
    fig_mstl.add_trace(go.Scatter(x=sub_mstl["Time"], y=sub_mstl["Demand_MW"], line=dict(color=THEME["navy"], width=1.2)), row=1, col=1)
    fig_mstl.add_trace(go.Scatter(x=sub_mstl["Time"], y=mstl_fit.trend, line=dict(color=THEME["crimson"], width=2.2)), row=2, col=1)
    fig_mstl.add_trace(go.Scatter(x=sub_mstl["Time"], y=mstl_fit.seasonal[:, 0], line=dict(color=THEME["blue"], width=1.2)), row=3, col=1)
    fig_mstl.add_trace(go.Scatter(x=sub_mstl["Time"], y=mstl_fit.seasonal[:, 1], line=dict(color=THEME["emerald"], width=1.4)), row=4, col=1)
    fig_mstl.add_trace(go.Scatter(x=sub_mstl["Time"], y=mstl_fit.resid, line=dict(color=THEME["muted"], width=1.0)), row=5, col=1)
    style_figure(fig_mstl, height=760, show_legend=False)
    for ann in fig_mstl.layout.annotations:
        ann.font.color = THEME["navy"]
        ann.font.size = 12
    st.plotly_chart(fig_mstl, use_container_width=True)

# =========================================================================
# TAB 3: STATIONARITY & PRE-WHITENED GRANGER CAUSALITY
# =========================================================================
with tab3:
    col_s, col_g = st.columns([1.0, 1.15])

    with col_s:
        st.markdown(
            """
            <div class="card-header">
                <div class="card-header-title">Joint ADF &amp; KPSS Stationarity Suite</div>
                <div class="card-header-sub">Unit-root and trend-stationarity verification prior to ARIMA integration and causal inference.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "stationarity" in saved_res:
            st.dataframe(saved_res["stationarity"], use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="card-header" style="border-left-color: #047857; margin-top: 0.65rem;">
                <div class="card-header-title">Econometric Specification Finding</div>
                <div class="card-header-sub">Raw hourly demand rejects KPSS stationarity (p = 0.01) across both grids. First differencing (Δy_t) and seasonal differencing (Δ24 y_t) achieve strict joint stationarity (ADF p &lt; 0.0001, KPSS p = 0.10), confirming d = 1 integration order.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_g:
        st.markdown(
            """
            <div class="card-header">
                <div class="card-header-title">Pre-Whitened Granger Causality (Renewables → Thermal Peaking)</div>
                <div class="card-header-sub">SSR F-statistics after removing deterministic 24h and 168h Fourier cycles to eliminate spurious clock correlation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "granger" in saved_res:
            gc = saved_res["granger"]
            fig_gc = px.bar(
                gc, x="Lag_Hours", y="F_Stat", color="Grid", barmode="group",
                text="F_Stat",
                color_discrete_map={
                    "California (CAL)": THEME["blue"],
                    "Pakistan (ISMO)": THEME["emerald"],
                },
                labels={"Lag_Hours": "Lag Horizon (Hours)", "F_Stat": "Granger F-Statistic (SSR)"},
            )
            fig_gc.update_traces(texttemplate="%{text:.1f}", textposition="outside", textfont=dict(color=THEME["navy"]))
            style_figure(fig_gc, title="Pre-Whitened Granger F-Statistic Across Lags 1–6", height=390)
            st.plotly_chart(fig_gc, use_container_width=True)

            with st.expander("View Full Pre-Whitened Granger Causality Table (Lags 1–6)"):
                st.dataframe(gc, use_container_width=True, hide_index=True)

# =========================================================================
# TAB 4: LIVE FORECASTING ENGINE & CROSS-GRID BENCHMARKS
# =========================================================================
with tab4:
    st.markdown(
        f"""
        <div class="card-header">
            <div class="card-header-title">Interactive Out-of-Sample Forecasting Simulator — {grid_short}</div>
            <div class="card-header-sub">Run live multi-horizon forecasts on unseen test windows using classical harmonic and gradient-boosted lag models.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3 = st.columns([1.0, 1.5, 1.3])

    with f_col1:
        horizon_choice = st.radio(
            "Forecast Horizon",
            [24, 168],
            format_func=lambda h: f"{h} Hours ({'Day-Ahead' if h == 24 else 'Week-Ahead'})",
        )

    test_min_idx = (
        int(active_df.index[active_df["Year"] == 2026].min())
        if is_cal
        else int(active_df.index[active_df["Time"] >= "2025-04-01"].min())
    )
    test_max_idx = len(active_df) - horizon_choice - 1

    with f_col2:
        origin_idx = st.slider(
            "Out-of-Sample Forecast Origin Index",
            min_value=test_min_idx,
            max_value=test_max_idx,
            value=test_min_idx + (test_max_idx - test_min_idx) // 2,
            step=24,
        )
        origin_ts = active_df.loc[origin_idx, "Time"]
        st.caption(f"Origin Timestamp: {origin_ts}")

    with f_col3:
        models_selected = st.multiselect(
            "Models to Evaluate",
            ["XGBoost (Causal Lags)", "Fourier-ARIMA (2,1,1)", "Seasonal Naive (s=24)", "Holt-Winters (s=24)"],
            default=["XGBoost (Causal Lags)", "Fourier-ARIMA (2,1,1)", "Seasonal Naive (s=24)"],
        )

    if st.button("Execute Forecast Simulation", type="primary"):
        with st.spinner("Computing multi-model forecasts..."):
            y_hist = active_df["Demand_MW"].iloc[:origin_idx].to_numpy(dtype=float)
            y_true = active_df["Demand_MW"].iloc[origin_idx : origin_idx + horizon_choice].to_numpy(dtype=float)
            t_fut = active_df["Time"].iloc[origin_idx : origin_idx + horizon_choice]

            preds_dict = {}
            if is_cal:
                preds_dict["EIA Official Forecast"] = active_df["Forecast_EIA_MW"].iloc[origin_idx : origin_idx + horizon_choice].to_numpy(dtype=float)

            if "Seasonal Naive (s=24)" in models_selected:
                preds_dict["Seasonal Naive (s=24)"] = np.tile(y_hist[-24:], int(np.ceil(horizon_choice / 24)))[:horizon_choice]

            if "Holt-Winters (s=24)" in models_selected:
                hw = ExponentialSmoothing(y_hist[-1080:], trend="add", seasonal="add", seasonal_periods=24).fit(optimized=True)
                preds_dict["Holt-Winters (s=24)"] = np.asarray(hw.forecast(horizon_choice), dtype=float)

            if "Fourier-ARIMA (2,1,1)" in models_selected:
                f_spec = [(24, 3), (168, 2), (8760, 3)] if is_cal else [(24, 3), (168, 2)]
                X_f = make_fourier_terms(len(active_df), f_spec, start_idx=0)
                win = 1440
                ar = ARIMA(
                    y_hist[-win:],
                    exog=X_f.iloc[origin_idx - win : origin_idx].to_numpy(dtype=float),
                    order=(2, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                preds_dict["Fourier-ARIMA (2,1,1)"] = np.asarray(
                    ar.forecast(steps=horizon_choice, exog=X_f.iloc[origin_idx : origin_idx + horizon_choice].to_numpy(dtype=float)),
                    dtype=float,
                )

            if "XGBoost (Causal Lags)" in models_selected:
                preds_dict["XGBoost (Causal Lags)"] = recursive_xgb_predict(
                    xgb_engines[grid_code],
                    active_df.iloc[:origin_idx],
                    active_df.iloc[origin_idx : origin_idx + horizon_choice],
                    horizon=horizon_choice,
                )

            fig_fc = go.Figure()
            t_ctx = active_df["Time"].iloc[origin_idx - 48 : origin_idx]
            y_ctx = active_df["Demand_MW"].iloc[origin_idx - 48 : origin_idx]
            fig_fc.add_trace(go.Scatter(
                x=t_ctx, y=y_ctx, name="Observed History (48h)",
                line=dict(color=THEME["muted"], width=1.8),
            ))
            fig_fc.add_trace(go.Scatter(
                x=t_fut, y=y_true, name="Actual Out-of-Sample Load",
                line=dict(color=THEME["navy"], width=3),
            ))

            model_colors = {
                "XGBoost (Causal Lags)": THEME["emerald"],
                "Fourier-ARIMA (2,1,1)": THEME["blue"],
                "Seasonal Naive (s=24)": THEME["amber"],
                "Holt-Winters (s=24)": THEME["crimson"],
                "EIA Official Forecast": THEME["indigo"],
            }
            live_metrics = []
            for m_name, y_hat in preds_dict.items():
                fig_fc.add_trace(go.Scatter(
                    x=t_fut, y=y_hat, name=m_name,
                    line=dict(
                        color=model_colors.get(m_name, THEME["blue"]),
                        width=2.3,
                        dash="dot" if "EIA" in m_name else "solid",
                    ),
                ))
                live_metrics.append(evaluate_forecast(y_true, y_hat, model_name=m_name))

            style_figure(fig_fc, title=f"{horizon_choice}-Hour Out-of-Sample Forecast Trajectory ({origin_ts})", height=450)
            fig_fc.update_xaxes(title_text="Timestamp")
            fig_fc.update_yaxes(title_text="Grid Demand (MW)")
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown("**Simulated Window Error Metrics**")
            st.dataframe(pd.DataFrame(live_metrics), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="card-header">
            <div class="card-header-title">Rolling-Origin Cross-Grid Benchmark Summary</div>
            <div class="card-header-sub">Aggregated out-of-sample accuracy across 20 rolling origins in California (2026) and 15 rolling origins in Pakistan (FY2025).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)
    with b1:
        if "cal_bench" in saved_res:
            st.caption("California ISO (CAL 2026 Test Set)")
            st.dataframe(saved_res["cal_bench"], use_container_width=True, hide_index=True)
    with b2:
        if "pak_bench" in saved_res:
            st.caption("Pakistan National Grid (ISMO FY2025 Test Set)")
            st.dataframe(saved_res["pak_bench"], use_container_width=True, hide_index=True)

    if "cross_grid" in saved_res:
        st.caption("Scale-Free Cross-Grid Transferability Matrix (MAPE % & NRMSE %)")
        st.dataframe(saved_res["cross_grid"], use_container_width=True, hide_index=True)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import MSTL

plt.rcParams["figure.dpi"] = 130
sns.set_theme(style="whitegrid", context="notebook")

DATA_DIR = Path("data/processed")
FIG_DIR = Path("figures")
RES_DIR = Path("results")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def seasonal_strength(seasonal_component: np.ndarray, remainder: np.ndarray) -> float:
    """Hyndman's seasonal strength metric Fs in [0, 1]."""
    var_r = np.var(remainder, ddof=1)
    var_sr = np.var(seasonal_component + remainder, ddof=1)
    return float(max(0.0, 1.0 - (var_r / var_sr)))


def plot_cross_grid_duck_curve(cal: pd.DataFrame, pak: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: California (CAL 2021-2026)
    cal_hr = cal.groupby("Local_Hour")[
        ["Demand_MW", "Net_Thermal_Load_MW", "Solar_MW", "Gas_MW", "Hydro_MW"]
    ].mean()
    ax = axes[0]
    ax.plot(cal_hr.index, cal_hr["Demand_MW"], label="Total Demand", color="#1f2937", lw=2.5)
    ax.plot(cal_hr.index, cal_hr["Net_Thermal_Load_MW"], label="Net Thermal Load", color="#dc2626", ls="--", lw=2.2)
    ax.fill_between(cal_hr.index, 0, cal_hr["Solar_MW"], color="#f59e0b", alpha=0.35, label="Solar (NG: SUN)")
    ax.plot(cal_hr.index, cal_hr["Gas_MW"], label="Natural Gas Peaking", color="#2563eb", lw=2.0)
    ax.set_title("California (CAL 2021–2026): Midday Solar Trough & Evening Gas Ramp", fontweight="bold")
    ax.set_xlabel("Local Hour of Day (0–23)")
    ax.set_ylabel("Average Power (MW)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel B: Pakistan (ISMO FY2025)
    pak_hr = pak.groupby("Hour")[
        ["Demand_MW", "Net_Thermal_Load_MW", "Solar_MW", "Hydel_MW", "RLNG_MW"]
    ].mean()
    ax = axes[1]
    ax.plot(pak_hr.index, pak_hr["Demand_MW"], label="Total Demand (Excl. KE)", color="#1f2937", lw=2.5)
    ax.plot(pak_hr.index, pak_hr["Net_Thermal_Load_MW"], label="Net Thermal Load", color="#dc2626", ls="--", lw=2.2)
    ax.fill_between(pak_hr.index, 0, pak_hr["Solar_MW"], color="#f59e0b", alpha=0.45, label="Utility Solar (680 MW)")
    ax.plot(pak_hr.index, pak_hr["Hydel_MW"], label="Hydel Dispatch", color="#059669", lw=2.0)
    ax.plot(pak_hr.index, pak_hr["RLNG_MW"], label="RLNG Thermal Peaking", color="#2563eb", lw=2.0)
    ax.set_title("Pakistan (ISMO FY2025): Midday Dip & 8 PM Hydel/RLNG Surge", fontweight="bold")
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("Average Power (MW)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_cross_grid_duck_curve.png", dpi=300)
    plt.close(fig)


def plot_summer_vs_winter(cal: pd.DataFrame, pak: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cal_season = np.where(cal["Month"].isin([6, 7, 8]), "Summer (Jun–Aug)",
                 np.where(cal["Month"].isin([12, 1, 2]), "Winter (Dec–Feb)", "Shoulder"))
    for season, color in [("Summer (Jun–Aug)", "#dc2626"), ("Winter (Dec–Feb)", "#2563eb")]:
        sub = cal[cal_season == season].groupby("Local_Hour")["Demand_MW"].mean()
        axes[0].plot(sub.index, sub.values, label=season, color=color, lw=2.4)
    axes[0].set_title("California Demand: Summer Cooling vs. Winter", fontweight="bold")
    axes[0].set_xlabel("Local Hour of Day")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    pak_season = np.where(pak["Month"].isin([6, 7, 8]), "Summer (Jun–Aug)",
                 np.where(pak["Month"].isin([12, 1, 2]), "Winter (Dec–Feb)", "Shoulder"))
    for season, color in [("Summer (Jun–Aug)", "#dc2626"), ("Winter (Dec–Feb)", "#2563eb")]:
        sub = pak[pak_season == season].groupby("Hour")["Demand_MW"].mean()
        axes[1].plot(sub.index, sub.values, label=season, color=color, lw=2.4)
    axes[1].set_title("Pakistan Demand: Summer Cooling vs. Winter", fontweight="bold")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Demand (MW)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_summer_vs_winter_regimes.png", dpi=300)
    plt.close(fig)


def run_mstl_analysis(cal: pd.DataFrame, pak: pd.DataFrame) -> pd.DataFrame:
    mstl_cal = MSTL(cal["Demand_MW"].values, periods=[24, 168, 8760], iterate=2).fit()
    s_cal = mstl_cal.seasonal

    mstl_pak = MSTL(pak["Demand_MW"].values, periods=[24, 168], iterate=2).fit()
    s_pak = mstl_pak.seasonal

    strength_df = pd.DataFrame([
        {
            "Grid": "California (CAL 2021–2026)",
            "Daily_Strength_s24": round(seasonal_strength(s_cal[:, 0], mstl_cal.resid), 4),
            "Weekly_Strength_s168": round(seasonal_strength(s_cal[:, 1], mstl_cal.resid), 4),
            "Annual_Strength_s8760": round(seasonal_strength(s_cal[:, 2], mstl_cal.resid), 4),
        },
        {
            "Grid": "Pakistan (ISMO FY2025)",
            "Daily_Strength_s24": round(seasonal_strength(s_pak[:, 0], mstl_pak.resid), 4),
            "Weekly_Strength_s168": round(seasonal_strength(s_pak[:, 1], mstl_pak.resid), 4),
            "Annual_Strength_s8760": np.nan,
        },
    ])
    strength_df.to_csv(RES_DIR / "mstl_seasonal_strength.csv", index=False)

    # Save California 5-panel MSTL figure
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    components = [
        ("Observed Demand (MW)", cal["Demand_MW"], "#111827"),
        ("Trend Component", mstl_cal.trend, "#dc2626"),
        ("Daily Seasonality (s = 24)", s_cal[:, 0], "#2563eb"),
        ("Weekly Seasonality (s = 168)", s_cal[:, 1], "#059669"),
        ("Annual Seasonality (s = 8,760)", s_cal[:, 2], "#d97706"),
    ]
    for ax, (title, series, col) in zip(axes, components):
        ax.plot(cal["Local_time"], series, color=col, lw=0.7)
        ax.set_ylabel("MW", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_mstl_california.png", dpi=300)
    plt.close(fig)

    return strength_df


if __name__ == "__main__":
    cal = pd.read_parquet(DATA_DIR / "cal_hourly_2021_2026.parquet")
    pak = pd.read_parquet(DATA_DIR / "pak_hourly_fy2025.parquet")

    print("Generating Step 2 figures and MSTL decomposition...")
    plot_cross_grid_duck_curve(cal, pak)
    plot_summer_vs_winter(cal, pak)
    strength_df = run_mstl_analysis(cal, pak)

    print("\n=== MSTL SEASONAL STRENGTH SUMMARY (0 to 1) ===")
    print(strength_df.to_string(index=False))
    print("\nSaved figures to figures/ and results/mstl_seasonal_strength.csv")
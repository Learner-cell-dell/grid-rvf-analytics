from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_california_dataset(
    filename: str = "Region_CAL.xlsx",
    start_local: str = "2021-01-01 00:00:00",
    end_utc: str = "2026-11-18 08:00:00",
) -> pd.DataFrame:
    """Clean and harmonize U.S. EIA-930 California (CAL) hourly telemetry."""
    path = RAW_DIR / filename
    raw = pd.read_excel(path, sheet_name="Published Hourly Data")

    raw["UTC_time"] = pd.to_datetime(raw["UTC time"])
    raw["Local_time"] = pd.to_datetime(raw["Local time"])

    df = raw[
        (raw["Local_time"] >= start_local) & (raw["UTC_time"] <= end_utc)
    ].copy()
    df = df.sort_values("UTC_time").reset_index(drop=True)

    # 1. Clean isolated telemetry dropout (< 15,000 MW on 2024-11-02) via time interpolation
    df["Demand_MW"] = pd.to_numeric(df["Demand"], errors="coerce")
    df.loc[df["Demand_MW"] < 15000, "Demand_MW"] = np.nan
    df["Demand_MW"] = (
        df.set_index("UTC_time")["Demand_MW"].interpolate(method="time").values
    )

    # 2. Operator benchmark forecast & fuel generation series (clip nighttime inverter draw < 0)
    df["Forecast_EIA_MW"] = pd.to_numeric(df["Demand forecast"], errors="coerce")
    df["Solar_MW"] = pd.to_numeric(df["NG: SUN"], errors="coerce").clip(lower=0.0)
    df["Wind_MW"] = pd.to_numeric(df["NG: WND"], errors="coerce").clip(lower=0.0)
    df["Hydro_MW"] = pd.to_numeric(df["NG: WAT"], errors="coerce")
    df["Gas_MW"] = pd.to_numeric(df["NG: NG"], errors="coerce")
    df["Nuclear_MW"] = pd.to_numeric(df["NG: NUC"], errors="coerce").fillna(0.0)

    # 3. Domain grid features
    df["VRE_MW"] = df["Solar_MW"] + df["Wind_MW"]
    df["Renewables_MW"] = df["VRE_MW"] + df["Hydro_MW"]
    df["Net_Thermal_Load_MW"] = df["Demand_MW"] - df["Renewables_MW"] - df["Nuclear_MW"]
    df["Demand_Ramp_MW"] = df["Demand_MW"].diff()
    df["Gas_Ramp_MW"] = df["Gas_MW"].diff()

    # 4. Calendar attributes
    df["Local_Hour"] = df["Local_time"].dt.hour
    df["DayOfWeek"] = df["Local_time"].dt.dayofweek
    df["Month"] = df["Local_time"].dt.month
    df["Year"] = df["Local_time"].dt.year

    keep_cols = [
        "UTC_time", "Local_time", "Year", "Month", "DayOfWeek", "Local_Hour",
        "Demand_MW", "Forecast_EIA_MW", "Solar_MW", "Wind_MW", "Hydro_MW",
        "Gas_MW", "Nuclear_MW", "VRE_MW", "Renewables_MW",
        "Net_Thermal_Load_MW", "Demand_Ramp_MW", "Gas_Ramp_MW",
    ]
    return df[keep_cols]


def build_pakistan_dataset(
    demand_file: str = "system_demand.xlsx",
    solar_ref_file: str = "solar_generation.xlsx",
    ismo_file: str = "FY2025 Capacities & Hourly Generation_ismo_NEDM7w05fo.xlsx",
) -> pd.DataFrame:
    """Merge and harmonize Pakistan ISMO FY2025 hourly demand, fuel dispatch, and solar benchmark."""
    # 1. System Demand (MW)
    dem = pd.read_excel(RAW_DIR / demand_file, sheet_name="System Demand Profile ")
    dem.columns = ["Date", "Hour", "Demand_MW"]
    dem["Date"] = pd.to_datetime(dem["Date"])
    dem["Timestamp"] = dem["Date"] + pd.to_timedelta(dem["Hour"], unit="h")

    # 2. Sindh 100 MW Solar Engineering Benchmark (MW)
    sol_ref = pd.read_excel(RAW_DIR / solar_ref_file, sheet_name="Sheet1")
    sol_ref.columns = ["Month", "Day", "Hour", "Sindh_100MW_Ref_MW"]
    sol_ref["Year"] = np.where(sol_ref["Month"] >= 7, 2024, 2025)
    sol_ref["Timestamp"] = pd.to_datetime(sol_ref[["Year", "Month", "Day", "Hour"]])

    # 3. Aggregate all 11 fuel sheets from kWh -> MW
    xls_fy = pd.ExcelFile(RAW_DIR / ismo_file)
    fuel_sheets = [
        "Wind", "Solar", "RLNG", "Dom Coal", "Nuclear",
        "Hydel", "Imp Coal", "Gas", "GENCO", "RFO", "Baggasse"
    ]

    df = dem[["Timestamp", "Date", "Hour", "Demand_MW"]].copy()
    for fuel in fuel_sheets:
        df_f = pd.read_excel(
            xls_fy, sheet_name=fuel, usecols=["Date", "Hour", "Generation (KWhs)"]
        )
        df_f["Timestamp"] = pd.to_datetime(df_f["Date"]) + pd.to_timedelta(df_f["Hour"], unit="h")
        agg = df_f.groupby("Timestamp", as_index=False)["Generation (KWhs)"].sum()
        col_name = fuel.replace(" ", "_") + "_MW"
        agg[col_name] = agg["Generation (KWhs)"] / 1000.0
        df = pd.merge(df, agg[["Timestamp", col_name]], on="Timestamp", how="left")

    df = pd.merge(df, sol_ref[["Timestamp", "Sindh_100MW_Ref_MW"]], on="Timestamp", how="left")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 4. Domain grid blocks & ramps
    df["VRE_MW"] = df["Solar_MW"] + df["Wind_MW"]
    df["Renewables_MW"] = df["VRE_MW"] + df["Hydel_MW"]
    df["Thermal_Peaking_MW"] = df["RLNG_MW"] + df["RFO_MW"]
    df["Baseload_MW"] = df["Nuclear_MW"] + df["Dom_Coal_MW"] + df["Imp_Coal_MW"]
    df["Net_Thermal_Load_MW"] = df["Demand_MW"] - df["Renewables_MW"] - df["Baseload_MW"]
    df["Demand_Ramp_MW"] = df["Demand_MW"].diff()
    df["RLNG_Ramp_MW"] = df["RLNG_MW"].diff()

    # 5. Calendar attributes
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    df["Year"] = df["Timestamp"].dt.year

    return df


if __name__ == "__main__":
    print("Building California (CAL 2021-2026) dataset...")
    cal_df = build_california_dataset()
    cal_path = PROCESSED_DIR / "cal_hourly_2021_2026.parquet"
    cal_df.to_parquet(cal_path, index=False)
    print(f"Saved {cal_path} -> shape={cal_df.shape}")

    print("Building Pakistan (ISMO FY2025) dataset...")
    pak_df = build_pakistan_dataset()
    pak_path = PROCESSED_DIR / "pak_hourly_fy2025.parquet"
    pak_df.to_parquet(pak_path, index=False)
    print(f"Saved {pak_path} -> shape={pak_df.shape}")
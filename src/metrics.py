import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Compute standard time-series accuracy metrics including scale-free NRMSE & sMAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100.0
    nrmse = (rmse / np.mean(y_true)) * 100.0  # Normalized RMSE (%) for cross-grid comparison

    return {
        "Model": model_name,
        "MAE_MW": round(mae, 2),
        "RMSE_MW": round(rmse, 2),
        "MAPE_pct": round(mape, 2),
        "sMAPE_pct": round(smape, 2),
        "NRMSE_pct": round(nrmse, 2),
    }


def make_fourier_terms(n_steps: int, periods_orders: list[tuple[float, int]], start_idx: int = 0) -> pd.DataFrame:
    """Generate deterministic sin/cos Fourier harmonics for multiple seasonalities (e.g., 24, 168, 8760)."""
    t = np.arange(start_idx, start_idx + n_steps)
    cols = {}
    for period, order in periods_orders:
        for k in range(1, order + 1):
            angle = 2.0 * np.pi * k * t / period
            cols[f"sin_{int(period)}_k{k}"] = np.sin(angle)
            cols[f"cos_{int(period)}_k{k}"] = np.cos(angle)
    return pd.DataFrame(cols)


def residual_diagnostics(residuals: np.ndarray, lags: list[int] = [24, 48, 168]) -> pd.DataFrame:
    """Run Ljung-Box autocorrelation test on forecast residuals."""
    lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
    return lb.round(4)
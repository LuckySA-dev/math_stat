"""
XAU/USD Time Series Analysis (2016-2025)
Core analysis module: descriptive stats + 4 forecasting methods
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ──────────────────────────────────────────────
def load_data(path: str = "datasets/xauusd_2016-2025.csv") -> pd.DataFrame:
    """Load cleaned XAU/USD CSV and return a time-indexed DataFrame."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df.set_index("date", inplace=True)

    # Convert change % string to float
    df["change %"] = df["change %"].str.rstrip("%").astype(float) / 100

    # Ensure numeric columns
    for col in ["price", "open", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def monthly_resample(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to monthly averages (better for time-series models)."""
    monthly = df.resample("MS").agg({
        "price": "mean",
        "open": "first",
        "high": "max",
        "low": "min",
        "change %": "sum",
    }).dropna()
    monthly.columns = ["avg_price", "open", "high", "low", "monthly_change"]
    return monthly


# ──────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return comprehensive descriptive statistics."""
    stats = df[["price", "open", "high", "low"]].describe()
    extras = pd.DataFrame({
        "price": {
            "median": df["price"].median(),
            "skewness": df["price"].skew(),
            "kurtosis": df["price"].kurtosis(),
            "variance": df["price"].var(),
            "range": df["price"].max() - df["price"].min(),
            "IQR": df["price"].quantile(0.75) - df["price"].quantile(0.25),
            "cv (%)": (df["price"].std() / df["price"].mean()) * 100,
        },
        "open": {
            "median": df["open"].median(),
            "skewness": df["open"].skew(),
            "kurtosis": df["open"].kurtosis(),
            "variance": df["open"].var(),
            "range": df["open"].max() - df["open"].min(),
            "IQR": df["open"].quantile(0.75) - df["open"].quantile(0.25),
            "cv (%)": (df["open"].std() / df["open"].mean()) * 100,
        },
        "high": {
            "median": df["high"].median(),
            "skewness": df["high"].skew(),
            "kurtosis": df["high"].kurtosis(),
            "variance": df["high"].var(),
            "range": df["high"].max() - df["high"].min(),
            "IQR": df["high"].quantile(0.75) - df["high"].quantile(0.25),
            "cv (%)": (df["high"].std() / df["high"].mean()) * 100,
        },
        "low": {
            "median": df["low"].median(),
            "skewness": df["low"].skew(),
            "kurtosis": df["low"].kurtosis(),
            "variance": df["low"].var(),
            "range": df["low"].max() - df["low"].min(),
            "IQR": df["low"].quantile(0.75) - df["low"].quantile(0.25),
            "cv (%)": (df["low"].std() / df["low"].mean()) * 100,
        },
    })
    return pd.concat([stats, extras])


# ──────────────────────────────────────────────
# 3. TIME SERIES DECOMPOSITION
# ──────────────────────────────────────────────
def decompose_series(monthly: pd.DataFrame, period: int = 12):
    """Decompose monthly avg_price into trend, seasonal, residual."""
    result = seasonal_decompose(monthly["avg_price"], model="additive", period=period)
    return result


# ──────────────────────────────────────────────
# 4. FORECASTING METHODS  (≥ 4 methods)
# ──────────────────────────────────────────────
def split_data(monthly: pd.DataFrame, test_months: int = 6):
    """Split into train/test."""
    train = monthly.iloc[:-test_months]
    test = monthly.iloc[-test_months:]
    return train, test


def evaluate(actual, predicted, method_name: str) -> dict:
    """Compute MAE, RMSE, MAPE for a forecast."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return {"method": method_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


# --- Method 1: Simple Moving Average (SMA) ---
def forecast_sma(train, test, window: int = 3):
    """Simple Moving Average forecast."""
    series = train["avg_price"]
    predictions = []
    history = list(series.values)
    for _ in range(len(test)):
        pred = np.mean(history[-window:])
        predictions.append(pred)
        history.append(pred)  # rolling forward
    return pd.Series(predictions, index=test.index, name="SMA")


# --- Method 2: Exponential Smoothing (Holt-Winters) ---
def forecast_exp_smoothing(train, test):
    """Holt-Winters Exponential Smoothing forecast."""
    model = ExponentialSmoothing(
        train["avg_price"],
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    ).fit(optimized=True)
    predictions = model.forecast(len(test))
    return predictions.rename("ExpSmoothing"), model


# --- Method 3: ARIMA ---
def forecast_arima(train, test, order=(2, 1, 2)):
    """ARIMA forecast."""
    model = ARIMA(train["avg_price"], order=order).fit()
    predictions = model.forecast(steps=len(test))
    return predictions.rename("ARIMA"), model


# --- Method 4: Prophet ---
def forecast_prophet(train, test):
    """Facebook Prophet forecast."""
    from prophet import Prophet
    prophet_df = train.reset_index().rename(columns={"date": "ds", "avg_price": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=len(test), freq="MS")
    forecast = m.predict(future)
    pred = forecast.set_index("ds").loc[test.index, "yhat"]
    return pred.rename("Prophet"), m


# --- Method 5: Double Exponential Smoothing (Holt) ---
def forecast_holt(train, test):
    """Holt's Double Exponential Smoothing (trend, no seasonal)."""
    model = ExponentialSmoothing(
        train["avg_price"],
        trend="add",
        seasonal=None,
    ).fit(optimized=True)
    predictions = model.forecast(len(test))
    return predictions.rename("Holt"), model


# ──────────────────────────────────────────────
# 5. RUN ALL MODELS & COMPARE
# ──────────────────────────────────────────────
def run_all_models(monthly: pd.DataFrame, test_months: int = 6):
    """Run all forecasting methods and return results dict + model params."""
    train, test = split_data(monthly, test_months)
    results = {}
    metrics = []
    model_params = {}

    # SMA
    sma_pred = forecast_sma(train, test)
    results["SMA"] = sma_pred
    metrics.append(evaluate(test["avg_price"], sma_pred, "SMA (3-month)"))
    last3 = train["avg_price"].values[-3:]
    model_params["SMA"] = {
        "window": 3,
        "last_values": [round(float(v), 2) for v in last3],
        "first_pred": round(float(np.mean(last3)), 2),
    }

    # Exponential Smoothing (Holt-Winters)
    es_pred, es_fit = forecast_exp_smoothing(train, test)
    results["ExpSmoothing"] = es_pred
    metrics.append(evaluate(test["avg_price"], es_pred, "Holt-Winters"))
    model_params["ExpSmoothing"] = {
        "alpha": round(float(es_fit.params["smoothing_level"]), 4),
        "beta": round(float(es_fit.params["smoothing_trend"]), 4),
        "gamma": round(float(es_fit.params["smoothing_seasonal"]), 4),
        "seasonal_periods": 12,
        "initial_level": round(float(es_fit.params["initial_level"]), 2),
        "initial_trend": round(float(es_fit.params["initial_trend"]), 2),
        "sse": round(float(es_fit.sse), 2),
        "aic": round(float(es_fit.aic), 2),
        "bic": round(float(es_fit.bic), 2),
    }

    # ARIMA
    arima_pred, arima_fit = forecast_arima(train, test)
    results["ARIMA"] = arima_pred
    metrics.append(evaluate(test["avg_price"], arima_pred, "ARIMA(2,1,2)"))
    model_params["ARIMA"] = {
        "order": (2, 1, 2),
        "ar_params": [round(float(c), 4) for c in arima_fit.arparams],
        "ma_params": [round(float(c), 4) for c in arima_fit.maparams],
        "sigma2": round(float(arima_fit.params.get('sigma2', 0)), 2),
        "aic": round(float(arima_fit.aic), 2),
        "bic": round(float(arima_fit.bic), 2),
    }

    # Prophet
    try:
        prophet_pred, prophet_fit = forecast_prophet(train, test)
        results["Prophet"] = prophet_pred
        metrics.append(evaluate(test["avg_price"], prophet_pred, "Prophet"))
        p_params = {
            "growth": "linear",
            "changepoint_prior_scale": round(float(prophet_fit.changepoint_prior_scale), 4),
            "n_changepoints": len(prophet_fit.changepoints) if prophet_fit.changepoints is not None else 0,
            "yearly_seasonality": True,
        }
        try:
            p_params["growth_rate"] = round(float(prophet_fit.params['k'][0][0]), 4)
            p_params["offset"] = round(float(prophet_fit.params['m'][0][0]), 4)
        except Exception:
            pass
        model_params["Prophet"] = p_params
    except Exception as e:
        print(f"Prophet error: {e}")

    # Holt
    holt_pred, holt_fit = forecast_holt(train, test)
    results["Holt"] = holt_pred
    metrics.append(evaluate(test["avg_price"], holt_pred, "Holt (Double Exp)"))
    model_params["Holt"] = {
        "alpha": round(float(holt_fit.params["smoothing_level"]), 4),
        "beta": round(float(holt_fit.params["smoothing_trend"]), 4),
        "initial_level": round(float(holt_fit.params["initial_level"]), 2),
        "initial_trend": round(float(holt_fit.params["initial_trend"]), 2),
        "sse": round(float(holt_fit.sse), 2),
        "aic": round(float(holt_fit.aic), 2),
        "bic": round(float(holt_fit.bic), 2),
    }

    metrics_df = pd.DataFrame(metrics).sort_values("MAPE (%)").reset_index(drop=True)
    return train, test, results, metrics_df, model_params


# ──────────────────────────────────────────────
# 6. HELPER: BEST MODEL
# ──────────────────────────────────────────────
def best_model(metrics_df: pd.DataFrame) -> str:
    return metrics_df.iloc[0]["method"]


# ──────────────────────────────────────────────
# MAIN (for standalone run)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print("=== Daily Data ===")
    print(df.head())
    print(f"\nDate range: {df.index.min()} → {df.index.max()}")
    print(f"Total days: {len(df)}")

    print("\n=== Descriptive Statistics ===")
    print(descriptive_stats(df).to_string())

    monthly = monthly_resample(df)
    print(f"\n=== Monthly Data ({len(monthly)} months) ===")
    print(monthly.head())

    train, test, results, metrics_df = run_all_models(monthly)
    print("\n=== Model Comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\n✓ Best model: {best_model(metrics_df)}")

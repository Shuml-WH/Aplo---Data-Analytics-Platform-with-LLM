import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")


def chronological_train_test_split(df, target_column, test_ratio=0.2):
    """Split data chronologically — last N% rows for test, no shuffle."""
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test, split_idx


def check_stationarity(series):
    """Run Augmented Dickey-Fuller test. Returns (is_stationary, adf_stat, p_value)."""
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < 0.05, result[0], result[1]


def determine_d(series, max_d=2):
    """Determine differencing order d via ADF test."""
    d = 0
    temp = series.dropna().copy()
    for _ in range(max_d):
        stationary, _, p_value = check_stationarity(temp)
        if stationary:
            break
        temp = temp.diff().dropna()
        d += 1
    return d


def auto_arima_order(target_series, exog=None, max_p=3, max_q=3, d=None):
    """Find best (p, d, q) via AIC minimization over a grid search."""
    if d is None:
        d = determine_d(target_series)

    best_aic = np.inf
    best_order = (0, d, 0)
    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)

    for p, q in product(p_range, q_range):
        if p == 0 and q == 0:
            continue
        try:
            model = SARIMAX(
                target_series,
                exog=exog,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=50)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order, best_aic


def evaluate_timeseries(y_actual, y_predicted):
    """Calculate MAPE, MAE, RMSE, R2 for time-series predictions."""
    actual = np.array(y_actual, dtype=float)
    predicted = np.array(y_predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {"mape": None, "mae": None, "rmse": None, "r2": None, "target_mean": None}

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    target_mean = float(np.mean(actual))

    nonzero = actual != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
    else:
        mape = None

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    return {"mape": mape, "mae": mae, "rmse": rmse, "r2": r2, "target_mean": target_mean}


def resample_to_regular(df, datetime_column, freq="D", agg_method="sum"):
    """Resample dataframe to regular frequency, forward-fill missing values."""
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
    df = df.dropna(subset=[datetime_column])
    df = df.set_index(datetime_column).sort_index()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if agg_method not in ("sum", "mean"):
        agg_method = "sum"
    resampled = getattr(df[numeric_cols].resample(freq), agg_method)()
    resampled[non_numeric_cols] = df[non_numeric_cols].resample(freq).first()
    resampled = resampled.ffill()

    resampled = resampled.reset_index()
    return resampled


def train_arimax_model(df, config, progress_callback=None):
    """Train ARIMAX model with auto order selection and optional exogenous features.

    Config keys:
        target_column: str
        datetime_column: str
        feature_columns: list[str]  (exogenous regressors, optional)
        forecast_horizon: int       (number of periods to forecast)
        frequency: str              ("D", "W", "M")
        test_size: float            (default 0.2)
        max_p: int                  (default 3)
        max_q: int                  (default 3)

    Returns dict with model_fit, metrics, forecast, config, etc.
    """
    target_column = config["target_column"]
    datetime_column = config["datetime_column"]
    feature_columns = config.get("feature_columns", [])
    forecast_horizon = config.get("forecast_horizon", 30)
    frequency = config.get("frequency", "D")
    test_size = config.get("test_size", 0.2)
    aggregation = config.get("aggregation", "sum")
    max_p = config.get("max_p", 3)
    max_q = config.get("max_q", 3)

    freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "D": "D", "W": "W", "M": "M"}
    freq = freq_map.get(frequency, frequency)

    def send_progress(data):
        if progress_callback:
            progress_callback(data)

    send_progress({"type": "progress", "percentage": 5, "step": "Preparing data..."})

    work_df = df.copy()
    work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
    work_df = work_df.dropna(subset=[datetime_column, target_column])
    work_df = work_df.sort_values(datetime_column).reset_index(drop=True)

    if len(work_df) < 30:
        raise ValueError(f"Insufficient data: {len(work_df)} rows (minimum 30 required).")

    work_df = resample_to_regular(work_df, datetime_column, freq, agg_method=aggregation)

    send_progress({"type": "progress", "percentage": 15, "step": "Checking stationarity..."})

    target_series = work_df[target_column].astype(float)

    exog_df = None
    if feature_columns:
        exog_cols = [c for c in feature_columns if c in work_df.columns and c != target_column and c != datetime_column]
        if exog_cols:
            exog_df = work_df[exog_cols].astype(float)
            for col in exog_df.columns:
                exog_df[col] = exog_df[col].fillna(exog_df[col].median())

    send_progress({"type": "progress", "percentage": 25, "step": "Finding best ARIMA order (auto)..."})

    best_order, best_aic = auto_arima_order(
        target_series, exog=exog_df, max_p=max_p, max_q=max_q
    )
    print(f"[timeseries] Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")

    send_progress({"type": "progress", "percentage": 40, "step": f"Fitting ARIMAX{best_order} model..."})

    train_data, test_data, split_idx = chronological_train_test_split(
        pd.concat([target_series, exog_df], axis=1) if exog_df is not None else target_series.to_frame(),
        target_column,
        test_ratio=test_size,
    )

    train_target = train_data[target_column] if isinstance(train_data, pd.DataFrame) else train_data.iloc[:, 0]
    test_target = test_data[target_column] if isinstance(test_data, pd.DataFrame) else test_data.iloc[:, 0]

    train_exog = train_data[exog_df.columns] if exog_df is not None and isinstance(train_data, pd.DataFrame) else None
    test_exog = test_data[exog_df.columns] if exog_df is not None and isinstance(test_data, pd.DataFrame) else None

    model = SARIMAX(
        train_target,
        exog=train_exog,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False, maxiter=100)

    send_progress({"type": "progress", "percentage": 65, "step": "Evaluating on test set..."})

    test_pred = model_fit.predict(
        start=len(train_target),
        end=len(train_target) + len(test_target) - 1,
        exog=test_exog,
    )

    test_metrics = evaluate_timeseries(test_target.values, test_pred.values)

    send_progress({"type": "progress", "percentage": 80, "step": "Fitting full model on all data..."})

    full_target = target_series
    full_exog = exog_df if exog_df is not None else None
    full_model = SARIMAX(
        full_target,
        exog=full_exog,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    full_model_fit = full_model.fit(disp=False, maxiter=100)

    send_progress({"type": "progress", "percentage": 90, "step": "Generating forecast..."})

    forecast_result = full_model_fit.get_forecast(steps=forecast_horizon, exog=None)
    forecast_values = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)

    last_date = work_df[datetime_column].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1) if freq == "D" else last_date + pd.tseries.frequencies.to_offset(freq),
        periods=forecast_horizon,
        freq=freq,
    )

    train_actual_dates = work_df[datetime_column].iloc[:split_idx].tolist()
    train_actual_values = work_df[target_column].iloc[:split_idx].tolist()
    test_actual_dates = work_df[datetime_column].iloc[split_idx:].tolist()
    test_actual_values = work_df[target_column].iloc[split_idx:].tolist()

    # Get in-sample predictions for training data
    train_fitted_values = full_model_fit.fittedvalues.iloc[:split_idx].tolist()
    # Get test predictions
    test_pred_values = test_pred.values.tolist()

    forecast_data = {
        "train_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in train_actual_dates],
        "train_values": [float(v) for v in train_actual_values],
        "train_predicted": [float(v) for v in train_fitted_values],
        "test_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in test_actual_dates],
        "test_values": [float(v) for v in test_actual_values],
        "test_predicted": [float(v) for v in test_pred_values],
        "forecast_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in forecast_dates],
        "forecast_values": [float(v) for v in forecast_values.values],
        "lower_bound": [float(v) for v in forecast_ci.iloc[:, 0].values],
        "upper_bound": [float(v) for v in forecast_ci.iloc[:, 1].values],
    }

    send_progress({"type": "progress", "percentage": 100, "step": "Complete!"})

    return {
        "task_type": "timeseries",
        "target_column": target_column,
        "datetime_column": datetime_column,
        "feature_columns": exog_df.columns.tolist() if exog_df is not None else [],
        "order": list(best_order),
        "aic": float(best_aic),
        "train_rows": len(train_target),
        "test_rows": len(test_target),
        "forecast_horizon": forecast_horizon,
        "frequency": freq,
        "test_metrics": test_metrics,
        "forecast_data": forecast_data,
        "model_fit": full_model_fit,
    }


def generate_forecast(model_fit, periods, exog_future=None, frequency="D", last_date=None):
    """Generate forecast with confidence intervals from a fitted model."""
    import pandas as pd
    forecast_result = model_fit.get_forecast(steps=periods, exog=exog_future)
    values = forecast_result.predicted_mean
    ci = forecast_result.conf_int(alpha=0.05)

    result = {
        "values": values.tolist(),
        "lower_bound": ci.iloc[:, 0].tolist(),
        "upper_bound": ci.iloc[:, 1].tolist(),
    }

    if last_date is not None:
        freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "D": "D", "W": "W", "M": "M"}
        freq = freq_map.get(frequency, frequency)
        if freq == "D":
            start_date = last_date + pd.Timedelta(days=1)
        else:
            start_date = last_date + pd.tseries.frequencies.to_offset(freq)
        forecast_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        result["dates"] = [d.strftime("%Y-%m-%d") for d in forecast_dates]

    return result


def get_timeseries_info(model_state):
    """Extract metadata from trained time-series model state."""
    config = model_state.get("config", {})
    results = model_state.get("results", {})
    model_type = model_state.get("model_type", ["arimax"])
    best_model = model_state.get("best_model", "arimax")

    info = {
        "target_column": config.get("target_column"),
        "datetime_column": config.get("datetime_column"),
        "feature_columns": config.get("feature_columns", []),
        "forecast_horizon": config.get("forecast_horizon"),
        "frequency": config.get("frequency"),
        "model_type": model_type,
        "best_model_key": best_model,
    }

    # Support both old format (dict) and new format (list)
    if isinstance(results, list):
        info["results"] = results
        first = results[0] if results else {}
        info["order"] = first.get("order")
        info["aic"] = first.get("aic")
        info["train_rows"] = first.get("train_rows")
        info["test_rows"] = first.get("test_rows")
        info["test_metrics"] = first.get("test_metrics", {})
        info["forecast_data"] = first.get("forecast_data")
    else:
        info["results"] = [{
            "model_type": "arimax",
            "model_label": "ARIMAX",
            "test_metrics": results.get("test_metrics", {}),
            "forecast_data": results.get("forecast_data"),
            "train_rows": results.get("train_rows"),
            "test_rows": results.get("test_rows"),
            "order": results.get("order"),
            "aic": results.get("aic"),
        }]
        info["order"] = results.get("order")
        info["aic"] = results.get("aic")
        info["train_rows"] = results.get("train_rows")
        info["test_rows"] = results.get("test_rows")
        info["test_metrics"] = results.get("test_metrics", {})
        info["forecast_data"] = results.get("forecast_data")

    return info


def _create_xgboost_features(df, target_column, freq):
    """Auto-generate lag and date features for XGBoost time-series model."""
    from pandas.tseries.frequencies import to_offset

    result = df.copy()

    # Lag features (3 lags based on frequency)
    lag_periods = {"D": [1, 2, 3], "W": [1, 2, 3], "M": [1, 2, 3]}
    for lag in lag_periods.get(freq, [1, 2, 3]):
        result[f"target_lag_{lag}"] = result[target_column].shift(lag)

    # Rolling features - window adapts to frequency
    rolling_window = {"D": 7, "W": 4, "M": 3}.get(freq, 3)
    result["target_rolling_mean"] = result[target_column].rolling(window=rolling_window).mean()
    result["target_rolling_std"] = result[target_column].rolling(window=rolling_window).std()

    # Date features
    if "date" in result.columns:
        dt_col = "date"
    elif hasattr(result.index, "date"):
        result = result.reset_index()
        dt_col = result.columns[0]
    else:
        dt_col = None

    if dt_col and pd.api.types.is_datetime64_any_dtype(result[dt_col]):
        result["month"] = result[dt_col].dt.month
        result["quarter"] = result[dt_col].dt.quarter
        result["day_of_week"] = result[dt_col].dt.dayofweek
        result["is_month_end"] = result[dt_col].dt.is_month_end.astype(int)
        result["is_quarter_end"] = result[dt_col].dt.is_quarter_end.astype(int)
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    result = result.dropna()
    return result


def train_xgboost_model(df, config, progress_callback=None):
    """Train XGBoost model with auto-generated lag and date features.

    Config keys:
        target_column: str
        datetime_column: str
        feature_columns: list[str]  (ignored — XGBoost auto-generates features)
        forecast_horizon: int
        frequency: str ("D", "W", "M")
        test_size: float (default 0.2)
        aggregation: str ("sum" or "mean")

    Returns dict with test_metrics, forecast_data, etc.
    """
    from xgboost import XGBRegressor

    target_column = config["target_column"]
    datetime_column = config["datetime_column"]
    forecast_horizon = config.get("forecast_horizon", 30)
    frequency = config.get("frequency", "D")
    test_size = config.get("test_size", 0.2)
    aggregation = config.get("aggregation", "sum")

    freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "D": "D", "W": "W", "M": "M"}
    freq = freq_map.get(frequency, frequency)

    def send_progress(data):
        if progress_callback:
            progress_callback(data)

    send_progress({"type": "progress", "percentage": 5, "step": "Preparing data..."})

    work_df = df.copy()
    work_df[datetime_column] = pd.to_datetime(work_df[datetime_column], errors="coerce")
    work_df = work_df.dropna(subset=[datetime_column, target_column])
    work_df = work_df.sort_values(datetime_column).reset_index(drop=True)

    if len(work_df) < 30:
        raise ValueError(f"Insufficient data: {len(work_df)} rows (minimum 30 required).")

    # Rename datetime column to "date" for feature engineering
    work_df = work_df.rename(columns={datetime_column: "date"})
    work_df = resample_to_regular(work_df, "date", freq, agg_method=aggregation)

    send_progress({"type": "progress", "percentage": 15, "step": "Generating lag and date features..."})

    feature_df = _create_xgboost_features(work_df, target_column, freq)

    feature_cols = [c for c in feature_df.columns if c not in [target_column, "date"] and pd.api.types.is_numeric_dtype(feature_df[c])]

    send_progress({"type": "progress", "percentage": 25, "step": "Splitting data chronologically..."})

    n = len(feature_df)
    split_idx = int(n * (1 - test_size))
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df[target_column].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_column].values

    # Train single model with standard MSE
    send_progress({"type": "progress", "percentage": 30, "step": "Training XGBoost model..."})

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    send_progress({"type": "progress", "percentage": 55, "step": "Evaluating on test set..."})

    # Evaluate on test set
    pred_test = model.predict(X_test)
    test_metrics = evaluate_timeseries(y_test, pred_test)

    # Get training predictions for forecast_data
    train_pred = model.predict(X_train)

    # Compute residual std from last 7 test predictions for adaptive confidence intervals
    # Use percentage errors for relative confidence bands
    nonzero_mask = y_test != 0
    if nonzero_mask.sum() > 0:
        test_pct_errors = np.abs((y_test[nonzero_mask] - pred_test[nonzero_mask]) / y_test[nonzero_mask])
        last_n = min(7, len(test_pct_errors))
        recent_pct_errors = test_pct_errors[-last_n:]
        residual_std = float(np.std(recent_pct_errors)) if len(recent_pct_errors) > 1 else float(np.std(test_pct_errors))
    else:
        test_residuals = y_test - pred_test
        last_n = min(7, len(test_residuals))
        recent_residuals = test_residuals[-last_n:]
        residual_std = float(np.std(recent_residuals)) if len(recent_residuals) > 1 else float(np.std(test_residuals))

    send_progress({"type": "progress", "percentage": 70, "step": "Fitting full model on all data..."})

    # Refit model on full data for forecasting
    X_full = feature_df[feature_cols].values
    y_full = feature_df[target_column].values
    full_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    full_model.fit(X_full, y_full)

    send_progress({"type": "progress", "percentage": 85, "step": "Generating forecast..."})

    # Iterative multi-step forecast
    rolling_window = {"D": 7, "W": 4, "M": 3}.get(freq, 3)
    last_values = list(feature_df[target_column].values[-rolling_window:])
    forecast_values = []
    forecast_lower = []
    forecast_upper = []

    last_date = feature_df["date"].iloc[-1]
    offset = pd.tseries.frequencies.to_offset(freq)

    for step in range(forecast_horizon):
        # Build feature vector for this step
        row = {}
        row["target_lag_1"] = last_values[-1]
        row["target_lag_2"] = last_values[-2] if len(last_values) >= 2 else last_values[-1]
        row["target_lag_3"] = last_values[-3] if len(last_values) >= 3 else last_values[-2]
        recent = last_values[-rolling_window:] if len(last_values) >= rolling_window else last_values
        row["target_rolling_mean"] = np.mean(recent)
        row["target_rolling_std"] = np.std(recent) if len(recent) > 1 else 0

        # Date features for forecast date
        fdate = last_date + (step + 1) * offset
        row["month"] = fdate.month
        row["quarter"] = fdate.quarter
        row["day_of_week"] = fdate.dayofweek
        row["is_month_end"] = int(fdate.is_month_end)
        row["is_quarter_end"] = int(fdate.is_quarter_end)
        row["month_sin"] = np.sin(2 * np.pi * fdate.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * fdate.month / 12)

        x = np.array([[row.get(c, 0) for c in feature_cols]])

        pred = full_model.predict(x)[0]

        # Confidence intervals based on percentage error (1.96 = 95% CI)
        ci_factor = 1.96
        lower = pred * (1 - ci_factor * residual_std * np.sqrt(step + 1))
        upper = pred * (1 + ci_factor * residual_std * np.sqrt(step + 1))

        forecast_values.append(float(pred))
        forecast_lower.append(float(lower))
        forecast_upper.append(float(upper))
        last_values.append(pred)

    forecast_dates = pd.date_range(
        start=last_date + offset,
        periods=forecast_horizon,
        freq=freq,
    )

    train_actual_dates = work_df["date"].iloc[:split_idx].tolist()
    train_actual_values = work_df[target_column].iloc[:split_idx].tolist()
    test_actual_dates = work_df["date"].iloc[split_idx:].tolist()
    test_actual_values = work_df[target_column].iloc[split_idx:].tolist()

    # Get predictions for training and test sets
    train_predicted_values = train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred)
    test_predicted_values = pred_test.tolist() if hasattr(pred_test, 'tolist') else list(pred_test)

    forecast_data = {
        "train_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in train_actual_dates],
        "train_values": [float(v) for v in train_actual_values],
        "train_predicted": [float(v) for v in train_predicted_values],
        "test_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in test_actual_dates],
        "test_values": [float(v) for v in test_actual_values],
        "test_predicted": [float(v) for v in test_predicted_values],
        "forecast_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in forecast_dates],
        "forecast_values": forecast_values,
        "lower_bound": forecast_lower,
        "upper_bound": forecast_upper,
    }

    send_progress({"type": "progress", "percentage": 100, "step": "Complete!"})

    return {
        "task_type": "timeseries",
        "target_column": target_column,
        "datetime_column": datetime_column,
        "feature_columns": feature_cols,
        "order": None,
        "aic": None,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "forecast_horizon": forecast_horizon,
        "frequency": freq,
        "test_metrics": test_metrics,
        "forecast_data": forecast_data,
        "model_fit": full_model,
    }

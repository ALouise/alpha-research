import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from arch import arch_model


def predict_garch_returns(df, window):
    df = df.copy()
    if 'return_t' not in df.columns or 'return_t+1' not in df.columns:
        raise ValueError("The DataFrame must contain 'return_t' and 'return_t+1' columns.")
    predictions = []
    prediction_dates = []
    for t in range(window, len(df) - 1):
        train_data = df['return_t'].iloc[t - window:t]
        train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(train_data) < window:
            continue
        train_data = train_data * 100
        model = arch_model(train_data, vol='Garch', p=1, q=1, mean='Constant', dist='normal') # constant mean
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=1)
        mean_forecast = forecast.mean.iloc[-1, 0]

        predictions.append(mean_forecast/100)
        prediction_dates.append(df.index[t + 1])
    actual_returns = df['return_t+1'].reindex(prediction_dates)
    return pd.DataFrame({
        "predict_return_t": predictions,
        "return_t": actual_returns.values
    }, index=prediction_dates)

def predict_arima_returns(df, window, order=(1, 0, 1)):
    df = df.copy()

    if 'return_t' not in df.columns or 'return_t+1' not in df.columns:
        raise ValueError("The DataFrame must contain 'return_t' and 'return_t+1' columns.")

    predictions = []
    prediction_dates = []

    for t in range(window, len(df) - 1):
        train_data = df['return_t'].iloc[t - window:t] 
        model = ARIMA(train_data, order=order, enforce_invertibility=False )
        res = model.fit()
        forecast = res.forecast(steps=1).iloc[0]

        predictions.append(forecast)
        prediction_dates.append(df.index[t + 1])
    actual_returns = df['return_t+1'].reindex(prediction_dates)

    return pd.DataFrame({
        "predict_return_t": predictions,
        "return_t": actual_returns.values 
    }, index=prediction_dates)

def predict_linear_returns(df, window):
    result = predict_global_model_returns(df, window, LinearRegression())
    return result
    
def predict_xgbregressor_returns(df, window):
    result = predict_global_model_returns(df, window, XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1))
    return result

def predict_global_model_returns(df, window, model):
    if 'return_t+1' not in df.columns:
        raise ValueError("The DataFrame must contain a 'return_t+1' column.")

    features = [col for col in df.columns if col != 'return_t+1'] 
    predictions = []
    actuals = []
    index = []

    for t in range(window, len(df)):
        X_train = df[features].iloc[t - window:t]
        y_train = df['return_t+1'].iloc[t - window:t]

        X_pred = df[features].iloc[t:t+1]
        y_true = df['return_t+1'].iloc[t]

        if X_train.isnull().values.any() or X_pred.isnull().values.any() or pd.isna(y_true):
            continue

        model.fit(X_train, y_train)

        y_pred = model.predict(X_pred)[0]

        predictions.append(y_pred)
        actuals.append(y_true)
        index.append(df.index[t])

    return pd.DataFrame({
        "predict_return_t": predictions,
        "return_t": actuals
    }, index=index)
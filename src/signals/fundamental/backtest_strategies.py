import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_ticker_long_short(df, ticker):
    df_ticker = df[df["ticker"] == ticker]
    df_ticker = df_ticker.dropna(subset=["predict_return_qt", "return_qt"])
    df_ticker["position"] = df_ticker["predict_return_qt"].apply(lambda x: 1 if x > 0 else -1)
    df_ticker["strategy_return"] = df_ticker["position"] * df_ticker["return_qt"]
    df_ticker["cumulative_return"] = (1 + df_ticker["strategy_return"]).cumprod()
    df_ticker["cumulative_buy_hold"] = (1 + df_ticker["return_qt"]).cumprod()
    return df_ticker[["period_quarter", "strategy_return", "return_qt", "cumulative_return", "cumulative_buy_hold"]]

def backtest_all_tickers_long_short(df, TICKERS:list):
    all_backtest = []
    for ticker in TICKERS:
        all_backtest.append((backtest_ticker_long_short(df, ticker), ticker))
    return all_backtest

def compute_global_cumulative_return(all_backtest):
    all_curves = []

    for df, _ in all_backtest:
        temp = df[["cumulative_return"]].copy()
        temp = temp.rename(columns={"cumulative_return": "cum_return"})
        temp = temp.resample("D").interpolate()  # interpolation daily
        all_curves.append(temp)

    df_concat = pd.concat(all_curves, axis=1)
    df_concat.columns = [f"ticker_{i}" for i in range(len(all_curves))]
    df_concat = df_concat.ffill().fillna(1)
    df_concat["global_cumulative_return"] = df_concat.sum(axis=1)
    return df_concat[["global_cumulative_return"]]

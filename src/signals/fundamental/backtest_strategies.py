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


# delete above
def backtest_long_short_neutral(df_clean: pd.DataFrame, quantile=0.25) -> pd.DataFrame:
    df = df_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    results = []

    for date, group in df.groupby("date"):
        group = group.dropna(subset=["alpha", "target"])
        if len(group) < 4:
            continue  

        low_q = group["alpha"].quantile(quantile)
        high_q = group["alpha"].quantile(1 - quantile)

        long = group[group["alpha"] >= high_q]
        short = group[group["alpha"] <= low_q]

        if len(long) == 0 or len(short) == 0:
            continue
        
        long_return = long["target"].mean()
        short_return = short["target"].mean()
        net_return = long_return - short_return
        
        results.append({
            "date": date,
            "long_return": long_return,
            "short_return": short_return,
            "long_count": len(long),
            "short_count": len(short),
            "net_return": net_return,
        })
    
    return pd.DataFrame(results)
def plot_long_short_quantile_strategy(perf_df: pd.DataFrame):
    plt.plot(perf_df["date"], perf_df["net_return"].cumsum(), marker='o')
    plt.title("Cumulative Return - Long/Short Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(perf_df)
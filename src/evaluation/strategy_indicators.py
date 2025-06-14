import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf

def compare_strategies_stats(df_garch, df_arima, df_lr, df_gbt, benchmark_ticker, freq=252):
    start = max(df.index.min() for df in [df_garch, df_arima, df_lr, df_gbt])
    end = min(df.index.max() for df in [df_garch, df_arima, df_lr, df_gbt])
    benchmark = yf.download(benchmark_ticker, start=start, end=end, progress=False)[["Close"]]
    benchmark = benchmark.asfreq('B').ffill().loc[start:end]
    bench_ret = benchmark['Close'].pct_change().fillna(0).squeeze() 

    def stats(r):
        cr = (1 + r).prod() - 1
        ar = (1 + cr) ** (freq / len(r)) - 1
        vol = r.std() * np.sqrt(freq)
        sharpe = ar / vol if np.isscalar(vol) and not np.isclose(vol, 0) else np.nan
        dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return [cr, ar, vol, sharpe, dd]

    def compute_for_df(df):
        r = df['strategy_return'].reindex(benchmark.index).fillna(0)
        return stats(r)

    result = pd.DataFrame({
        benchmark_ticker: stats(bench_ret),
        "GARCH": compute_for_df(df_garch),
        "ARIMA": compute_for_df(df_arima),
        "Linear Regression": compute_for_df(df_lr),
        "Gradient Boosted Tree": compute_for_df(df_gbt)
    }, index=['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'])

    return result

def compute_stats(df, benchmark_ticker, freq=252):
    start, end = df.index.min(), df.index.max()
    benchmark = yf.download(benchmark_ticker, start=start, end=end, progress=False)[["Close"]]
    benchmark = benchmark.asfreq('B').ffill().loc[start:end]

    strat_ret = df['strategy_return'].reindex(benchmark.index).fillna(0)
    bench_ret = benchmark['Close'].pct_change().fillna(0).squeeze() 

    def stats(r):
        cr = (1 + r).prod() - 1
        ar = (1 + cr) ** (freq / len(r)) - 1
        vol = r.std() * np.sqrt(freq)
        sharpe = ar / vol if np.isscalar(vol) and not np.isclose(vol, 0) else np.nan
        dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return [cr, ar, vol, sharpe, dd]

    return pd.DataFrame(
        [stats(strat_ret), stats(bench_ret)],
        columns=['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        index=['Strategy', benchmark_ticker]
    ).T
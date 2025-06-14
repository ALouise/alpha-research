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

def calculate_annual_return(df: pd.DataFrame) -> float:
    cumulative_return = (1 + df["net_return"]).prod()
    n_years = len(df)
    return cumulative_return**(1 / n_years) - 1

def calculate_information_ratio(excess_return: pd.Series) -> float:
    return_ratio = excess_return.mean() / excess_return.std() if excess_return.std() > 0 else np.nan
    return return_ratio

def calculate_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_best_periods(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    return df.nlargest(top_n, "net_return")

def calculate_worst_periods(df: pd.DataFrame, bottom_n: int = 3) -> pd.DataFrame:
    return df.nsmallest(bottom_n, "net_return")

def calculate_alpha_beta(df: pd.DataFrame) -> tuple:
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["benchmark_return"], df["net_return"])
    return intercept, slope, r_value, r_value**2  # alpha, beta, correlation, R²

def evaluate_long_short_results(df_strategy: pd.DataFrame, benchmark_df: pd.DataFrame) -> tuple:
    df_strategy = df_strategy.sort_values("date").reset_index(drop=True)
    benchmark_df = benchmark_df.sort_values("date").reset_index(drop=True)

    benchmark_returns = []

    for i in range(1, len(df_strategy)):
        start_date = df_strategy.loc[i - 1, "date"]
        end_date = df_strategy.loc[i, "date"]
        mask = (benchmark_df["date"] > start_date) & (benchmark_df["date"] <= end_date)
        window = benchmark_df.loc[mask, "benchmark_return"]
        ret = (1 + window).prod() - 1 if not window.empty else np.nan
        benchmark_returns.append(ret)

    df_eval = df_strategy.iloc[1:].copy().reset_index(drop=True)
    df_eval["benchmark_return"] = benchmark_returns
    df_eval["excess_return"] = df_eval["net_return"] - df_eval["benchmark_return"]

    alpha, beta, corr, r2 = calculate_alpha_beta(df_eval)

    metrics = {
        "Annual Return": calculate_annual_return(df_eval),
        "Information Ratio": calculate_information_ratio(df_eval["excess_return"]),
        "Max Drawdown": calculate_max_drawdown(df_eval["net_return"]),
        "Alpha": alpha,
        "Beta": beta,
        "Correlation": corr,
        "R²": r2
    }

    return df_eval, metrics, calculate_best_periods(df_eval), calculate_worst_periods(df_eval)


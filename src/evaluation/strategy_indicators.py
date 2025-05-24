import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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


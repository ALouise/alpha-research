import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import math
from sklearn.linear_model import LinearRegression

def plot_multiple_strategies_performance(df_garch, df_arima, df_lr, df_gbt, benchmark_ticker):
    start = min(df_garch.index.min(), df_arima.index.min(), df_lr.index.min(), df_gbt.index.min())
    end = max(df_garch.index.max(), df_arima.index.max(), df_lr.index.max(), df_gbt.index.max())
    benchmark = yf.download(benchmark_ticker, start=start, end=end, progress=False)[["Close"]]
    benchmark = benchmark.asfreq('B').ffill().loc[start:end]
    benchmark_returns = benchmark['Close'].pct_change().fillna(0)
    benchmark_cum = (1 + benchmark_returns).cumprod()
    benchmark_cum /= benchmark_cum.iloc[0]
    models = [
        ("GARCH", df_garch),
        ("ARIMA", df_arima),
        ("Linear Regression", df_lr),
        ("Gradient Boosted Tree", df_gbt)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axs = axs.flatten()

    for i, (label, df_model) in enumerate(models):
        strat_returns = df_model['strategy_return'].reindex(benchmark.index).fillna(0)
        strat_cum = (1 + strat_returns).cumprod()
        strat_cum /= strat_cum.iloc[0]

        axs[i].plot(strat_cum.index, strat_cum, label=label)
        axs[i].plot(benchmark_cum.index, benchmark_cum, label=f"{benchmark_ticker} (Benchmark)")
        axs[i].set_title(label)
        axs[i].set_ylabel("Cumulative Return")
        axs[i].grid(True)
        axs[i].legend()

    axs[2].set_xlabel("Date")
    axs[3].set_xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_strategy_performance(df, benchmark_ticker, label):
    start = df.index.min()
    end = df.index.max()
    benchmark = yf.download(benchmark_ticker, start=start, end=end, progress=False)
    benchmark = benchmark[["Close"]]
    benchmark.columns = ["Close"]
    benchmark = benchmark.asfreq('B').ffill().loc[start:end]
    benchmark = benchmark.loc[df.index.min():df.index.max()]
    
    benchmark_returns = benchmark['Close'].pct_change().fillna(0)
    strategy_returns = df['strategy_return'].reindex(benchmark.index).fillna(0)

    benchmark_cum = (1 + benchmark_returns).cumprod()
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum /= benchmark_cum.iloc[0]
    strategy_cum /= strategy_cum.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cum.index, strategy_cum, label=label)
    plt.plot(benchmark_cum.index, benchmark_cum / benchmark_cum.iloc[0], label=f"{benchmark_ticker} (Benchmark)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"{label} vs {benchmark_ticker}")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_serie(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label="Close")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Close")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_two_series(df1, df2, label_model, label1='predict', label2='real'):
    if not df1.index.equals(df2.index):
        raise ValueError("DataFrames must have the same index.")
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape.")

    y_pred = df1.values.flatten()
    y_true = df2.values.flatten()
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_pred_valid = y_pred[mask]
    y_true_valid = y_true[mask]
    errors = np.abs(y_pred - y_true)
    rmse = np.sqrt(np.mean((y_pred_valid - y_true_valid) ** 2))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    axs[0].plot(df1.index, y_pred, label=label1)
    axs[0].plot(df2.index, y_true, label=label2)
    axs[0].set_ylabel("Value")
    axs[0].set_title(f"{label_model} — RMSE = {rmse:.6f}")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].bar(df1.index, errors, width=1.0)
    axs[1].set_ylabel("Absolute Error")
    axs[1].set_xlabel("Date")
    axs[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_smoothed_series(df1, df2, label1='predicted', label2='real', window=5):
    if not df1.index.equals(df2.index):
        raise ValueError("DataFrames must have the same index.")

    smooth1 = df1.rolling(window=window).mean()
    smooth2 = df2.rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(smooth1, label=f"{label1} (MA-{window})")
    plt.plot(smooth2, label=f"{label2} (MA-{window})")
    plt.xlabel("Date")
    plt.ylabel("Smoothed Value")
    plt.title(f"{label1} vs {label2} (Smoothed over {window} days)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_grid_predictions_by_ticker(df_result, label_model):
    tickers = df_result["ticker"].unique()
    n = len(tickers)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows * 2, ncols, figsize=(5 * ncols, 5 * nrows), sharex=False)
    fig.suptitle(label_model, fontsize=16, fontweight='bold', y=1.02)
    axs = axs.reshape(nrows * 2, ncols)  
    for i, ticker in enumerate(tickers):
        row = (i // ncols) * 2
        col = i % ncols

        df_ticker = df_result[df_result["ticker"] == ticker].sort_index()
        df1 = df_ticker["predict_return_qt"]
        df2 = df_ticker["return_qt"]

        y_pred = df1.values.flatten()
        y_true = df2.values.flatten()
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]
        errors = np.abs(y_pred - y_true)
        rmse = np.sqrt(np.mean((y_pred_valid - y_true_valid) ** 2))

        axs[row, col].plot(df1.index, df1, label="predict")
        axs[row, col].plot(df2.index, df2, label="real")
        axs[row, col].set_title(f"{ticker} — RMSE={rmse:.4f}")
        axs[row, col].legend()
        axs[row, col].grid(True)

        axs[row + 1, col].bar(df1.index, errors, width=5)
        axs[row + 1, col].set_ylabel("Abs. Error")
        axs[row + 1, col].set_xlabel("Date")
        axs[row + 1, col].grid(True)

    for j in range(i + 1, nrows * ncols):
        row = (j // ncols) * 2
        col = j % ncols
        axs[row, col].axis("off")
        axs[row + 1, col].axis("off")
    plt.tight_layout()
    plt.show()

def plot_sum_cumulative_returns(df):
    df.sum(axis=1).plot(figsize=(12, 6), title="Sum of Cumulative Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_strategy_vs_benchmark(global_return_df, benchmark_df):
    common_start = max(global_return_df.index.min(), benchmark_df.index.min())
    common_end = min(global_return_df.index.max(), benchmark_df.index.max())

    strategy = global_return_df.loc[common_start:common_end].copy()
    benchmark = benchmark_df.loc[common_start:common_end].sum(axis=1).copy()

    strategy /= strategy.iloc[0]
    benchmark /= benchmark.iloc[0]

    plt.figure(figsize=(12, 6))
    strategy.plot(ax=plt.gca(), label="Strategy Global Cumulative Return")
    benchmark.plot(ax=plt.gca(), label="Benchmark Cumulative Return", linestyle="--")
    plt.title("Strategy vs Benchmark Cumulative Returns (Rebased)")
    plt.ylabel("Rebased Cumulative Return")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(df):
    df.plot(figsize=(12, 6), title="Cumulative Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_backtest_grid(all_backtest):
    n = len(all_backtest)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = axes.flatten()
    for i, elem in enumerate(all_backtest):
        df, ticker_str = elem
        ax = axes[i]
        ax.plot(df.index, df["cumulative_return"], label="Strategy")
        ax.plot(df.index, df["cumulative_buy_hold"], label="Buy & Hold")
        ax.set_title(ticker_str)
        ax.grid(True)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Long short strategy on each tickers", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_prediction_scatter(df_result):
    df_result = df_result.dropna(subset=["predict_return_qt", "return_qt"])
    x = df_result["predict_return_qt"].values.reshape(-1, 1)
    y = df_result["return_qt"].values.reshape(-1, 1)

    reg = LinearRegression().fit(x, y)
    y_line = reg.predict(x)
    r2 = reg.score(x, y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, label="Points")
    plt.plot(x, y_line, color="red", label=f"Fit line (R² = {r2:.2f})")
    plt.xlabel("Predicted Return")
    plt.ylabel("Actual Return")
    plt.title("Regression Fit: Predicted vs Actual Quarterly Returns")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
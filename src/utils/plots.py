import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

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
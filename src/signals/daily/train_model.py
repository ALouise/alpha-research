import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def train_linear_model(df, target_col="return_t+1"):
    df = df.dropna()
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = Ridge()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_lgbm_model(df, target_col="return_t+1"):
    df = df.dropna()
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LGBMRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    return model,  X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    hit_rate = (np.sign(y_test) == np.sign(y_pred)).mean()

    print("MSE:", round(mse, 6))
    print("R²:", round(r2, 4))
    print("Hit rate:", round(hit_rate, 4))

    plt.figure(figsize=(12, 4))
    plt.plot(y_test.index, y_test.values, label="True")
    plt.plot(y_test.index, y_pred, label="Pred", alpha=0.7)
    plt.legend()
    plt.title(f"Predicted vs True Returns for {model}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 2.5))
    plt.bar(y_test.index, y_pred - y_test.values)
    plt.title("Prediction Error")
    plt.tight_layout()
    plt.show()

def backtest_strategy(model, X_test, y_test, signal_bull = 1, signal_bear= -1):
    y_pred = model.predict(X_test)
    signal = np.where(y_pred > 0, signal_bull, signal_bear)
    pnl = signal * y_test.values
    pnl_series = pd.Series(pnl, index=y_test.index)
    return pnl_series

def compare_strategy_vs_stock(pnl_series, y_test):
    strat_cum = pnl_series.cumsum()
    stock_cum = y_test.cumsum()

    df = pd.DataFrame({
        "Strategy": strat_cum,
        "Stock": stock_cum
    })

    df.plot(figsize=(12, 4), title="Cumulative Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    strat_total = strat_cum.iloc[-1]
    stock_total = stock_cum.iloc[-1]

    print("Strategy Return:", round(strat_total * 100, 2), "%")
    print("Stock Return:", round(stock_total * 100, 2), "%")


def create_features_dataframe(ticker: str, start_date:str, end_date:str):
    df = yf.download(ticker, start_date, end_date, progress=False)
    df = df[["Close", "Volume"]]
    df.columns = ["Close", "Volume"]
    df["return_t+1"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["return_t"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df.dropna()
    return df

def strategy_return(tickers:list, start_date: str, end_date:str, signal_bull=1, signal_bear=-1):
    strategy_returns = {}
    benchmark_returns = {}

    for ticker in tickers:
        try:
            df_feat = create_features_dataframe(ticker, start_date, end_date)
            model, X_test, y_test = train_linear_model(df_feat)
            pnl = backtest_strategy(model, X_test, y_test, signal_bull, signal_bear)
            strategy_returns[ticker] = pnl
            benchmark_returns[ticker] = y_test
        except Exception as e:
            print(f"Error on {ticker}: {e}")
    return strategy_returns, benchmark_returns, model


def strategy_return_lgbm(tickers:list, start_date: str, end_date:str, signal_bull=1, signal_bear=-1):
    strategy_returns = {}
    benchmark_returns = {}

    for ticker in tickers:
        try:
            df_feat = create_features_dataframe(ticker, start_date, end_date)
            model, X_test, y_test = train_lgbm_model(df_feat)
            pnl = backtest_strategy(model, X_test, y_test, signal_bull, signal_bear)
            strategy_returns[ticker] = pnl
            benchmark_returns[ticker] = y_test
        except Exception as e:
            print(f"Error on {ticker}: {e}")
    return strategy_returns, benchmark_returns, model

def compare_portfolio_vs_benchmark(strategy_returns, benchmark_returns, model):
    df_strat = pd.DataFrame(strategy_returns).dropna(how="any", axis=0)
    df_bench = pd.DataFrame(benchmark_returns).dropna(how="any", axis=0)

    strat_cum = df_strat.mean(axis=1).cumsum()
    bench_cum = df_bench.mean(axis=1).cumsum()

    df = pd.DataFrame({
        "Strategy": strat_cum,
        "Benchmark": bench_cum
    })

    df.plot(figsize=(12, 4), title=f"Portfolio Strategy vs Equal-Weighted Benchmark for {model} model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Strategy Return:", round(strat_cum.iloc[-1] * 100, 2), "%")
    print("Benchmark Return:", round(bench_cum.iloc[-1] * 100, 2), "%")
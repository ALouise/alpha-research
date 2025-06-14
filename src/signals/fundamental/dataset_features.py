import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import pearsonr
import yfinance as yf
from sklearn.preprocessing import StandardScaler


def create_benchmark_cumulative_returns(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end)["Close"]
    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0)
    return (1 + returns).cumprod()

def build_multiticker_fundamental_dataset(tickers: list) -> pd.DataFrame:
    all_dfs = []

    for ticker in tickers:
        try:
            df = build_quarterly_fundamental_dataset(ticker)
            df["ticker"] = ticker
            all_dfs.append(df)
        except Exception as e:
            print(f"Error for {ticker} : {e}")
            continue
    df_panel = pd.concat(all_dfs)
    df_panel["period_quarter"] = df_panel.index.to_period("Q").astype(str)
    cols = ["period_quarter", "ticker"] + [col for col in df_panel.columns if col not in ["ticker", "period_quarter"]]
    df_panel = df_panel[cols]
    df_panel = df_panel.sort_values(by=["period_quarter", "ticker"]).reset_index()
    return df_panel

def get_aligned_quarterly_return(str_ticker: str) -> pd.Series:
    ticker = yf.Ticker(str_ticker)
    reference_dates = list(ticker.quarterly_financials.columns.sort_values())
    start_date = (min(reference_dates) - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    end_date = (max(reference_dates) + pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    prices = yf.download(str_ticker, start=start_date, end=end_date, interval='1d')[['Close']]
    prices.columns = ['Close']
    closes = prices.reindex(prices.index.union(reference_dates)).ffill().loc[reference_dates]  
    returns = closes.pct_change().dropna()
    returns.columns =["Quarterly Return"]
    return returns

def build_quarterly_fundamental_dataset(ticker_str: str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_str)

    income_cols = [
        "Net Income", "Gross Profit",  "Total Revenue"
    ]

    balance_cols = [
        "Total Debt", "Total Capitalization"
    ]

    cashflow_cols = [
        "Free Cash Flow"
    ]

    df_income = ticker.quarterly_financials.T[income_cols] if not ticker.quarterly_financials.empty else pd.DataFrame()
    df_balance = ticker.quarterly_balance_sheet.T[balance_cols] if not ticker.quarterly_balance_sheet.empty else pd.DataFrame()
    df_cashflow = ticker.quarterly_cashflow.T[cashflow_cols] if not ticker.quarterly_cashflow.empty else pd.DataFrame()

    df = df_income.join([df_balance, df_cashflow], how='outer')
    df = df.dropna(how='all')
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)   
    y = get_aligned_quarterly_return(ticker_str)
    dataset = y.join(X, how="inner")  
    return dataset



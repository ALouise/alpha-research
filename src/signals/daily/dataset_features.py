import yfinance as yf
import numpy as np


# dataframe 3 columns : return_t, return_t+1, sma_5
def create_features_dataframe(ticker: str, start_date:str, end_date:str): 
    df = yf.download(ticker, start_date, end_date, progress=False)
    df = df[["Close"]]
    df.columns = ["Close"]
    df["return_t+1"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["return_t"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df.dropna()
    return df



import yfinance as yf
import numpy as np

def _get_close_vol(ticker: str, start_date:str, end_date:str):
    df = yf.download(ticker, start_date, end_date, progress=False)
    df_close_vol = df[["Close", "Volume"]]
    df_close_vol.columns = ["Close", "Volume"]
    return df_close_vol

def compute_log_returns(df):
    df["return_t+1"] = np.log(df["Close"].shift(-1) / df["Close"])
    df["return_t"] = np.log(df["Close"] / df["Close"].shift(1))
    #df["return_5d"] = np.log(df["Close"] / df["Close"].shift(5))

def compute_sma(df):
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    #df["sma_10"] = df["Close"].rolling(window=10).mean()

def compute_ema(df):
    df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()

def compute_vol_5d(df):
    df["vol_5d"] = df["return_t"].rolling(window=5).std()

def compute_momentum_5(df):
    df["momentum_5"] = df["return_t"].rolling(5).sum()

def compute_volume_trend(df):
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_log"] = np.log(df["Volume"])
    mean = df["Volume"].rolling(20).mean()
    std = df["Volume"].rolling(20).std()
    df["volume_z"] = (df["Volume"] - mean) / std

#Relative Strength Index over 14 days. Values near 70 = overbought, near 30 = oversold.
def compute_rsi_14(df):
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(14).mean()
    avg_loss = down.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

#Price tends to stay within upper and lower bands. Useful for mean-reversion signals.
def compute_bollinger_bands(df):
    ma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["boll_upper"] = ma + 2 * std
    df["boll_lower"] = ma - 2 * std

#MACD = 12-day EMA minus 26-day EMA. Positive = bullish momentum, negative = bearish.
def compute_macd(df):
    ema_12 = df["Close"].ewm(span=12).mean()
    ema_26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26

def create_features_dataframe(ticker: str, start_date:str, end_date:str):
    df =_get_close_vol(ticker, start_date, end_date) 
    compute_log_returns(df)
    compute_sma(df)
    #compute_ema(df)
    #compute_vol_5d(df)
    #compute_momentum_5(df)
    #compute_volume_trend(df)
    #compute_rsi_14(df)
    #compute_bollinger_bands(df)
    #compute_macd(df)
    df.dropna()
    return df



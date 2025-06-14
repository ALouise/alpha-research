import numpy as np

def backtest_long_only(df):
    df = df.copy()
    if 'predict_return_t' not in df.columns or 'return_t' not in df.columns:
        raise ValueError("DataFrame must contain 'predict_return_t' and 'return_t' columns.")

    df['position'] = (df['predict_return_t'] > 0).astype(int)  # 1 si > 0, sinon 0
    df['position'] = df['position'].shift(1)  # Décalage pour éviter le look-ahead
    df['strategy_return'] = df['position'] * df['return_t']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def backtest_long_short(df):
    df = df.copy()
    if 'predict_return_t' not in df.columns or 'return_t' not in df.columns:
        raise ValueError("DataFrame must contain 'predict_return_t' and 'return_t' columns.")
    df['position'] = np.sign(df['predict_return_t']).shift(1)  # +1 (long), -1 (short), 0 (flat)
    df['strategy_return'] = df['position'] * df['return_t']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def backtest_threshold_strategy(df, threshold=0.001):
    df = df.copy()
    if 'predict_return_t' not in df.columns or 'return_t' not in df.columns:
        raise ValueError("DataFrame must contain 'predict_return_t' and 'return_t' columns.")

    df['position'] = 0
    df.loc[df['predict_return_t'] > threshold, 'position'] = 1
    df.loc[df['predict_return_t'] < -threshold, 'position'] = -1

    df['position'] = df['position'].shift(1)

    df['strategy_return'] = df['position'] * df['return_t']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    return df
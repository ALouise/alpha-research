import matplotlib.pyplot as plt
import numpy as np

def plot_serie(df):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'].values)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Close")
    plt.legend()
    plt.show()
    
def plot_two_series(df1, df2, label_model, label1='predict', label2='real'):
    if not df1.index.equals(df2.index):
        raise ValueError("DataFrames must have the same index.")
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape.")
    y_pred = df1.values.flatten()
    y_true = df2.values.flatten()

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, y_pred, label=label1)
    plt.plot(df2.index, y_true, label=label2)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"{label_model} — RMSE = {rmse:.6f}")
    plt.legend()
    plt.grid(True)
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
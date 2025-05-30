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



def create_fundamental_dataset(df_z: pd.DataFrame, df_returns: pd.DataFrame) -> pd.DataFrame:
    df_z = df_z.copy()
    df_returns = df_returns.copy()

    df_z["date"] = pd.to_datetime(df_z["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    df_benchmark = df_returns.groupby("date")["return"].mean().reset_index()
    df_benchmark = df_benchmark.rename(columns={"return": "benchmark_return"})

    df_ret_merged = pd.merge(df_returns, df_benchmark, on="date", how="left")
    df_ret_merged["target"] = df_ret_merged["return"] - df_ret_merged["benchmark_return"]

    dataset = pd.merge(df_z, df_ret_merged[["date", "ticker", "target"]], on=["date", "ticker"], how="left")

    dataset = dataset.dropna(subset=["target"]).reset_index(drop=True)

    return dataset

def train_ridge_on_fundamental_alpha(dataset: pd.DataFrame) -> Tuple[Dict[str, float], RidgeCV, StandardScaler, pd.DataFrame]:
    features = ["FCF_Yield", "ROIC", "Gearing", "Revenue_Growth_YOY"]
    df_clean = dataset.dropna(subset=features + ["target"]).copy()
    X = df_clean[features]
    y = df_clean["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    model.fit(X_scaled, y)
    df_clean["alpha"] = model.predict(X_scaled)
    weights = {feat: coef for feat, coef in zip(features, model.coef_)}
    return weights, model, scaler, df_clean

def plot_analyse_alpha_target(df_clean):
    slope, intercept, r_value, p_value, std_err = linregress(df_clean["alpha"], df_clean["target"])
    r2 = r_value**2
    corr, pval_corr = pearsonr(df_clean["target"], df_clean["alpha"])

    print("Linear regression:")
    print(f"  slope               = {slope:.4f}")
    print(f"  intercept           = {intercept:.4f}")
    print(f"  R² (from linregress)= {r2:.4f}")
    print(f"  Corr (Pearson)      = {corr:.4f}")
    print(f"  p-value regression  = {p_value:.4g}")
    print(f"  p-value Pearson     = {pval_corr:.4g}")
    print(f"  std error of slope  = {std_err:.4g}")

    x_vals = df_clean["alpha"]
    y_vals = slope * x_vals + intercept

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x="alpha", y="target", label="Points")
    plt.plot(x_vals, y_vals, color="red", linestyle="--", label="Linear regression")
    plt.title(f"Alpha vs Target\nR² = {r2:.3f}, Corr = {corr:.3f} (p={pval_corr:.3g})")
    plt.xlabel("Alpha (predicted)")
    plt.ylabel("Target (actual)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "correlation": corr,
        "p_value_linreg": p_value,
        "p_value_corr": pval_corr,
        "std_error": std_err
    }



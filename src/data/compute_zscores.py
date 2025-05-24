import pandas as pd
from scipy.stats import zscore

def compute_fundamental_zscores(list_of_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    indicators = [
        "FCF_Yield",
        "ROIC",
        "Gearing",
        "Revenue_Growth_YOY"
    ]

    zscore_df = (
        combined_df
        .sort_values("date")
        .groupby("date")
        [indicators]
        .transform(lambda x: zscore(x, nan_policy="omit"))
    )

    final_df = pd.concat([combined_df[["date", "ticker"]], zscore_df], axis=1).dropna(subset=indicators, how="all")
    return final_df

def get_zscores_for_ticker(ticker: str, list_of_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df_z = compute_fundamental_zscores(list_of_dfs)
    return df_z[df_z["ticker"] == ticker].reset_index(drop=True)
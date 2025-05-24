

def threshold_strategy(alpha_df, threshold=0.5):
    alpha_df["position"] = 0
    alpha_df.loc[alpha_df["alpha"] > threshold, "position"] = 1
    alpha_df.loc[alpha_df["alpha"] < -threshold, "position"] = -1
    return alpha_df
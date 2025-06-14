from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd

def predict_linear_quarter_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "Quarterly Return" not in df.columns:
        raise ValueError("Il manque la colonne 'Quarterly Return'.")
    ignore_cols = ["index", "ticker", "period_quarter", "Quarterly Return"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    X = df[feature_cols]
    y = df["Quarterly Return"]
    dates = df["index"]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), index=dates, columns=feature_cols)
    model = LinearRegression()
    model.fit(X_imputed, y)
    predictions = model.predict(X_imputed)
    df_result = pd.DataFrame({
        "ticker": df["ticker"].values,
        "period_quarter": df["period_quarter"].values,
        "predict_return_qt": predictions,
        "return_qt": y.values
    }, index=dates)
    beta = pd.Series(model.coef_, index=X_imputed.columns).sort_values(ascending=False)
    return df_result, beta
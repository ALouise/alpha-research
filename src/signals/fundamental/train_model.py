from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd

def predict_linear_quarter_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_sorted = df.sort_values("index")
    ignore_cols = ["index", "ticker", "period_quarter", "Quarterly Return"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    results = []
    last_model = None

    for i in range(1, len(df_sorted)):
        train = df_sorted.iloc[:i]
        test = df_sorted.iloc[i:i+1]

        X_train = train[feature_cols]
        y_train = train["Quarterly Return"]
        X_test = test[feature_cols]
        y_test = test["Quarterly Return"]

        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

        model = LinearRegression()
        model.fit(X_train_imp, y_train)
        pred = model.predict(X_test_imp)[0]

        results.append({
            "index": test["index"].values[0],
            "ticker": test["ticker"].values[0],
            "period_quarter": test["period_quarter"].values[0],
            "predict_return_qt": pred,
            "return_qt": y_test.values[0]
        })

        last_model = model

    df_result = pd.DataFrame(results).set_index("index")
    beta = pd.Series(last_model.coef_, index=feature_cols).sort_values(ascending=False) if last_model else None
    return df_result, beta

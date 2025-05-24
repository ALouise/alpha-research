from src.data.load_data import load_all_fundamentals
from src.data.compute_zscores import compute_fundamental_zscores
from signals.fundamental.dataset_regression_alpha_result import prepare_ml_dataset, train_ridge_model, compute_alpha_scores

if __name__ == "__main__":
    dfs = load_all_fundamentals()
    df_z = compute_fundamental_zscores(dfs)

    X, y = prepare_ml_dataset(df_z)
    model = train_ridge_model(X, y)

    df_alpha = compute_alpha_scores(df_z, model)
    print(df_alpha.head())

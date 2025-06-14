# Alpha Research — Aerospace Sector (Equity)

This project focuses on identifying alpha in the aerospace sector within the equity asset class across US and European markets.

## Fundamental Alpha (Quarterly)

- Benchmark: Basket of aerospace stocks  
- Frequency: Quarterly (2023Q4 to 2025Q1)  
- Features: Net Income, Gross Profit, Total Revenue, Total Debt, Total Capitalization, Free Cash Flow  
- Models: Linear Regression (Ordinary Least Squares), Ridge Regression, Lasso Regression,  
- Strategy: Long/Short  
- Evaluation Criteria: CAGR: ?, Sharpe Ratio: ?, Max Drawdown: ?, Alpha: ?, Information Ratio: ?

## Daily Alpha (Short-Term)

- Benchmark: Individual stocks and aerospace stock basket  
- Frequency: Daily (over a few months)  
- Features: Close price, Return, 5-day Moving Average  
- Models: Ordinary Least Squares, Gradient Boosted Trees, ARIMA, GARCH  
- Strategies: Long only, Long/Short, Long/Short with threshold  
- Evaluation Criteria:   CAGR: ?, Sharpe Ratio: ?, Max Drawdown: ?, Alpha: ?, Information Ratio: ?

## Repository Structure

- `notebooks/`  
  - `evaluations/`: Statistical evaluation of predicted vs actual returns (e.g., RMSE, visual comparisons across models)  
  - `results/`: Figures and plots of strategy performance vs benchmark  

- `src/`  
  - Core modules and all implemented functions

## TODO

- Add evaluation metrics for each strategy: P&L, Alpha, IR, Drawdown, Sharpe Ratio  
- Backtest the fundamental strategy using Ridge and Lasso  
- Try to compute daily alpha based on fundamental signals

# Pairs Trading with Machine Learning and IBKR Integration

This repository contains a Jupyter notebook (`main.ipynb`) implementing a pairs trading strategy. It demonstrates how to:

* **Identify a pair of correlated stocks**
* **Calculate the spread** between their price series
* **Engineer features** from the spread (e.g., rolling statistics)
* **Train a machine learning model** to predict the next-period spread
* **Generate trading signals** based on predicted spread deviations
* **Execute buy/sell orders** for both stocks via the Interactive Brokers (IBKR) API

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Notebook Structure](#notebook-structure)
7. [Data Source](#data-source)
8. [Methodology](#methodology)
9. [Machine Learning Model](#machine-learning-model)
10. [IBKR API Integration](#ibkr-api-integration)
11. [Results and Evaluation](#results-and-evaluation)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

---

## Project Overview

Pairs trading is a market-neutral strategy that exploits mean-reversion in the spread between two historically correlated assets. This project shows how to:

1. Select a candidate stock pair using correlation analysis.
2. Compute the price spread (difference or ratio) and its statistical properties.
3. Build and evaluate a machine learning model (e.g., Ordinary Least Squares, Time-Series Linear Regression) to predict next-day spread.
4. Generate entry and exit signals when the predicted spread deviates from its historical mean beyond a threshold.
5. Automate trade execution by placing corresponding long/short orders on both legs via the IBKR API.

## Features

* **Pair Selection**: Correlation matrix and distance metrics for candidate pairs.
* **Spread Calculation**: Rolling mean, rolling standard deviation, and z-score computation.
* **Feature Engineering**: Lagged spread values, rolling window statistics, and technical indicators.
* **Modeling**: Train/test split, Ordinary Least-Squares, Time-Series Linear Regression for spread prediction.
* **Signal Generation**: Mean-reversion thresholds and take-profit levels.
* **IBKR Trading**: Live order placement with ib\_insync, order management, and fill tracking.

## Requirements

* Python 3.8+
* Jupyter Notebook

Python Libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* statsmodels
* ib\_insync

Optional for enhancements:

* pmdarima (for ARIMA comparisons)
* rapidfuzz (for ticker validation in dashboard)

## Installation

```bash
git clone https://github.com/marabsatt/blackwell.git
cd blackwell
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Usage

1. Open `main.ipynb` in Jupyter.
2. **Pair Selection**: Run cells to compute correlations and choose a stock pair.
3. **Spread Engineering**: Compute and visualize spread statistics.
4. **Model Training**: Fit the ML model to predict next-day spread.
5. **Signal Logic**: Define entry/exit thresholds based on predicted spread z-scores.
6. **Trade Execution**: Ensure IBKR Gateway/TWS is running and execute the trading cells to place orders.

Modify parameters (e.g., lookback windows, thresholds, model hyperparameters) directly in the notebook.

## Notebook Structure

1. **Imports & Configuration**: Load libraries and API settings
2. **Data Ingestion**: Fetch historical price data for candidate tickers
3. **Pair Selection**: Correlation analysis and scatter plots
4. **Spread Calculation**: Compute spread, rolling mean/std, and z-score
5. **Feature Engineering**: Create lagged and rolling features from the spread
6. **Modeling**:

   * Ordinary Least Squares
   * Train/test split by date
   * Time-Series Linear Regression
   * Model fitting and validation
7. **Prediction & Signals**: Generate predicted spread and trading signals
8. **IBKR API Integration**: Connect to IBKR, define contracts, place orders
9. **Results & Evaluation**: Backtest P\&L, plot performance, and diagnostics

## Data Source

* Historical price data via **yfinance** or **IBKR API**.
* Ensure trading calendar consistency by aligning dates and handling missing market days.

## Methodology

1. **Stationarity Check**: Ensure spread series is mean-reverting (augmented Dickey-Fuller test).
2. **Rolling Statistics**: Compute moving averages and standard deviations to detect deviations.
3. **Z-score Thresholds**: Entry when |z-score| > 2.0; exit when z-score returns to zero or hits stop-loss.

## Machine Learning Model

* **Algorithms**: Linear Regression, Ordinary Least Squares.
* **Features**:

  * Lagged spread values (t-1, t-2, ...)
  * Rolling mean and std (window sizes 5, 10, 20)

* **Metrics**: R², RMSE, MAE on hold-out set.

## IBKR API Integration

1. **Connection**: `ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1)`
2. **Contract Definition**: `Stock(ticker, 'SMART', 'USD')` and `ib.qualifyContracts()`
3. **Order Placement**: Market and limit orders via `ib.placeOrder(contract, order)`
4. **Event Handling**: Subscribe to `trade.filledEvent` for execution feedback

> **Warning**: Test in paper trading before deploying to a live account.

## Results and Evaluation

* **Backtest Performance**: P\&L curve
* **Signal Efficacy**: Hit rate of profitable trades, average return per trade.
* **Residual Analysis**: Actual v. Predicted, Residuals v. Predicted, Residuals Distribution, and Q-Q plot of Residuals to verify model adequacy.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Acknowledgements

* [ib\_insync](https://github.com/erdewit/ib_insync) for easy IBKR API access.
* [Statsmodels documentation](https://www.statsmodels.org/) for time-series tools.
* [Scikit-learn](https://scikit-learn.org/) for machine learning utilities.
* Inspired by various pairs trading strategy tutorials and academic research.

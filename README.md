# Machine Learning for Cross-Sectional Stock Return Prediction

This project investigates whether machine learning models can predict cross-sectional differences in future stock returns using market-derived features. A leakage controlled pipeline was developed using historical price and volume data, with Ridge Regression and Gradient Boosting models evaluated using chronological validation. Model performance was assessed using both prediction accuracy and stock-ranking metrics.

## Research Question
Can machine learning models identify stocks that will outperform their peers over the following 30 trading days using histoical price and volume information?

The application of continuation of such a project would be ranking stocks for a long/short strategy. This project was performed as a way to demonstrate knowledge of typical data science/ML practices.

## Dataset
Historical daily price and volume data were obtained using yfinance for a fixed universe of 10 S&P 500 constituents between 2015 and 2025, covering the COVID-19 pandemic period.

- ADBE
- AMD
- AXP
- XOM
- FDX
- HSY
- HPE
- JPM
- MRNA
- NVDA
The selection involves current constituents, and as such introduces survivorship bias and therefore limits the interpretation of the results as representative of the historical S&P 500 universe.
Some constituents, for example, Moderna (MRNA), has data only from 2018 IPO onwards. No artifical values were introduced to fill the missing observations.

## Features
Features introduced to the models were:
- 1 month momentum
- 3 momth momentum
- 6 month momentum
- 12 month momentum
- volatility
- volume ratio

The target was the subsequent 30-trading-day return. Final observations without a complete 30-day future window, or 12 month momentum, were removed from the training set.

## Processing

Features were calculated before concatenating stocks to prevent rolling/shifting operations to cross ticker boundaries. The data range was then split chronologically into training, validation and test.

For Ridge, standardisation was performed using StandardScalar only on training data.

## Models

### Ridge Regression
Ridge regression is a simple linear model, which used L2 regularisation to reduce coefficient size to prevent overfitting to noise in a training set. 

### Gradient Boosting
Gradient boosting is a non-linear ensemble model which sequentially builds trees. This model was testing with different learning rates, number of trees, and tree depths.

## Model Evaluation

Three metrics were used to evaluate performance:
- Root mean square error: how close are predicted returns to actual returns?
- Information coefficient (IC): does the model correctly rank stock performance compared to others?
- Top stock hit rate: can the model predict which stock performs best better than just guessing?

With random selection, we expect a hit rate of 10%.

## Results
|Model|Validation RMSE|Mean IC|Median IC|Positive IC Days|Top-stock hit rate|
|--|---:|---:|---:|---:|---:|
|Mean-return baseline|-|-|-|-|10%|
|Ridge|-|-|-|-|-|
|Gradient Boosting|-|-|-|-|-|
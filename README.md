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
|Mean-return baseline|0.1512|-|-|-|10%|
|Ridge|0.1545|0.001671|-0.02975|0.4861|0.29880|
|Gradient Boosting|0.1508|-0.0658|-0.0928|0.4541|0.1235|

### Mean-return baseline:
Achieved a validation RMSE of 0.1512, providing a useful benchmark for evaluating whether the models improve on simply predicting the average return. Its 10% top-stock hit rate represents the random-selection benchmark of a 10-stock universe.

### Ridge:
Performed worse than the baseline on RMSE, but produces a near zero mean IC and a 29.9% top-stock hit rate, suggesting some ability to rank the highest-performing stock despite limited overall predictive accuracy.

### Gradient Boosting:
Achieved the lowest RMSE, marginally outperforming the baseline. However, its mean IC was negative and its top-stock hit rate was only 12.4%, indicating that the improved RMSE did not translate reliably into cross-sectional ranking ability.

## Critical Analysis

### Limited Universe
The use of only 10 stocks results in noisy cross-sectional estimates. Expanding the universe would provide more observation per date and potentially more reliable estimates of IC.

### Survivorship Bias
Using current S&P 500 constituents introduces survivorship bias becasue companies that left the index historically are not represented.

### Overlapping Targets
The 30-trading-day forward-return target produces overlapping observations, since consecutive trading days have a 29 day overlap in forward-return.

### Transaction Costs
The analysis does not incorporate transaction costs, which mitigates the predictive performance and as such cannot be interpreted directly as a trading strategy's profitability.

### Feature Limitations
The feature set is intentionally simple and relies primarily on momentum, volatility and volume information. Additional features may provide substantially more prediction power.

### Model Limitations
The dataset used is relatively small so highly complex models are not suitable. Increasing model complexity without increasing the amount/quality of the data could increase risk of overfitting.

## Conclusion

The project finds limited evidence that the selected market-derived features provide a robust out-of-sample signal for predicting cross-sectional stock returns. Whilst Gradient Boosting acheived a modest improvement over the baseline in RMSE, this did not translate into a positive IC. The best model achieved a 12.4% top-stock hit rate compared with a 10% random-selection benchmark, suggesting a small degree of predictive behaviour but insufficient evidence of a robust trading signal.

The results highlight the importance of evaluating quantitative models using metrics aligned with their intended application, rather than relying on prediciton error. Future work would focus on expanding the stock universe, addressing survivorship bias, incorporating additional features and evaluating the predictions through a transaction cost aware portfolio backtest.

## Tech Stack
- Python 3.11
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- yfinance
- Jupyter Notebook
- Git


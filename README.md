# Risk Engine

A Python-based portfolio risk management engine using free tools, deployed as a Streamlit dashboard.

## Setup Instructions

1. Ensure you have Python 3.11+ installed.
2. Clone this repository and navigate to the project directory:
   ```bash
   cd risk-engine
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. The application will automatically open in your default web browser (usually at http://localhost:8501).

## Value at Risk (VaR) Methods & Assumptions

**Historical Simulation VaR**  
Historical Simulation assumes that the past is a reliable indicator of the future. It calculates VaR by taking the empirical percentile of actual historical portfolio returns. This method makes no assumptions about the underlying distribution of returns (it is non-parametric), allowing it to naturally capture fat tails, skewness, and complex correlations. However, its major limitation is that it cannot predict losses worse than what has already been observed in the historical window, and it responds slowly to sudden changes in market volatility if the historical window is long.

**Parametric (Variance-Covariance) VaR**  
Parametric VaR assumes that asset returns follow a multivariate Normal distribution. It calculates risk using the portfolio's mean return, the standard deviation of returns, and the correlation between assets. This method is computationally very fast and easy to implement. However, financial returns frequently exhibit "fat tails" (extreme events occur more often than a normal distribution predicts) and skewness. Because it assumes normality, Parametric VaR tends to significantly underestimate the severity and frequency of extreme losses.

**Monte Carlo VaR**  
Monte Carlo Simulation VaR generates thousands of potential future return paths based on the historical mean and covariance matrix of the assets, using a Cholesky decomposition to maintain correlations. The VaR is then extracted as the percentile of the resulting simulated portfolio distribution. While highly flexible and capable of incorporating complex pricing models (e.g., for options), this specific implementation relies on generating Normally distributed random variables based on the historical covariance. Therefore, like Parametric VaR, it assumes normality and may fail to capture fat tails unless more advanced, non-normal random generation techniques are applied. It is also the most computationally intensive of the three methods.

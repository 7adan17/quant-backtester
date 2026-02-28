# Backtester & Options Pricer

**Backtester** (`backtester/`): a vectorized daily backtester built on pandas and NumPy.
- Positions lag one bar so there's no look-ahead; costs are charged on turnover in bps
- Metrics: CAGR, annualized volatility, Sharpe, Sortino, max drawdown, hit rate, turnover
- Strategies: buy and hold, SMA crossover, z-score mean reversion, and a volatility-targeting wrapper
- Works on synthetic GBM paths or a CSV of daily closes (`engine.load_csv`)

**Pricer** (`pricing/black_scholes.py`):
- Closed-form Black-Scholes price and Greeks, plus implied volatility (Newton with a bisection fallback)
- Monte Carlo pricer with antithetic variates, a control variate (the discounted terminal stock price), standard errors and 95% confidence intervals, and pathwise delta and vega

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
.venv/bin/python run_demo.py
```

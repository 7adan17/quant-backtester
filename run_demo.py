"""Runs the strategies on synthetic data and compares MC pricing to closed form."""
import time

from backtester import engine, strategies
from pricing.black_scholes import bs_greeks, bs_price, implied_vol, mc_price


def backtest_demo():
    prices = engine.synthetic_prices(seed=7)
    candidates = [
        strategies.buy_and_hold,
        strategies.sma_crossover(20, 100),
        strategies.zscore_mean_reversion(20),
        strategies.vol_target(strategies.sma_crossover(20, 100)),
    ]
    print(f"{'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Turnover':>10}")
    for strat in candidates:
        m = engine.run(prices, strat, cost_bps=2.0).metrics
        print(f"{strat.__name__:<22}{m['cagr']:>8.2%}{m['sharpe']:>8.2f}"
              f"{m['max_drawdown']:>8.2%}{m['turnover']:>10.1f}")


def pricing_demo():
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    exact = bs_price(S, K, T, r, sigma)
    greeks = bs_greeks(S, K, T, r, sigma)
    print(f"\nBlack-Scholes call: {exact:.4f}  delta={greeks['delta']:.4f} vega={greeks['vega']:.4f}")
    for av, cv in [(False, False), (True, False), (True, True)]:
        t0 = time.perf_counter()
        res = mc_price(S, K, T, r, sigma, n_paths=400_000, seed=1, antithetic=av, control_variate=cv)
        lo, hi = res.ci95()
        print(f"MC antithetic={av!s:<5} control={cv!s:<5} {res.price:.4f} ± {res.std_error:.5f}"
              f"  [{lo:.4f}, {hi:.4f}]  delta={res.delta:.4f}  ({time.perf_counter() - t0:.2f}s)")
    print(f"Implied vol recovered from BS price: {implied_vol(exact, S, K, T, r):.6f}")


if __name__ == "__main__":
    backtest_demo()
    pricing_demo()

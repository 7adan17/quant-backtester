"""European option pricing: closed-form Black-Scholes and Monte Carlo.

The Monte Carlo pricer uses antithetic variates plus a control variate (the
discounted terminal stock price, whose expectation is known exactly) to cut
variance. Greeks use pathwise estimators on the same simulated paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, call: bool = True) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, call: bool = True) -> dict:
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    delta = _norm_cdf(d1) if call else _norm_cdf(d1) - 1
    gamma = _norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * _norm_pdf(d1) * sqrt(T)
    if call:
        theta = -S * _norm_pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * _norm_cdf(d2)
    else:
        theta = -S * _norm_pdf(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * _norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price: float, S: float, K: float, T: float, r: float, call: bool = True,
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton-Raphson with a bisection fallback when vega gets too small."""
    lo, hi, sigma = 1e-4, 5.0, 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, call) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        vega = bs_greeks(S, K, T, r, sigma, call)["vega"]
        step = sigma - diff / vega if vega > 1e-8 else None
        sigma = step if step is not None and lo < step < hi else 0.5 * (lo + hi)
    return sigma


@dataclass
class MCResult:
    price: float
    std_error: float
    delta: float
    vega: float

    def ci95(self) -> tuple:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def mc_price(S: float, K: float, T: float, r: float, sigma: float, call: bool = True,
             n_paths: int = 200_000, seed: int | None = None,
             antithetic: bool = True, control_variate: bool = True) -> MCResult:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (r - 0.5 * sigma**2) * T
    ST = S * np.exp(drift + sigma * sqrt(T) * z)
    disc = exp(-r * T)
    payoff = disc * (np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0))

    if control_variate:
        # E[disc * ST] = S under the risk-neutral measure.
        control = disc * ST
        cov = np.cov(payoff, control)
        beta = cov[0, 1] / cov[1, 1]
        estimator = payoff - beta * (control - S)
    else:
        estimator = payoff

    if antithetic:
        # Pair up antithetic draws so the standard error accounts for their correlation.
        half = len(estimator) // 2
        samples = 0.5 * (estimator[:half] + estimator[half:])
    else:
        samples = estimator

    itm = ST > K if call else ST < K
    sign = 1.0 if call else -1.0
    delta = np.mean(disc * sign * itm * ST / S)
    vega = np.mean(disc * sign * itm * ST * (np.log(ST / S) - (r + 0.5 * sigma**2) * T) / sigma)

    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / sqrt(len(samples))),
        delta=float(delta),
        vega=float(vega),
    )

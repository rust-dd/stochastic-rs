"""Quant-surface pytest coverage.

Exercises the analytic Black-Scholes-Merton pricer binding: ATM price
range, monotonicity in strike, and the no-arbitrage call price bounds.
"""

from __future__ import annotations

import stochastic_rs as sr


def test_bsm_atm_price_range():
    p = sr.BSMPricer(s=100.0, v=0.2, k=100.0, r=0.05, tau=1.0)
    price = p.price()
    assert 9.0 < price < 12.0


def test_bsm_call_decreasing_in_strike():
    prices = [
        sr.BSMPricer(s=100.0, v=0.2, k=k, r=0.05, tau=1.0).price()
        for k in (80.0, 100.0, 120.0)
    ]
    assert prices[0] > prices[1] > prices[2]


def test_bsm_call_within_no_arbitrage_bounds():
    s, k, r, tau = 100.0, 90.0, 0.05, 1.0
    price = sr.BSMPricer(s=s, v=0.25, k=k, r=r, tau=tau).price()
    import math

    lower = max(0.0, s - k * math.exp(-r * tau))
    assert lower <= price <= s


def test_bsm_price_increases_with_vol():
    lo = sr.BSMPricer(s=100.0, v=0.1, k=100.0, r=0.05, tau=1.0).price()
    hi = sr.BSMPricer(s=100.0, v=0.4, k=100.0, r=0.05, tau=1.0).price()
    assert hi > lo


def test_bsm_deep_itm_positive():
    price = sr.BSMPricer(s=200.0, v=0.2, k=100.0, r=0.05, tau=1.0).price()
    assert price > 90.0


def test_bsm_short_maturity_near_intrinsic():
    # Very short maturity ATM: price small but positive.
    price = sr.BSMPricer(s=100.0, v=0.2, k=100.0, r=0.05, tau=0.01).price()
    assert 0.0 < price < 2.0

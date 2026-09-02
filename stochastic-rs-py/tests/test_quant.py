"""Quant-surface pytest coverage.

Exercises the analytic Black-Scholes-Merton pricer binding: ATM price
range, monotonicity in strike, and the no-arbitrage call price bounds.
"""

from __future__ import annotations

import math

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


def test_quanto_pricer_matches_the_reiner_formula():
    kwargs = dict(s=100.0, v=0.2, k=105.0, r=0.08, tau=0.5, r_f=0.05, v_fx=0.12, rho=0.3, fixed_rate=1.5, q=0.04)
    pricer = sr.QuantoPricer(**kwargs)
    call, put = pricer.call_put()
    assert abs(call - 5.2936847941) < 1e-4
    assert abs(put - 12.2976985036) < 1e-4
    assert abs(pricer.forward() - 150.2101470686) < 1e-6
    assert pricer.price() == call
    assert sr.QuantoPricer(option_type="put", **kwargs).price() == put

def test_dual_curve_bootstrap_recovers_the_tenor_curve():
    import numpy as np

    ois = sr.DiscountCurve.from_zero_rates(np.array([0.25, 1.0, 5.0, 10.0]), np.full(4, 0.02), interp="log_df")
    tenor_df = lambda t: np.exp(-0.025 * t)  # noqa: E731
    deposits = [(0.25, (1.0 / tenor_df(0.25) - 1.0) / 0.25)]
    fras = [(0.25, 0.5, (tenor_df(0.25) / tenor_df(0.5) - 1.0) / 0.25)]
    swaps = []
    for years in (1, 2, 3):
        fixed = [float(i) for i in range(1, years + 1)]
        flt = [0.25 * i for i in range(1, 4 * years + 1)]
        float_pv = sum(ois.discount_factor(t) * (tenor_df(s) / tenor_df(t) - 1.0) for s, t in zip([0.0] + flt[:-1], flt))
        annuity = sum((t - s) * ois.discount_factor(t) for s, t in zip([0.0] + fixed[:-1], fixed))
        swaps.append((float_pv / annuity, fixed, flt))
    forecast = sr.DiscountCurve.bootstrap_forecast(ois, deposits, fras, swaps, interp="log_df")
    for t in (0.25, 0.5, 1.0, 2.0, 3.0):
        assert abs(forecast.discount_factor(t) - tenor_df(t)) < 1e-8
    multi = sr.MultiCurve(ois)
    multi.add_forecast("3M", forecast)
    assert abs(multi.basis_spread("3M", 1.0, 1.25) - 0.005) < 2e-4
    assert multi.projected_forward("6M", 1.0, 1.5) is None
    ois2 = sr.DiscountCurve.bootstrap_ois([[1.0], [1.0, 2.0]], [0.0202, 0.0204], interp="log_df")
    assert 0.9 < ois2.discount_factor(2.0) < ois2.discount_factor(1.0) < 1.0

def test_callable_bond_on_the_hull_white_tree():
    tree = sr.HullWhiteCallableBond(initial_rate=0.04, mean_reversion=0.3, theta=0.04, sigma=0.01, horizon=3.0, steps=36)
    straight = tree.price(100.0, 0.06, [1.0, 2.0, 3.0])
    callable_ = tree.price(100.0, 0.06, [1.0, 2.0, 3.0], calls=[(1.0, 100.0), (2.0, 100.0)])
    puttable = tree.price(100.0, 0.06, [1.0, 2.0, 3.0], puts=[(1.0, 100.0), (2.0, 100.0)])
    assert straight[0] == straight[1] and straight[2] == 0.0 and straight[3] == 0.0
    assert callable_[0] < straight[0] < puttable[0]
    assert abs(callable_[2] - (straight[0] - callable_[0])) < 1e-12
    assert abs(puttable[3] - (puttable[0] - straight[0])) < 1e-12
    flat = sr.HullWhiteCallableBond(0.02, 0.3, 0.02, 1e-9, 3.0, 36)
    called = flat.price(100.0, 0.05, [1.0, 2.0, 3.0], calls=[(1.0, 100.0), (2.0, 100.0)])
    assert abs(called[0] - 105.0 * math.exp(-0.02)) < 1e-6

def test_bachelier_pricer_round_trips_the_normal_volatility():
    pricer = sr.BachelierPricer(s=100.0, v=20.0, k=95.0, r=0.05, tau=0.75, q=0.02)
    call, put = pricer.call_put()
    forward = pricer.forward()
    assert abs(call - put - math.exp(-0.05 * 0.75) * (forward - 95.0)) < 1e-9
    assert pricer.price() == call
    assert abs(pricer.implied_volatility(call, "call") - 20.0) < 1e-9
    assert abs(pricer.implied_volatility(put, "put") - 20.0) < 1e-9
    assert pricer.vega() > 0.0
    assert math.isnan(pricer.implied_volatility(0.0, "call"))

def test_tree_swaption_calibrators_fit_a_small_grid():
    import numpy as np

    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, 0.03), interp="log_df")
    quotes = [(1.0, 2.0, 0.22, 0.5, "payer"), (2.0, 2.0, 0.20, 0.5, "payer")]
    a, sigma, rmse, converged = sr.BlackKarasinskiSwaptionCalibrator(
        quotes, curve, initial_rate=0.03, long_run_rate=0.03, steps_per_year=8, max_iters=200
    ).calibrate(initial_guess=(0.1, 0.2))
    assert a > 0.0 and sigma > 0.0 and math.isfinite(rmse)
    hw = sr.HullWhiteSwaptionCalibrator(quotes, curve).calibrate()
    assert hw[1] > 0.0 and math.isfinite(hw[2])
    g2 = sr.G2ppSwaptionCalibrator(quotes, curve, initial_rate=0.03, steps_per_year=4, max_iters=60).calibrate()
    assert len(g2) == 7 and abs(g2[4]) < 1.0 and math.isfinite(g2[5])

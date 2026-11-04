"""Quant-surface pytest coverage.

Exercises the analytic Black-Scholes-Merton pricer binding: ATM price
range, monotonicity in strike, and the no-arbitrage call price bounds.
"""

from __future__ import annotations

import pytest
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


BELOW_FELLER = (0.04, 1.5, 0.04, 0.5, -0.7)
HESTON_STRIKES = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0]
HESTON_MATURITIES = (0.25, 0.5, 1.0, 2.0, 3.0)


def _heston_surface(params, s0=100.0, r=0.02, q=0.0):
    model = sr.HestonFourier(*params, r, q)
    return [
        sr.MarketSlice(
            HESTON_STRIKES,
            [model.price_call(s0, k, r, q, t) for k in HESTON_STRIKES],
            [True] * len(HESTON_STRIKES),
            t,
        )
        for t in HESTON_MATURITIES
    ]


def test_heston_calibrator_reports_the_fitted_pricer_rmse_and_termination():
    calibrator = sr.HestonCalibrator(_heston_surface(BELOW_FELLER), 100.0, 0.02, "call", 0.0)
    assert len(calibrator.calibrate()) == 7
    *fit, converged, rmse, termination = calibrator.calibrate(
        initial=BELOW_FELLER, full_output=True
    )
    assert converged and termination.startswith(("Converged", "Orthogonal", "ResidualsZero"))
    assert max(abs(a - b) for a, b in zip(fit, BELOW_FELLER)) < 1e-4
    assert rmse < 1e-6


def test_heston_calibrator_options_reach_the_rust_calibrator():
    slices = _heston_surface(BELOW_FELLER)
    weights = [1.0 + abs(k / 100.0 - 1.0) for _ in HESTON_MATURITIES for k in HESTON_STRIKES]
    for options in ({"jacobian_method": "fd"}, {"residual_weights": weights}):
        *fit, converged, _ = sr.HestonCalibrator(slices, 100.0, 0.02, "call", 0.0, **options).calibrate()
        assert converged and max(abs(a - b) for a, b in zip(fit, BELOW_FELLER)) < 1e-3, options
    v0, kappa, theta, sigma, rho, *_ = sr.HestonCalibrator(
        slices, 100.0, 0.02, "call", 0.0, feller_projection=True
    ).calibrate()
    assert sigma * sigma <= 2.0 * kappa * theta * (1.0 + 1e-12)
    for bad in ({"jacobian_method": "newton"}, {"residual_weights": [1.0]}):
        with pytest.raises(ValueError):
            sr.HestonCalibrator(slices, 100.0, 0.02, "call", 0.0, **bad)

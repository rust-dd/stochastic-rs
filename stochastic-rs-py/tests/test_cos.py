"""`CosEngine`: Fang and Oosterlee's COS method on one strike or a strike array."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import stochastic_rs as sr


def black_scholes_call(s, k, r, q, tau, sigma):
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    return s * math.exp(-q * tau) * cdf(d1) - k * math.exp(-r * tau) * cdf(d1 - sigma * math.sqrt(tau))


def test_cos_engine_reproduces_the_black_scholes_call_and_put_on_a_strike_array():
    s, r, q, tau, sigma = 100.0, 0.05, 0.02, 1.0, 0.2
    strikes = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0])
    model = sr.BSMFourier(sigma)
    for engine in (sr.CosEngine(), sr.CosEngine(256, 10.0)):
        calls = engine.price_strikes(model, s, strikes, r, q, tau)
        puts = engine.price_strikes(model, s, strikes, r, q, tau, option_type="put")
        closed = np.array([black_scholes_call(s, k, r, q, tau, sigma) for k in strikes])
        assert np.abs(calls - closed).max() < 1e-12
        assert np.abs(puts - (closed - s * math.exp(-q * tau) + strikes * math.exp(-r * tau))).max() < 1e-12
        assert list(calls) == [engine.price(model, s, k, r, q, tau) for k in strikes]
        assert list(puts) == [engine.price(model, s, k, r, q, tau, "put") for k in strikes]


def test_cos_engine_prices_every_fourier_model_as_its_own_quadrature_does():
    models = [
        sr.BSMFourier(0.2),
        sr.HestonFourier(0.04, 1.5, 0.04, 0.3, -0.7),
        sr.DoubleHestonFourier(0.02, 1.5, 0.02, 0.3, -0.7, 0.02, 0.5, 0.02, 0.2, -0.3),
        sr.BatesFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, -0.05, 0.1),
        sr.HKDEFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, 0.4, 10.0, 5.0),
        sr.VarianceGammaFourier(0.2, -0.1, 0.2),
        sr.CGMYFourier(1.0, 5.0, 10.0, 0.5),
        sr.MertonJDFourier(0.2, 0.5, -0.05, 0.1),
        sr.KouFourier(0.2, 0.5, 0.4, 10.0, 5.0),
    ]
    engine = sr.CosEngine()
    for model in models:
        for k in (90.0, 100.0, 110.0):
            cos = engine.price(model, 100.0, k, 0.05, 0.02, 1.0)
            assert abs(cos - model.price_call(100.0, k, 0.05, 0.02, 1.0)) < 1e-6, (type(model).__name__, k)
    assert engine.price(sr.NigFourier(15.0, -5.0, 0.5), 100.0, 100.0, 0.05, 0.02, 1.0) > 0.0


def test_cos_engine_refuses_a_bad_grid_a_bad_option_type_and_a_model_it_cannot_price():
    with pytest.raises(ValueError, match=re.escape("n must satisfy `n >= 2`, got n = 1")):
        sr.CosEngine(1)
    for l, shown in ((0.0, "0.0"), (math.inf, "inf"), (math.nan, "NaN")):
        message = re.escape(f"l must satisfy `l.is_finite() && l > 0.0`, got l = {shown}")
        with pytest.raises(ValueError, match=message):
            sr.CosEngine(l=l)
        with pytest.raises(ValueError, match=message):
            sr.CosEngine(256, l)
    engine, model = sr.CosEngine(), sr.BSMFourier(0.2)
    assert sr.CosEngine(l=10.0).price(model, 100.0, 90.0, 0.05, 0.02, 1.0) == engine.price(model, 100.0, 90.0, 0.05, 0.02, 1.0)
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put', got 'x'"):
        engine.price(model, 100.0, 100.0, 0.05, 0.02, 1.0, option_type="x")
    with pytest.raises(TypeError, match="model must be one of the Fourier models .*, got CarrMadanPricer"):
        engine.price(sr.CarrMadanPricer(), 100.0, 100.0, 0.05, 0.02, 1.0)


def test_cos_engine_answers_the_payoff_at_expiry_and_nan_off_its_query_domain():
    engine, model = sr.CosEngine(), sr.BSMFourier(0.25)
    strikes = np.array([90.0, 0.0, 110.0])
    prices = engine.price_strikes(model, 100.0, strikes, 0.05, 0.0, 1.0)
    assert math.isnan(prices[1]) and math.isnan(engine.price(model, 100.0, 0.0, 0.05, 0.0, 1.0))
    assert prices[0] == engine.price(model, 100.0, 90.0, 0.05, 0.0, 1.0) > 0.0
    for s, r in ((math.inf, 0.05), (100.0, math.inf)):
        assert np.isnan(engine.price_strikes(model, s, strikes, r, 0.0, 1.0)).all()
        assert math.isnan(engine.price(model, s, 100.0, r, 0.0, 1.0))
    assert list(engine.price_strikes(model, 100.0, strikes, 0.05, 0.0, 0.0)) == [10.0, 100.0, 0.0]
    assert engine.price(model, 100.0, 110.0, 0.05, 0.0, 0.0, "put") == 10.0
    for tau in (-1.0, math.nan, math.inf):
        assert np.isnan(engine.price_strikes(model, 100.0, strikes, 0.05, 0.0, tau)).all()

"""Curve building: tenor-keyed forecast curves, the bootstrap's errors, the short end below a
curve's first pillar and the interpolation between pillars."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import stochastic_rs as sr

INTERPS = ["linear", "log_df", "cubic", "monotone_convex"]
NAN = float("nan")


def flat(rate):
    return sr.DiscountCurve.from_zero_rates(np.array([0.25, 1.0, 5.0, 10.0]), np.full(4, rate), interp="log_df")


def assert_close(got, want):
    """A bootstrapped read passes through libm and a root: 1e-15, not the last bit."""
    assert len(got) == len(want) and all(abs(g - w) <= 1e-15 for g, w in zip(got, want)), (got, want)


def test_multi_curve_reads_a_tenor_string_and_rejects_one_that_names_no_tenor():
    ois, forecast = flat(0.02), flat(0.025)
    for tenor in ["3M", "3m", "O/N", "ON", "1M", "6M", "12M", "1Y"]:
        multi = sr.MultiCurve(ois)
        multi.add_forecast(tenor, forecast)
        assert multi.projected_forward(tenor, 1.0, 1.25) == 0.025078288015047967
        assert multi.basis_spread(tenor, 1.0, 1.25) == 0.005028204577444129
        assert multi.fair_swap_rate(tenor, [0.0, 0.5, 1.0, 1.5, 2.0]) == 0.025156903081268722
    multi = sr.MultiCurve(ois)
    multi.add_forecast("12M", forecast)
    assert multi.projected_forward("1Y", 1.0, 1.25) == 0.025078288015047967
    assert multi.projected_forward("6M", 1.0, 1.5) is None
    assert multi.discount_factor(1.0) == ois.discount_factor(1.0)
    for call in (
        lambda: multi.add_forecast("2W", forecast),
        lambda: multi.projected_forward("2W", 1.0, 1.5),
        lambda: multi.basis_spread("", 1.0, 1.5),
        lambda: multi.fair_swap_rate("CMS", [0.0, 1.0]),
    ):
        with pytest.raises(ValueError, match="tenor must be one of"):
            call()


def test_the_zero_rate_methods_are_flat_below_the_first_pillar_and_monotone_convex_starts_at_the_origin():
    times, rates = np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.03, 0.032, 0.035, 0.04])
    # cubic is scipy's CubicSpline(bc_type="natural") on the zero rates; monotone_convex is Hagan &
    # West with tau_0 = 0
    from_the_first_pillar_on = {
        "linear": [0.9851119396030626, 0.9770181986892416, 0.9509916468680314, 0.8958341352965283, 0.7557837414557255],
        "log_df": [0.9851119396030626, 0.9767739746688439, 0.950278670532427, 0.8928529928432785, 0.7557837414557255],
        "cubic": [0.9851119396030626, 0.9769982868364686, 0.9507668466123762, 0.8946664177101229, 0.7557837414557255],
        "monotone_convex": [0.9851119396030626, 0.9769774904457238, 0.950753928672327, 0.8948393168143698, 0.7557837414557255],
    }
    for interp in INTERPS:
        curve = sr.DiscountCurve.from_zero_rates(times, rates, interp=interp)
        if interp == "monotone_convex":
            below = [curve.discount_factor(t) for t in (0.05, 0.1, 0.25, 0.4)]
            assert below == [0.9985505514666794, 0.997100212529285, 0.9927141712772897, 0.9882140054334019]
            for t1, t2 in ((0.5 - 1e-7, 0.5), (0.5, 0.5 + 1e-7)):
                assert abs(curve.forward_rate(t1, t2) - 0.032) < 1e-6
        else:
            for t in (0.05, 0.1, 0.25, 0.4):
                assert abs(curve.discount_factor(t) - math.exp(-0.03 * t)) < 1e-15
                assert abs(curve.zero_rate(t) - 0.03) < 1e-13
            assert abs(curve.forward_rate(0.1, 0.3) - 0.03) < 1e-13
        assert [curve.discount_factor(t) for t in (0.5, 0.75, 1.5, 3.0, 7.0)] == from_the_first_pillar_on[interp]
        # A pillar at t = 0 is none: the curve is 1 there whatever rate it carries.
        with_origin = sr.DiscountCurve.from_zero_rates(np.append(0.0, times), np.append(0.07, rates), interp=interp)
        for t in (0.0, 0.1, 0.5, 0.75, 3.0, 7.0):
            assert with_origin.discount_factor(t) == curve.discount_factor(t)
    schedules, par = [[0.5], [0.5, 1.0], [0.5, 1.0, 1.5, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]], [0.03, 0.031, 0.033, 0.036]
    between_pillars = {
        "cubic": [0.9970267065196697, 0.9775543514599756, 0.95341309136086, 0.9026089232232755, 0.8694864074212039],
        "monotone_convex": [0.997074211487444, 0.9775513069916826, 0.953270279696725, 0.903217911261217, 0.8699591028427767],
    }
    for interp, pins in between_pillars.items():
        curve = sr.DiscountCurve.bootstrap_ois(schedules, par, interp=interp)
        assert_close([curve.discount_factor(t) for t in (0.1, 0.75, 1.5, 3.0, 4.0)], pins)


def test_a_log_linear_bootstrap_stays_and_a_linear_one_moves_after_a_read_below_its_first_pillar():
    schedules, rates = [[0.5], [0.25, 0.5, 0.75, 1.0], [0.25 * i for i in range(1, 9)]], [0.03, 0.031, 0.033]
    reads = (0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
    log_df = sr.DiscountCurve.bootstrap_ois(schedules, rates, interp="log_df")
    assert_close([log_df.discount_factor(t) for t in reads], [
        0.9985122465546779,
        0.9970267065196697,
        0.9925833339709303,
        0.9881597638602578,
        0.9852216748768474,
        0.9773711175149336,
        0.9695831158726758,
        0.9528031937739929,
        0.9363136704879836,
    ])
    linear = sr.DiscountCurve.bootstrap_ois(schedules, rates, interp="linear")
    assert_close([linear.discount_factor(t) for t in reads], [
        0.9985122465546779,
        0.9970267065196697,
        0.9925833339709303,
        0.9881597638602578,
        0.9852216748768474,
        0.9775065696759145,
        0.9695820741914576,
        0.9532779954762216,
        0.9363028823658401,
    ])
    swaps = [(0.032, [1.0], [0.25, 0.5, 0.75, 1.0]), (0.034, [1.0, 2.0], [0.25 * i for i in range(1, 9)])]
    forecast = sr.DiscountCurve.bootstrap_forecast(flat(0.02), [(0.5, 0.03)], [], swaps, interp="log_df")
    assert_close([forecast.discount_factor(t) for t in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)], [
        0.9970267065196697,
        0.9925833339709303,
        0.9852216748768474,
        0.9688515555947406,
        0.9517530767743727,
        0.9349563552008011,
    ])
    forecast = sr.DiscountCurve.bootstrap_forecast(flat(0.02), [(0.5, 0.03)], [], swaps, interp="linear")
    assert_close([forecast.discount_factor(t) for t in (1.0, 2.0)], [0.9688504774188735, 0.9349496554140198])


def test_a_bootstrap_quote_that_gives_no_pillar_raises_value_error():
    ois = flat(0.02)
    cases = [
        ("needs at least 1", lambda: sr.DiscountCurve.bootstrap_ois([], [])),
        ("fixed_times must satisfy", lambda: sr.DiscountCurve.bootstrap_ois([[]], [0.02])),
        ("rate must satisfy", lambda: sr.DiscountCurve.bootstrap_ois([[1.0]], [NAN])),
        ("quote 1 cannot be used: maturity must differ", lambda: sr.DiscountCurve.bootstrap_ois([[1.0], [0.5, 1.0]], [0.02, 0.021])),
        ("strictly increasing", lambda: sr.DiscountCurve.bootstrap_ois([[1.0, 0.5, 2.0]], [0.02])),
        ("needs at least 1", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [], [], [])),
        ("rate must satisfy", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [(0.5, NAN)], [], [])),
        ("quote 1 cannot be used: start must satisfy", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [(0.25, 0.03)], [(NAN, 0.5, 0.03)], [])),
        ("quote 1 cannot be used: rate must satisfy", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [(0.25, 0.03)], [(0.25, 0.5, NAN)], [])),
        ("no earlier pillar", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [], [(0.25, 0.5, 0.03)], [])),
        ("fixed_times must satisfy", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [], [], [(0.03, [], [])])),
        ("float_times must end at the maturity", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [], [], [(0.03, [1.0, 2.0], [0.5, 1.0, 1.5])])),
        ("no pseudo-discount factor brackets swap quote 1", lambda: sr.DiscountCurve.bootstrap_forecast(ois, [(0.5, 0.03)], [], [(-5.0, [1.0], [0.5, 1.0])])),
    ]
    for message, build in cases:
        with pytest.raises(ValueError, match=message):
            build()


def test_a_discount_curve_needs_distinct_times_from_the_origin_on():
    for times, message in (([1.0, 1.0, 2.0], "a.time != b.time"), ([-1.0, 1.0, 2.0], "p.time >= T::zero"), ([NAN, 1.0, 2.0], "is_finite")):
        with pytest.raises(ValueError, match=message):
            sr.DiscountCurve.from_zero_rates(np.array(times), np.full(3, 0.03), interp="linear")


SVENSSON = (0.04, -0.02, 0.01, 0.005, 0.5, 2.0)
NELSON_SIEGEL = (0.04, -0.02, 0.01, 0.5)
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])


def svensson_zero(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    """Svensson (1994) zero rate from the formula, the loadings through `expm1`."""
    x1, x2 = tau / lambda1, tau / lambda2
    slope1, slope2 = -math.expm1(-x1) / x1, -math.expm1(-x2) / x2
    return beta0 + beta1 * slope1 + beta2 * (slope1 - math.exp(-x1)) + beta3 * (slope2 - math.exp(-x2))


def test_svensson_reads_its_closed_forms_at_a_point_and_on_an_array():
    curve = sr.Svensson(*SVENSSON)
    assert curve.params() == SVENSSON
    taus = np.array([0.01, 0.5, 1.0, 5.0, 30.0, 100.0])
    zeros, forwards, discounts = curve.zero_rates(taus), curve.forward_rates(taus), curve.discount_factors(taus)
    assert np.abs(zeros - [svensson_zero(t, *SVENSSON) for t in taus]).max() < 1e-15
    assert np.abs(discounts - np.exp(-zeros * taus)).max() < 1e-15
    assert list(zeros) == [curve.zero_rate(t) for t in taus]
    assert list(forwards) == [curve.forward_rate(t) for t in taus]
    assert list(discounts) == [curve.discount_factor(t) for t in taus]
    # The instantaneous forward is the derivative of tau * y(tau).
    h = 1e-5
    slope = ((taus + h) * curve.zero_rates(taus + h) - (taus - h) * curve.zero_rates(taus - h)) / (2 * h)
    assert np.abs(slope - forwards).max() < 1e-9
    assert curve.zero_rate(0.0) == curve.forward_rate(0.0) == SVENSSON[0] + SVENSSON[1]
    assert curve.discount_factor(0.0) == 1.0


def test_a_svensson_fit_recovers_the_curve_that_made_its_quotes():
    taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    quotes = sr.Svensson(*SVENSSON).zero_rates(taus)
    fitted = sr.Svensson.fit_curve(taus, quotes)
    assert np.abs(fitted.zero_rates(taus) - quotes).max() < 1e-12
    assert all(abs(got - want) < 1e-9 for got, want in zip(fitted.params(), SVENSSON))


def test_a_parametric_curve_is_nan_at_a_maturity_with_no_rate():
    for curve in (sr.Svensson(*SVENSSON), sr.NelsonSiegel(*NELSON_SIEGEL)):
        for tau in (-1.0, math.inf, -math.inf, NAN):
            values = (curve.zero_rate(tau), curve.forward_rate(tau), curve.discount_factor(tau))
            assert all(math.isnan(v) for v in values), (type(curve).__name__, tau, values)
        assert curve.zero_rate(0.0) == curve.forward_rate(0.0) == 0.02
        assert curve.discount_factor(0.0) == 1.0
    assert np.isnan(sr.Svensson(*SVENSSON).zero_rates(np.array([-1.0, math.inf]))).all()


def test_svensson_refuses_a_parameter_off_its_domain_and_quotes_it_cannot_fit():
    for args, message in (
        ((NAN, -0.02, 0.01, 0.005, 0.5, 2.0), "beta0 must satisfy `beta0.is_finite()`, got beta0 = NaN"),
        ((0.04, -0.02, 0.01, math.inf, 0.5, 2.0), "beta3 must satisfy `beta3.is_finite()`, got beta3 = inf"),
        (
            (0.04, -0.02, 0.01, 0.005, 0.0, 2.0),
            "lambda1 must satisfy `lambda1.is_finite() && lambda1 > T::zero()`, got lambda1 = 0.0",
        ),
        (
            (0.04, -0.02, 0.01, 0.005, 0.5, -1.0),
            "lambda2 must satisfy `lambda2.is_finite() && lambda2 > T::zero()`, got lambda2 = -1.0",
        ),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            sr.Svensson(*args)
    quotes = sr.Svensson(*SVENSSON).zero_rates(MATURITIES)
    with pytest.raises(ValueError, match="market rates, one per maturity: expected 11, got 10"):
        sr.Svensson.fit_curve(MATURITIES, quotes[:10])
    with pytest.raises(ValueError, match="quotes for a Svensson fit: needs at least 6, got 5"):
        sr.Svensson.fit_curve(MATURITIES[:5], quotes[:5])
    nan_quote = quotes.copy()
    nan_quote[4] = NAN
    rate = "quote 4 cannot be used: rate must satisfy `rate.is_finite()`, got rate = NaN"
    with pytest.raises(ValueError, match=re.escape(rate)):
        sr.Svensson.fit_curve(MATURITIES, nan_quote)
    maturity = "quote 0 cannot be used: maturity must satisfy `maturity.is_finite() && maturity >= T::zero()`"
    with pytest.raises(ValueError, match=re.escape(maturity)):
        sr.Svensson.fit_curve(-MATURITIES, quotes)
    with pytest.raises(TypeError):
        sr.Svensson(*SVENSSON).zero_rates(MATURITIES.astype(np.float32))


def test_nelson_siegel_refuses_a_parameter_off_its_domain_and_quotes_it_cannot_fit():
    for args, message in (
        ((NAN, -0.02, 0.01, 0.5), "beta0 must satisfy `beta0.is_finite()`, got beta0 = NaN"),
        ((0.04, -0.02, 0.01, 0.0), "lambda must satisfy `lambda.is_finite() && lambda > T::zero()`, got lambda = 0.0"),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            sr.NelsonSiegel(*args)
    curve = sr.NelsonSiegel(*NELSON_SIEGEL)
    quotes = np.array([curve.zero_rate(t) for t in MATURITIES])
    fitted = sr.NelsonSiegel.fit_curve(MATURITIES, quotes)
    assert max(abs(fitted.zero_rate(t) - q) for t, q in zip(MATURITIES, quotes)) < 1e-12
    with pytest.raises(ValueError, match="market rates, one per maturity: expected 11, got 10"):
        sr.NelsonSiegel.fit_curve(MATURITIES, quotes[:10])
    with pytest.raises(ValueError, match="quotes for a Nelson-Siegel fit: needs at least 4, got 3"):
        sr.NelsonSiegel.fit_curve(MATURITIES[:3], quotes[:3])
    nan_quote = quotes.copy()
    nan_quote[4] = NAN
    rate = "quote 4 cannot be used: rate must satisfy `rate.is_finite()`, got rate = NaN"
    with pytest.raises(ValueError, match=re.escape(rate)):
        sr.NelsonSiegel.fit_curve(MATURITIES, nan_quote)

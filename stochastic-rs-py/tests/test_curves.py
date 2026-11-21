"""Curve building: tenor-keyed forecast curves, the bootstrap's errors, the short end below a
curve's first pillar and the interpolation between pillars."""

from __future__ import annotations

import math

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

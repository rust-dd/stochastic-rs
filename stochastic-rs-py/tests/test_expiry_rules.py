"""The pricers at expiry, Kemna and Vorst's geometric Asian, and a barrier at zero volatility."""

import math

import pytest
import stochastic_rs as sr

QUERY = dict(k=100.0, r=0.05, q=0.02)


@pytest.mark.parametrize(
    "s, k, r, q, tau, v, call, put",
    [
        # The law of ln G integrated by mpmath at 50 digits.
        (100.0, 105.0, 0.05, 0.02, 0.75, 0.25, 3.1533660465714502, 7.2593885793044147),
        (110.0, 100.0, 0.03, 0.0, 2.0, 0.4, 18.307267848961415, 8.5437326880311886),
        # Haug (2007) 4.20.1 prints this put as 4.6922.
        (80.0, 85.0, 0.05, -0.03, 0.25, 0.2, 0.48188555462727999, 4.6922213122453357),
    ],
)
def test_the_geometric_asian_is_kemna_and_vorsts(s, k, r, q, tau, v, call, put):
    got = sr.AsianPricer(s=s, v=v, k=k, r=r, tau=tau, q=q).call_put()
    assert got == pytest.approx((call, put), rel=1e-12)


@pytest.mark.parametrize("s", [90.0, 100.0, 110.0])
@pytest.mark.parametrize(
    "build, scale",
    [
        (lambda **kw: sr.AsianPricer(v=0.25, **kw), 1.0),
        (lambda **kw: sr.BjerksundStensland2002Pricer(v=0.25, **kw), 1.0),
        (lambda **kw: sr.BachelierPricer(v=20.0, **kw), 1.0),
        (lambda **kw: sr.QuantoPricer(v=0.2, r_f=0.05, v_fx=0.12, rho=0.3, fixed_rate=1.5, **kw), 1.5),
    ],
)
def test_an_expired_option_is_its_payoff_and_a_negative_maturity_nan(build, scale, s):
    call, put = build(s=s, tau=0.0, **QUERY).call_put()
    assert (call, put) == (scale * max(s - 100.0, 0.0), scale * max(100.0 - s, 0.0))
    assert all(math.isnan(x) for x in build(s=s, tau=-1.0, **QUERY).call_put())


def test_the_bachelier_vega_is_nan_at_expiry():
    assert math.isnan(sr.BachelierPricer(s=110.0, v=20.0, tau=0.0, **QUERY).vega())
    assert sr.BachelierPricer(s=110.0, v=20.0, tau=1.0, **QUERY).vega() > 0.0


def test_bsm_is_nan_for_a_spot_that_is_not_positive_before_expiry():
    assert all(math.isnan(x) for x in sr.BSMPricer(s=0.0, v=0.2, tau=1.0, **QUERY).call_put())
    assert sr.BSMPricer(s=0.0, v=0.2, tau=0.0, **QUERY).call_put() == (0.0, 100.0)


@pytest.mark.parametrize("s, k", [(0.0, 100.0), (-5.0, 100.0), (100.0, 0.0), (100.0, -5.0)])
@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: sr.BSMPricer(v=0.2, **kw),
        lambda **kw: sr.AsianPricer(v=0.25, **kw),
        lambda **kw: sr.BjerksundStensland2002Pricer(v=0.25, **kw),
        lambda **kw: sr.QuantoPricer(v=0.2, r_f=0.05, v_fx=0.12, rho=0.3, fixed_rate=1.5, **kw),
    ],
)
def test_a_lognormal_pricer_is_nan_for_a_spot_or_strike_that_is_not_positive(build, s, k):
    assert all(math.isnan(x) for x in build(s=s, k=k, r=0.05, q=0.02, tau=1.0).call_put())
    assert not any(math.isnan(x) for x in build(s=s, k=k, r=0.05, q=0.02, tau=0.0).call_put())


def test_bachelier_prices_a_negative_spot_and_strike():
    call, put = sr.BachelierPricer(s=-5.0, v=20.0, k=-3.0, r=0.05, q=0.02, tau=1.0).call_put()
    assert call > 0.0 and put > 0.0


@pytest.mark.parametrize("barrier_type, h", [("up_in", 97.0), ("down_in", 103.0)])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_a_knocked_in_barrier_at_zero_volatility_with_its_forward_at_the_strike(barrier_type, h, option_type):
    pricer = sr.BarrierPricer(
        s=100.0, k=100.0, h=h, r=0.03, q=0.03, sigma=0.0, t=1.0,
        barrier_type=barrier_type, option_type=option_type, rebate=3.0,
    )
    assert pricer.price() == 0.0


@pytest.mark.parametrize(
    "barrier_type, option_type, h, want",
    [
        # The path 100 e^{0.03 t} ends at 103.0454 and meets 102 at t* = ln(1.02) / 0.03.
        ("up_out", "call", 102.0, 3.0 * math.exp(-0.05 * math.log(1.02) / 0.03)),
        ("up_out", "call", 120.0, math.exp(-0.05) * (100.0 * math.exp(0.03) - 95.0)),
        ("up_in", "call", 102.0, math.exp(-0.05) * (100.0 * math.exp(0.03) - 95.0)),
        ("up_in", "put", 120.0, 3.0 * math.exp(-0.05)),
    ],
)
def test_a_barrier_at_zero_volatility_prices_its_deterministic_path(barrier_type, option_type, h, want):
    pricer = sr.BarrierPricer(
        s=100.0, k=95.0, h=h, r=0.05, q=0.02, sigma=0.0, t=1.0,
        barrier_type=barrier_type, option_type=option_type, rebate=3.0,
    )
    assert pricer.price() == pytest.approx(want, rel=1e-12)

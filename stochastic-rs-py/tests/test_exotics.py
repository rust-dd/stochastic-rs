"""The exotic pricers' Python classes against their Rust values."""

import math

import pytest
import stochastic_rs as sr

LIBM = {"rel": 1e-12, "abs": 0.0}
BARRIER = dict(s=100.0, k=100.0, r=0.05, q=0.02, sigma=0.25)
CORRIDOR = dict(k=100.0, h_upper=130.0, h_lower=70.0, r=0.05, q=0.02, sigma=0.25)
LOOKBACK = dict(s=100.0, r=0.05, q=0.02, sigma=0.25)


@pytest.mark.parametrize(
    "barrier_type, option_type, plain, with_rebate",
    [
        ("up_in", "call", 6.436584456764569, 8.455776261906534),
        ("up_in", "put", 0.14288692243022005, 2.162078727572185),
        ("up_out", "call", 4.687177471293554, 5.54111046936928),
        ("up_out", "put", 8.083950125023785, 8.93788312309951),
        ("down_in", "call", 0.3854876045521558, 2.1726177751007674),
        ("down_in", "put", 7.055231729522353, 8.842361900070964),
        ("down_out", "call", 10.738274323505967, 11.83254742187439),
        ("down_out", "put", 1.1716053179316517, 2.2658784163000742),
    ],
)
def test_the_barrier_pricer_keeps_its_prices(barrier_type, option_type, plain, with_rebate):
    h = 130.0 if barrier_type.startswith("up") else 80.0
    kw = dict(h=h, t=1.0, barrier_type=barrier_type, option_type=option_type, **BARRIER)
    assert sr.BarrierPricer(**kw).price() == pytest.approx(plain, **LIBM)
    assert sr.BarrierPricer(rebate=3.0, **kw).price() == pytest.approx(with_rebate, **LIBM)


def test_the_double_barrier_pricer_keeps_its_prices():
    pinned = [
        ("call", 90.0, 1.0, 5.52676147834582),
        ("call", 100.0, 1.0, 10.416829363785439),
        ("call", 110.0, 0.25, 11.973631651189004),
        ("put", 90.0, 1.0, 12.946402490967737),
        ("put", 100.0, 0.25, 2.419613886597333),
    ]
    for option_type, s, t, price in pinned:
        pricer = sr.DoubleBarrierPricer(s=s, t=t, option_type=option_type, **CORRIDOR)
        assert pricer.price() == pytest.approx(price, **LIBM), (option_type, s, t)


def test_the_lookback_pricers_keep_their_prices():
    floating = sr.FloatingLookbackPricer
    fixed = sr.FixedLookbackPricer
    pinned = [
        (floating(t=1.0, **LOOKBACK), 19.29808796226954),
        (floating(t=1.0, s_min=90.0, **LOOKBACK), 20.83910394711078),
        (floating(t=1.0, option_type="put", **LOOKBACK), 19.418793165628067),
        (floating(t=1.0, option_type="put", s_max=110.0, **LOOKBACK), 20.878121871030842),
        (fixed(k=90.0, t=1.0, **LOOKBACK), 31.828012291239318),
        (fixed(k=105.0, t=1.0, s_max=110.0, **LOOKBACK), 19.018899629131397),
        (fixed(k=110.0, t=1.0, option_type="put", **LOOKBACK), 25.91345732667257),
        (fixed(k=95.0, t=1.0, option_type="put", s_min=90.0, **LOOKBACK), 13.18603194400309),
        (floating(s=100.0, r=0.05, q=0.05, sigma=0.25, t=1.0), 12.730787441362034),
    ]
    for pricer, price in pinned:
        assert pricer.price() == pytest.approx(price, **LIBM)


def test_the_chooser_compound_and_cliquet_pricers_keep_their_prices():
    haug = sr.SimpleChooserPricer(s=50.0, k=50.0, r=0.08, q=0.0, sigma=0.25, t1=0.25, t=0.5)
    assert haug.price() == pytest.approx(6.10707749816234, **LIBM)
    with_yield = sr.SimpleChooserPricer(s=100.0, k=95.0, r=0.05, q=0.02, sigma=0.3, t1=0.4, t=1.0)
    assert with_yield.price() == pytest.approx(19.398120558726195, **LIBM)
    compound = dict(s=100.0, k1=5.0, k2=100.0, t1=0.25, t2=1.0, r=0.05, q=0.02, sigma=0.25)
    for compound_type, price in [
        ("call_on_call", 6.5922904220493415),
        ("call_on_put", 3.856686630328778),
        ("put_on_call", 0.40641749646062775),
        ("put_on_put", 0.5677385853441723),
    ]:
        assert sr.CompoundPricer(compound_type=compound_type, **compound).price() == pytest.approx(price, **LIBM)
    cliquet = dict(s=100.0, notional=100.0, m=12, t=1.0, r=0.04, q=0.01, sigma=0.25)
    for floor, cap, price in [
        (None, None, 2.8859742821983847),
        (0.0, None, 34.69196090198439),
        (None, 0.04, -13.683474058824492),
        (-0.01, 0.04, 12.670379765513067),
    ]:
        pricer = sr.CliquetPricer(local_floor=floor, local_cap=cap, **cliquet)
        assert pricer.price() == pytest.approx(price, **LIBM)


def test_the_variance_swap_pricer_keeps_its_strikes():
    swap = sr.VarianceSwapPricer(100.0, 0.05, 0.02, 0.5)
    assert swap.forward() == pytest.approx(101.51130646157189, **LIBM)
    assert swap.fair_strike_bsm(0.2) == pytest.approx(0.04, **LIBM)
    assert swap.fair_strike_heston(0.04, 1.5, 0.06) == pytest.approx(0.04592977473976039, **LIBM)
    strikes = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
    otm = [0.02, 0.11, 0.48, 1.57, 5.2, 2.31, 0.83, 0.25, 0.07]
    assert swap.fair_strike_replication(strikes, otm) == pytest.approx(0.043994906076447834, **LIBM)


@pytest.mark.parametrize(
    "build",
    [
        lambda sigma: sr.BarrierPricer(h=130.0, t=1.0, barrier_type="up_out", **{**BARRIER, "sigma": sigma}),
        lambda sigma: sr.DoubleBarrierPricer(s=100.0, t=1.0, **{**CORRIDOR, "sigma": sigma}),
        lambda sigma: sr.FloatingLookbackPricer(t=1.0, **{**LOOKBACK, "sigma": sigma}),
        lambda sigma: sr.FixedLookbackPricer(k=90.0, t=1.0, **{**LOOKBACK, "sigma": sigma}),
        lambda sigma: sr.SimpleChooserPricer(s=100.0, k=95.0, r=0.05, q=0.02, sigma=sigma, t1=0.4, t=1.0),
        lambda sigma: sr.CompoundPricer(s=100.0, k1=5.0, k2=100.0, t1=0.25, t2=1.0, r=0.05, q=0.02, sigma=sigma, compound_type="coc"),
        lambda sigma: sr.CliquetPricer(s=100.0, notional=100.0, m=12, t=1.0, r=0.04, q=0.01, sigma=sigma),
    ],
)
def test_an_exotic_pricer_refuses_a_volatility_it_cannot_divide_by(build):
    for sigma in (0.0, -0.25, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"sigma must satisfy `sigma\.is_finite\(\) && sigma > 0\.0`"):
            build(sigma)

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
        ("up_in", "call", 8.99025449535496, 11.009446300496926),
        ("up_in", "put", 0.14288692243022005, 2.162078727572185),
        ("up_out", "call", 2.133507432703177, 2.987440430778903),
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
        ("call", 90.0, 1.0, 1.7990839293170673),
        ("call", 100.0, 1.0, 2.1200695605968036),
        ("call", 110.0, 0.25, 6.769301632410013),
        ("put", 90.0, 1.0, 4.1516621028716365),
        ("put", 100.0, 0.25, 4.448559533559411),
    ]
    for option_type, s, t, price in pinned:
        pricer = sr.DoubleBarrierPricer(s=s, t=t, option_type=option_type, **CORRIDOR)
        assert pricer.price() == pytest.approx(price, **LIBM), (option_type, s, t)
    tiny = dict(s=100.0, k=100.0, h_lower=70.0, r=0.05, q=-0.03, t=1.0)
    for sigma, h_upper, price in [(1e-9, 108.32870687582458, 6.665563224788113), (1e-10, 108.32870677832874, 6.665564841963476)]:
        pricer = sr.DoubleBarrierPricer(h_upper=h_upper, sigma=sigma, **tiny)
        assert pricer.price() == pytest.approx(price, rel=4.8e-16 / sigma, abs=0.0), sigma


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
        (floating(s=100.0, r=0.05, q=0.05, sigma=0.25, t=1.0), 17.537359445903526),
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
def test_an_exotic_pricer_takes_a_finite_non_negative_volatility(build):
    build(0.0)
    for sigma in (-0.25, math.inf, math.nan):
        with pytest.raises(ValueError, match=r"sigma must satisfy `sigma\.is_finite\(\) && sigma >= 0\.0`"):
            build(sigma)


def test_an_exotic_pricer_at_zero_volatility_prices_its_deterministic_limit_or_nan():
    chooser = sr.SimpleChooserPricer(s=100.0, k=100.0, r=0.05, q=0.02, sigma=0.0, t1=0.5, t=1.0)
    assert chooser.price() == pytest.approx(2.896924880604118, **LIBM)
    floored = sr.CliquetPricer(s=100.0, notional=100.0, m=12, t=1.0, r=0.04, q=0.01, sigma=0.0, local_floor=0.0)
    assert floored.price() == pytest.approx(2.8859742821983847, **LIBM)
    corridor = sr.DoubleBarrierPricer(s=100.0, t=1.0, **{**CORRIDOR, "sigma": 0.0})
    assert corridor.price() == pytest.approx(2.896924880604118, **LIBM)
    barrier = sr.BarrierPricer(h=130.0, t=1.0, barrier_type="up_out", **{**BARRIER, "sigma": 0.0})
    assert barrier.price() == pytest.approx(2.896924880604118, **LIBM)
    for pricer in [
        sr.FloatingLookbackPricer(t=1.0, **{**LOOKBACK, "sigma": 0.0}),
        sr.CompoundPricer(s=100.0, k1=5.0, k2=100.0, t1=0.25, t2=1.0, r=0.05, q=0.02, sigma=0.0, compound_type="coc"),
    ]:
        assert math.isnan(pricer.price())


def test_a_breached_or_expired_barrier_pays_out_and_a_query_outside_the_domain_is_nan():
    knocked_out = dict(s=75.0, k=100.0, h=80.0, r=0.05, q=0.0, sigma=0.2, barrier_type="down_out", option_type="put")
    assert sr.BarrierPricer(t=0.5, **knocked_out).price() == 0.0
    assert sr.BarrierPricer(t=0.0, **knocked_out).price() == 0.0
    up_out = dict(s=135.0, k=100.0, h=130.0, r=0.05, q=0.02, sigma=0.25, t=1.0, barrier_type="up_out", rebate=3.0)
    assert sr.BarrierPricer(**up_out).price() == 3.0
    knocked_in = sr.BarrierPricer(s=75.0, k=100.0, h=80.0, r=0.05, q=0.02, sigma=0.25, t=1.0, barrier_type="down_in")
    vanilla = sr.BSMPricer(s=75.0, v=0.25, k=100.0, r=0.05, tau=1.0, q=0.02)
    assert knocked_in.price() == pytest.approx(vanilla.price(), **LIBM)
    for barrier_type, option_type, rebate, payoff in [
        ("up_in", "call", 3.0, 3.0),
        ("up_out", "put", 0.0, 0.0),
        ("down_out", "call", 0.0, 0.0),
    ]:
        h = 130.0 if barrier_type.startswith("up") else 80.0
        kw = dict(h=h, t=0.0, barrier_type=barrier_type, option_type=option_type, rebate=rebate, **BARRIER)
        assert sr.BarrierPricer(**kw).price() == payoff
    for t in (-1.0, math.nan):
        assert math.isnan(sr.BarrierPricer(h=130.0, t=t, barrier_type="up_out", **BARRIER).price())
        assert math.isnan(sr.DoubleBarrierPricer(s=100.0, t=t, **CORRIDOR).price())
        assert math.isnan(sr.FloatingLookbackPricer(t=t, **LOOKBACK).price())


def test_the_double_barrier_and_the_lookbacks_pay_out_at_expiry():
    assert sr.DoubleBarrierPricer(s=110.0, t=0.0, **CORRIDOR).price() == 10.0
    assert sr.DoubleBarrierPricer(s=90.0, t=0.0, option_type="put", **CORRIDOR).price() == 10.0
    assert sr.DoubleBarrierPricer(s=140.0, t=1.0, **CORRIDOR).price() == 0.0
    assert sr.DoubleBarrierPricer(s=60.0, t=1.0, option_type="put", **CORRIDOR).price() == 0.0
    assert sr.FloatingLookbackPricer(t=0.0, **LOOKBACK).price() == 0.0
    assert sr.FixedLookbackPricer(k=90.0, t=0.0, **LOOKBACK).price() == 10.0
    assert sr.FixedLookbackPricer(k=110.0, t=0.0, option_type="put", **LOOKBACK).price() == 10.0


def test_the_variance_swap_strikes_are_nan_for_a_maturity_outside_their_domain():
    strikes = [90.0, 100.0, 110.0]
    for t in (0.0, -1.0, math.nan):
        assert math.isnan(sr.VarianceSwapPricer(100.0, 0.05, 0.02, t).fair_strike_replication(strikes, [1.0, 2.0, 1.0]))
    assert math.isnan(sr.VarianceSwapPricer(100.0, 0.05, 0.02, -1.0).fair_strike_heston(0.04, 1.5, 0.06))
    assert sr.VarianceSwapPricer(100.0, 0.05, 0.02, 0.0).fair_strike_heston(0.04, 1.5, 0.06) == 0.04


def test_the_cliquet_and_the_simple_chooser_pay_out_at_expiry_and_a_query_outside_the_domain_is_nan():
    cliquet = dict(s=100.0, notional=100.0, m=12, r=0.04, q=0.01, sigma=0.25)
    assert sr.CliquetPricer(t=0.0, local_floor=0.0, **cliquet).price() == 0.0
    assert sr.CliquetPricer(t=0.0, local_floor=0.01, **cliquet).price() == 100.0 * 12.0 * 0.01
    assert math.isnan(sr.CliquetPricer(t=-1.0, **cliquet).price())
    chooser = dict(k=100.0, r=0.05, q=0.02, sigma=0.25)
    assert sr.SimpleChooserPricer(s=110.0, t1=0.0, t=0.0, **chooser).price() == 10.0
    assert sr.SimpleChooserPricer(s=100.0, t1=0.0, t=0.0, **chooser).price() == 0.0
    for t1, t in [(0.0, -1.0), (0.5, 0.0), (1.5, 1.0), (-0.1, 1.0)]:
        assert math.isnan(sr.SimpleChooserPricer(s=100.0, t1=t1, t=t, **chooser).price())
    compound = dict(s=100.0, k1=5.0, k2=100.0, r=0.05, q=0.02, sigma=0.25, compound_type="coc")
    for t1, t2 in [(0.25, 0.1), (0.25, math.nan), (-0.25, 1.0)]:
        assert math.isnan(sr.CompoundPricer(t1=t1, t2=t2, **compound).price())
    strip = sr.VarianceSwapPricer(0.0, 0.05, 0.02, 0.5)
    assert math.isnan(strip.fair_strike_replication([90.0, 100.0, 110.0], [1.0, 2.0, 1.0]))
    assert math.isnan(sr.FixedLookbackPricer(k=90.0, t=1.0, s_max=math.nan, **LOOKBACK).price())
    assert math.isnan(sr.BarrierPricer(h=math.nan, t=0.0, barrier_type="up_out", **BARRIER).price())
    assert math.isnan(sr.DoubleBarrierPricer(s=110.0, t=0.0, **{**CORRIDOR, "h_lower": math.nan}).price())


def test_a_compound_decided_at_0_prices_its_exercise_value():
    compound = dict(k1=5.0, k2=100.0, t1=0.0, r=0.05, q=0.02, sigma=0.25)
    call_on_call = sr.CompoundPricer(s=100.0, t2=1.0, compound_type="coc", **compound)
    assert call_on_call.price() == pytest.approx(6.123761928058123, **LIBM)
    assert sr.CompoundPricer(s=100.0, t2=1.0, compound_type="pop", **compound).price() == 0.0
    assert sr.CompoundPricer(s=110.0, t2=0.0, compound_type="pop", **compound).price() == 5.0


def test_the_exotics_answer_at_their_edges():
    assert sr.VarianceSwapPricer(100.0, 0.05, 0.02, math.inf).fair_strike_heston(0.04, 0.0, 0.04) == 0.04
    assert math.isnan(sr.VarianceSwapPricer(100.0, 0.05, 0.02, 1.0).fair_strike_heston(0.04, math.nan, 0.04))
    deciding_now = sr.SimpleChooserPricer(s=100.0, k=100.0, r=0.03, q=0.03, sigma=0.25, t1=0.0, t=1.0)
    assert deciding_now.price() == pytest.approx(9.653647626602911, **LIBM)
    assert sr.DoubleBarrierPricer(s=0.0, t=0.0, **CORRIDOR).price() == 0.0
    expired_put = sr.FixedLookbackPricer(s=0.0, k=100.0, r=0.05, q=0.02, sigma=0.25, t=0.0, option_type="put")
    assert expired_put.price() == 100.0
    chooser = dict(s=0.0, k=100.0, r=0.05, q=0.02, sigma=0.25, t1=0.0)
    assert sr.SimpleChooserPricer(t=0.0, **chooser).price() == 100.0
    assert math.isnan(sr.SimpleChooserPricer(t=1.0, **chooser).price())
    compound = dict(s=0.0, k1=5.0, k2=100.0, t1=0.0, r=0.05, q=0.02, sigma=0.25, compound_type="poc")
    assert sr.CompoundPricer(t2=0.0, **compound).price() == 5.0
    assert math.isnan(sr.CompoundPricer(t2=1.0, **compound).price())


def test_the_mc_barrier_pricer_sees_q_and_replays_its_seed():
    seeded = sr.MCBarrierPricer(n_paths=4_000, n_steps=64, seed=11)
    args = (100.0, 100.0, 130.0, 0.03, 0.2, 1.0, "up_out")
    first, second = seeded.price(*args), seeded.price(*args)
    assert (first.mean, first.std_err, first.n_samples) == (second.mean, second.std_err, 4_000)
    assert first.mean == sr.MCBarrierPricer(n_paths=4_000, n_steps=64, seed=11).price(*args).mean
    assert first.mean != sr.MCBarrierPricer(n_paths=4_000, n_steps=64, seed=12).price(*args).mean
    shifted = 130.0 * math.exp(0.5826 * 0.2 * math.sqrt(1.0 / 64))  # Broadie, Glasserman & Kou (1997)
    for q in (0.0, 0.04):
        estimate = seeded.price(*args, q=q)
        closed = sr.BarrierPricer(s=100.0, k=100.0, h=shifted, r=0.03, q=q, sigma=0.2, t=1.0, barrier_type="up_out").price()
        assert abs(estimate.mean - closed) < 4.0 * estimate.std_err + 0.1
    assert seeded.price(*args, q=0.04).mean < first.mean
    assert sr.MCBarrierPricer(n_paths=500, n_steps=16).price(*args).n_samples == 500
    with pytest.raises(ValueError, match="n_paths must satisfy `n_paths >= 1`"):
        sr.MCBarrierPricer(n_paths=0)

"""Quant-surface pytest coverage.

Exercises the analytic Black-Scholes-Merton pricer binding: ATM price
range, monotonicity in strike, and the no-arbitrage call price bounds.
"""

from __future__ import annotations

import math

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


def test_bsm_pricer_prices_with_its_dividend_yield():
    # Merton (1973) evaluated at 50 digits with mpmath; QuantLib's analytic engine agrees.
    merton_call = 10.197535275462173
    pricer = sr.BSMPricer(s=100.0, v=0.25, k=100.0, r=0.03, tau=1.0, q=0.02)
    call, put = pricer.call_put()
    assert abs(pricer.price() - merton_call) < 1e-10
    assert abs(pricer.implied_volatility(merton_call, "call") - 0.25) < 1e-10
    assert abs(call - put - (100.0 * math.exp(-0.02) - 100.0 * math.exp(-0.03))) < 1e-10


def test_an_expired_bsm_option_is_worth_its_payoff_and_has_no_greek():
    pricer = sr.BSMPricer(s=110.0, v=0.2, k=100.0, r=0.05, tau=0.0)
    assert pricer.price() == 10.0
    assert sr.BSMPricer(s=100.0, v=0.2, k=100.0, r=0.05, tau=0.0).price() == 0.0
    assert math.isnan(pricer.delta()) and math.isnan(pricer.vega()) and math.isnan(pricer.rho())


def test_discount_curve_rates_are_nan_outside_their_domain():
    import numpy as np

    curve = sr.DiscountCurve.from_zero_rates(np.array([1.0, 2.0, 5.0]), np.full(3, 0.03), interp="log_df")
    assert math.isnan(curve.forward_rate(2.0, 1.0)) and math.isnan(curve.forward_rate(1.0, 1.0))
    assert abs(curve.forward_rate(1.0, 2.0) - 0.03) < 1e-12
    assert math.isnan(curve.par_rate(0.0, 2))


def test_a_bad_curve_quote_or_merton_parameter_raises_value_error():
    import numpy as np

    with pytest.raises(ValueError, match="maturity must satisfy"):
        sr.DiscountCurve.bootstrap_ois([[1.0], [1.0, float("nan")]], [0.02, 0.02], interp="log_df")
    ois = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 2.0]), np.full(3, 0.02), interp="log_df")
    with pytest.raises(ValueError, match="maturity must satisfy"):
        sr.DiscountCurve.bootstrap_forecast(ois, [(float("nan"), 0.03)], [], [], interp="log_df")
    multi = sr.MultiCurve(ois)
    multi.add_forecast("3M", ois)
    assert multi.fair_swap_rate("3M", [1.0]) is None
    with pytest.raises(ValueError, match="gamma must satisfy"):
        sr.Merton1976Pricer(100.0, 0.2, 100.0, 0.05, 0.5, 1.5, 1.0)


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

def test_callable_bond_tree_reads_each_of_its_arguments():
    tree = sr.HullWhiteCallableBond(initial_rate=0.04, mean_reversion=0.3, theta=0.05, sigma=0.012, horizon=3.0, steps=36)
    assert tree.zero_coupon_bond_price() == pytest.approx(0.8784165416032598, rel=1e-12)
    priced = tree.price(100.0, 0.05, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], calls=[(2.0, 101.0)], puts=[(1.0, 99.0)])
    assert priced == pytest.approx((101.64174381552358, 101.77794153378728, 0.2028463240456233, 0.06617004095160439), rel=1e-9)
    positional = sr.HullWhiteCallableBond(0.02, 0.1, 0.03, 0.008, 2.0, 24)
    assert positional.zero_coupon_bond_price() == pytest.approx(0.9591305136608149, rel=1e-12)

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
    hw = sr.HullWhiteSwaptionCalibrator(quotes, curve, initial_rate=0.03, theta=0.03).calibrate()
    assert hw[1] > 0.0 and math.isfinite(hw[2])
    g2 = sr.G2ppSwaptionCalibrator(quotes, curve, initial_rate=0.03, max_iters=60).calibrate()
    assert len(g2) == 7 and abs(g2[4]) < 1.0 and math.isfinite(g2[5])
    assert "max_iters=None" in sr.G2ppSwaptionCalibrator.__text_signature__
    sr.G2ppSwaptionCalibrator(quotes, curve, initial_rate=0.03, max_iters=None)

def test_essvi_surface_calibrates_and_interpolates():
    import numpy as np

    ks = np.linspace(-0.5, 0.5, 21)
    maturities = [0.25, 0.5, 1.0, 2.0]
    ssvi = sr.SsviParams(rho=-0.4, eta=0.6, gamma=0.4)
    slices = [(ks, np.array([ssvi.total_variance(k, 0.04 * t) for k in ks]), 0.04 * t) for t in maturities]
    surface = sr.EssviSurface.calibrate(maturities, slices)
    params = surface.slices()
    assert len(params) == 4 and all(abs(p[2] + 0.4) < 2e-3 for p in params)
    assert surface.is_butterfly_free() and surface.is_calendar_spread_free()
    assert abs(surface.total_variance(0.2, 1.0) - ssvi.total_variance(0.2, 0.04)) < 2e-6
    assert surface.total_variance(0.1, 0.5) < surface.total_variance(0.1, 0.75) < surface.total_variance(0.1, 1.5)
    assert abs(surface.implied_vol(0.0, 1.0) - math.sqrt(surface.total_variance(0.0, 1.0))) < 1e-12

def test_sabr_regularization_pulls_nu_to_the_anchor():
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    prices = [sr.SabrPricer(s=100.0, alpha=0.2, beta=1.0, nu=0.6, rho=-0.3, k=k, r=0.01, tau=1.0).price() for k in strikes]
    plain = sr.SabrCalibrator(strikes, prices, s=100.0, r=0.01, tau=1.0).calibrate()
    pulled = sr.SabrCalibrator(
        strikes, prices, s=100.0, r=0.01, tau=1.0, regularization=([0.2, 0.9, -0.3], [0.0, 1e4, 0.0])
    ).calibrate()
    assert abs(plain[2] - 0.6) < 0.05 and abs(pulled[2] - 0.9) < 0.05

def _hagan_lognormal_vols(alpha, nu, rho, forward, tau, strikes):
    vols = []
    for k in strikes:
        z = nu / alpha * math.log(forward / k)
        x = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)) if abs(z) > 1e-12 else 1.0
        ratio = z / x if abs(z) > 1e-12 else 1.0
        vols.append(alpha * ratio * (1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0) * tau))
    return vols


def test_sabr_caplet_calibrator_default_fit_recovers_nu():
    forward, tau, strikes = 0.03, 2.0, [0.02, 0.025, 0.03, 0.035, 0.04]
    vols = _hagan_lognormal_vols(0.25, 0.5, -0.3, forward, tau, strikes)
    fit = sr.SabrCapletCalibrator(forward, tau, 1.0, strikes, vols).calibrate()
    alpha, beta, nu, rho, rmse, converged = fit
    assert beta == 1.0 and converged and rmse < 1e-4
    assert abs(nu - 0.5) < 0.05 and abs(rho + 0.3) < 0.1


def test_sabr_caplet_calibrator_rejects_a_beta_outside_the_unit_interval():
    with pytest.raises(ValueError, match="beta must satisfy"):
        sr.SabrCapletCalibrator(0.03, 2.0, 1.5, [0.02, 0.03, 0.04], [0.3, 0.25, 0.3])


def test_sabr_calibrator_rejects_a_wrong_anchor_count():
    strikes = [90.0, 100.0, 110.0]
    prices = [sr.SabrPricer(s=100.0, alpha=0.2, beta=1.0, nu=0.6, rho=-0.3, k=k, r=0.01, tau=1.0).price() for k in strikes]
    calibrator = sr.SabrCalibrator(strikes, prices, s=100.0, r=0.01, tau=1.0, regularization=([0.2, 0.6], [1.0, 1.0]))
    with pytest.raises(ValueError, match="regularization anchors"):
        calibrator.calibrate()


def test_heston_adi_pricer_matches_the_analytic_price():
    kwargs = dict(s=100.0, k=100.0, r=0.025, tau=1.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.9, q=0.0)
    pde = sr.HestonAdiPricer(m1=100, m2=50, steps=50, scheme="mcs", **kwargs)
    call, put = pde.call_put()
    analytic = sr.HestonPricer(**kwargs).price()
    assert abs(call - analytic) / analytic < 1e-2
    assert abs(call - put - (100.0 - 100.0 * math.exp(-0.025))) < 1e-9
    knocked = sr.HestonAdiPricer(barrier=90.0, **kwargs).price()
    assert 0.0 < knocked < call

def test_xva_exposure_profile_and_adjustments():
    import numpy as np

    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, 0.03), interp="log_df")
    rng = np.random.default_rng(1)
    mtm = 10.0 * np.cumsum(rng.standard_normal((20000, 4)) * np.sqrt(0.5), axis=1)
    profile = sr.ExposureProfile.from_mtm(mtm, [0.5, 1.0, 1.5, 2.0], quantile=0.95)
    for t, epe in zip(profile.times(), profile.epe()):
        assert abs(epe - 10.0 * math.sqrt(t) / math.sqrt(2.0 * math.pi)) < 0.5
    assert profile.cva(0.02, curve, 0.6) > 0.0 and profile.cva(0.0, curve, 0.6) == 0.0
    assert abs(profile.fva(curve, 0.01)) < 0.2
    swap = sr.HullWhiteSwapExposure(0.1, 0.01, 1e6, [1.0, 2.0, 3.0, 4.0, 5.0])
    prof = swap.profile(curve, paths=2000, seed=7)
    assert prof.epe()[-1] == 0.0 and prof.peak_epe() > 0.0
    assert prof.cva(0.02, curve, 0.6) > prof.cva(0.01, curve, 0.6) > 0.0

def test_cds_index_and_cdo_tranche():
    import numpy as np

    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, 0.03), interp="log_df")
    names = [(1 / 125, 0.4, 0.02)] * 125
    index = sr.CdsIndex(names, 0.01, 1e7, "2026-03-20", "2031-06-20")
    fair = index.fair_spread("2026-03-20", curve)
    assert 0.005 < fair < 0.03
    assert abs(index.isda_upfront("2026-03-20", curve, 0.01)) < 1.0
    assert index.isda_upfront("2026-03-20", curve, 0.012) > 0.0 > index.isda_upfront("2026-03-20", curve, 0.008)
    equity = sr.CdoTranche(names, 0.0, 0.03, 0.05, [1.0, 2.0, 3.0, 4.0, 5.0], 0.3)
    senior = sr.CdoTranche(names, 0.12, 0.22, 0.0, [1.0, 2.0, 3.0, 4.0, 5.0], 0.3)
    assert equity.valuation(curve)[3] > senior.valuation(curve)[3] > 0.0
    dist = equity.loss_distribution(5.0)
    assert abs(sum(dist) - 1.0) < 1e-9 and dist[0] > 0.0


def test_survival_curve_is_accepted_wherever_a_flat_hazard_is():
    import numpy as np

    flat = sr.SurvivalCurve.flat(0.02)
    assert abs(flat.survival_probability(5.0) - math.exp(-0.1)) < 1e-12
    assert abs(flat.average_hazard(7.0) - 0.02) < 1e-12
    curve = sr.SurvivalCurve([1.0, 3.0, 5.0], [0.01, 0.02, 0.03])
    assert abs(curve.survival_probability(1.0) - math.exp(-0.01)) < 1e-12
    assert abs(curve.survival_probability(3.0) - math.exp(-0.01 - 0.04)) < 1e-12
    assert abs(curve.forward_hazard(3.0, 5.0) - 0.03) < 1e-12
    assert len(curve) == 4 and curve.times() == [0.0, 1.0, 3.0, 5.0] and curve.pillars()[0] == 1.0
    assert curve.survival_probabilities([1.0, 3.0]) == [curve.survival_probability(1.0), curve.survival_probability(3.0)]
    via_probs = sr.SurvivalCurve.from_survival_probs(curve.times(), curve.pillars())
    assert abs(via_probs.default_probability(4.0) - curve.default_probability(4.0)) < 1e-12
    discount = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, 0.03), interp="log_df")
    rng = np.random.default_rng(1)
    mtm = 10.0 * np.cumsum(rng.standard_normal((5000, 4)) * np.sqrt(0.5), axis=1)
    profile = sr.ExposureProfile.from_mtm(mtm, [0.5, 1.0, 1.5, 2.0], quantile=0.95)
    assert profile.cva(flat, discount, 0.6) == profile.cva(0.02, discount, 0.6)
    assert profile.bilateral_cva(curve, 0.01, discount, 0.6) > 0.0
    names = [(1 / 10, 0.4, 0.02)] * 5 + [(1 / 10, 0.4, flat)] * 5
    index = sr.CdsIndex(names, 0.01, 1e7, "2026-03-20", "2031-06-20")
    same = sr.CdsIndex([(1 / 10, 0.4, 0.02)] * 10, 0.01, 1e7, "2026-03-20", "2031-06-20")
    assert index.fair_spread("2026-03-20", discount) == same.fair_spread("2026-03-20", discount)
    tranche = sr.CdoTranche([(0.5, 0.4, curve), (0.5, 0.4, 0.02)], 0.0, 0.03, 0.05, [1.0, 2.0, 3.0], 0.3)
    assert tranche.valuation(discount)[3] > 0.0
    with pytest.raises(ValueError):
        sr.SurvivalCurve([1.0, 0.5], [0.01, 0.02])
    with pytest.raises(ValueError):
        profile.cva("0.02", discount, 0.6)


def test_empirical_cvar_rejects_an_empty_sample():
    import numpy as np

    with pytest.raises(ValueError, match="empirical CVaR: needs at least 1, got 0"):
        sr.empirical_cvar(np.array([], dtype=np.float64), 0.05)


def test_expected_shortfall_and_empirical_cvar_are_the_tail_mean():
    import numpy as np

    assert math.isclose(sr.ExpectedShortfall(-np.arange(1.0, 11.0)).value, 10.0, rel_tol=1e-15)
    returns = np.array([0.0] * 28 + [-10.0, -2.0])
    assert math.isclose(sr.empirical_cvar(returns, 0.05), 22.0 / 3.0, rel_tol=1e-15)


def test_the_risk_classes_raise_value_error_on_short_or_non_finite_data():
    import numpy as np

    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="historical VaR: needs at least 1, got 0"):
        sr.VaR(empty)
    with pytest.raises(ValueError, match="sample 1 is NaN"):
        sr.VaR(np.array([0.01, np.nan, -0.02]), method="gaussian")
    with pytest.raises(ValueError, match="drawdown statistics: needs at least 1, got 0"):
        sr.DrawdownStats(empty)
    with pytest.raises(ValueError, match="sample 0 is -10.0, not positive"):
        sr.DrawdownStats(np.array([-10.0, -20.0]))


def test_implied_vol_surface_rejects_an_unsorted_strike_grid():
    model = sr.HestonFourier(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.6)
    with pytest.raises(ValueError, match="strikes must satisfy"):
        sr.ImpliedVolSurface.from_heston(model, 100.0, 0.01, 0.0, [110.0, 100.0], [1.0])


def test_carr_madan_pricer_takes_the_carry_from_the_query():
    model = sr.HestonFourier(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    pricer = sr.CarrMadanPricer()
    for k in (90.0, 100.0, 110.0):
        assert pricer.strike_in_grid_heston(model, 100.0, k, 0.05, 0.02, 1.0)
        fft = pricer.price_heston_call(model, 100.0, k, 0.05, 0.02, 1.0)
        assert abs(fft - model.price_call(100.0, k, 0.05, 0.02, 1.0)) < 5e-3


def test_the_fourier_models_and_carr_madan_pay_out_at_expiry_and_are_nan_past_it():
    heston = sr.HestonFourier(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    bates = sr.BatesFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, -0.05, 0.1)
    kou = sr.KouFourier(0.2, 1.0, 0.4, 10.0, 5.0)
    pricer = sr.CarrMadanPricer()
    calls = [model.price_call for model in (sr.BSMFourier(0.2), heston, bates, kou)]
    calls.append(lambda *query: pricer.price_heston_call(heston, *query))
    calls.append(lambda *query: pricer.price_bates_call(bates, *query))
    calls.append(lambda *query: pricer.price_kou_call(kou, *query))
    for call in calls:
        for s in (90.0, 100.0, 110.0):
            assert call(s, 100.0, 0.05, 0.02, 0.0) == max(s - 100.0, 0.0)
            for tau in (-1.0, math.nan, math.inf):
                assert math.isnan(call(s, 100.0, 0.05, 0.02, tau))
        for s, k in ((0.0, 100.0), (-5.0, 100.0), (110.0, 0.0), (110.0, -5.0)):
            assert math.isnan(call(s, k, 0.05, 0.02, 1.0))


def test_heston_slv_calibrates_a_heston_surface_and_prices():
    import numpy as np

    s, r, q = 100.0, 0.02, 0.005
    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.05, 0.4, -0.6
    strikes = np.linspace(70.0, 140.0, 36)
    maturities = [0.25, 0.5, 0.75, 1.0]
    model = sr.HestonFourier(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    calls = np.array([[model.price_call(s, k, r, q, t) for k in strikes] for t in maturities])
    calibrator = sr.HestonSlvCalibrator(
        s, r, q, list(strikes), maturities, calls, eta=0.7,
        heston=(v0, kappa, theta, sigma, rho), n_particles=5_000, steps_per_year=50, seed=11,
    )
    result = calibrator.calibrate()
    assert result.converged and result.heston() is None
    assert result.params() == (kappa, theta, sigma, rho, v0, 0.7)
    assert result.rates == (r, q)
    assert math.isfinite(result.rmse) and result.rmse < 1.0
    surface = result.leverage()
    assert surface.values.shape == (len(surface.times), len(surface.spots))
    assert surface.covers(100.0, 0.5) and not surface.covers(100.0, 1.5)
    assert 0.5 < surface.interpolate(100.0, 0.5) < 1.5
    pricer = result.to_model(n_paths=4_000, steps_per_year=50, seed=7)
    assert pricer.calibration_rates == (r, q)
    atm = pricer.price_call(s, 100.0, r, q, 0.5)
    assert abs(atm - model.price_call(s, 100.0, r, q, 0.5)) < 1.5
    estimate = pricer.price_call_estimate(s, 100.0, r, q, 0.5)
    assert estimate.mean == atm and estimate.std_err > 0.0
    assert math.isnan(pricer.price_call(1000.0, 1000.0, r, q, 0.5))
    with pytest.raises(ValueError):
        pricer.price_call(s, 100.0, 0.05, q, 0.5)
    unanchored = sr.HestonSlvPricer(kappa, theta, sigma, rho, v0, 1.0, surface, n_paths=2_000, steps_per_year=25)
    assert unanchored.calibration_rates is None
    assert unanchored.price_call(s, 100.0, 0.05, 0.0, 0.5) > 0.0
    with pytest.raises(ValueError):
        sr.HestonSlvPricer(kappa, theta, sigma, rho, v0, 1.0, surface, r=r)
    with pytest.raises(ValueError):
        sr.HestonSlvPricer(math.nan, theta, sigma, rho, v0, 1.0, surface)
    with pytest.raises(ValueError):
        sr.LeverageSurface(np.array([80.0, math.nan]), np.array([0.5]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        sr.LeverageSurface(np.array([80.0, 120.0]), np.array([0.5]), np.array([[1.0, math.nan]]))
    with pytest.raises(ValueError):
        sr.HestonSlvCalibrator(s, r, q, list(strikes), maturities, calls, eta=2.0).calibrate()
    process = sr.PyHestonSlv(kappa, theta, sigma, rho, r - q, 0.7, surface, 65, s0=s, v0=v0, t=1.0, seed=5)
    paths, _ = process.sample_par(16)
    assert paths.shape == (16, 65)
    pde = sr.HestonSlvCalibrator(
        s, r, q, list(strikes), maturities, calls, eta=0.7, heston=(v0, kappa, theta, sigma, rho),
        method="fokker_planck", log_spot_nodes=101, variance_nodes=50, steps_per_year=50,
    ).calibrate()
    assert pde.converged and pde.rmse < 1.0
    assert pde.leverage().covers(100.0, 0.5)
    assert 0.5 < pde.leverage().interpolate(100.0, 0.5) < 1.5
    with pytest.raises(ValueError):
        sr.HestonSlvCalibrator(s, r, q, list(strikes), maturities, calls, method="galerkin")


def test_heston_slv_starts_its_heston_fit_from_the_initial_guess():
    import numpy as np

    s, r, q = 100.0, 0.02, 0.005
    model = sr.HestonFourier(v0=0.04, kappa=2.0, theta=0.05, sigma=0.4, rho=-0.6)
    strikes = list(np.linspace(80.0, 120.0, 9))
    maturities = [0.25, 0.5, 1.0]
    calls = np.array([[model.price_call(s, k, r, q, t) for k in strikes] for t in maturities])
    result = sr.HestonSlvCalibrator(
        s, r, q, strikes, maturities, calls,
        heston_initial_guess=(0.05, 1.5, 0.04, 0.5, -0.5), n_particles=2_000, steps_per_year=50, seed=3,
    ).calibrate()
    heston = result.heston()
    assert heston is not None and abs(heston[0] - 0.04) < 0.02, heston
    assert result.rates == (r, q) and math.isfinite(result.max_error)
    outside = sr.HestonSlvCalibrator(s, r, q, strikes, maturities, calls, heston_initial_guess=(0.05, 1.5, 0.04, 0.5, 1.5))
    with pytest.raises(ValueError, match="invalid start: rho must satisfy"):
        outside.calibrate()


def test_hscm_reports_an_evaluation_cap_as_no_convergence():
    quotes = ((95.0, 0.25, 8.0), (100.0, 0.5, 7.5), (105.0, 1.0, 6.0))
    options = [sr.HscmMarketOption(k, t, price, 0.03) for k, t, price in quotes]
    *_, rmse, converged = sr.HscmCalibrator(100.0, options, max_iter=3).calibrate()
    assert not converged and math.isfinite(rmse)


BELOW_FELLER = (0.04, 1.5, 0.04, 0.5, -0.7)
HESTON_STRIKES = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0]
HESTON_MATURITIES = (0.25, 0.5, 1.0, 2.0, 3.0)


def _heston_surface(params, s0=100.0, r=0.02, q=0.0):
    model = sr.HestonFourier(*params)
    return [
        sr.MarketSlice(
            HESTON_STRIKES,
            [model.price_call(s0, k, r, q, t) for k in HESTON_STRIKES],
            [True] * len(HESTON_STRIKES),
            t,
        )
        for t in HESTON_MATURITIES
    ]


def test_heston_calibrator_reports_the_fitted_pricer_rmse_and_stop_reason():
    calibrator = sr.HestonCalibrator(_heston_surface(BELOW_FELLER), 100.0, 0.02, "call", 0.0)
    assert len(calibrator.calibrate()) == 7
    *fit, converged, rmse, message = calibrator.calibrate(
        initial_guess=BELOW_FELLER, full_output=True
    )
    assert converged and message == "converged"
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
    assert sigma * sigma <= 2.0 * kappa * theta
    for bad in ({"jacobian_method": "newton"}, {"residual_weights": [1.0]}):
        with pytest.raises(ValueError):
            sr.HestonCalibrator(slices, 100.0, 0.02, "call", 0.0, **bad)


def _merton_call(s, k, r, q, sigma, tau):
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    return s * math.exp(-q * tau) * cdf(d1) - k * math.exp(-r * tau) * cdf(d2)


def test_bsm_calibrator_prices_with_its_dividend_yield_and_a_ragged_slice_is_an_error():
    strikes, tau = [90.0, 100.0, 110.0], 1.0
    prices = [_merton_call(100.0, k, 0.03, 0.02, 0.25, tau) for k in strikes]
    slices = [sr.MarketSlice(strikes, prices, [True] * 3, tau)]
    sigma, converged, rmse = sr.BSMCalibrator(slices, 100.0, 0.03, q=0.02).calibrate()
    assert converged and abs(sigma - 0.25) < 1e-8 and rmse < 1e-8
    with pytest.raises(ValueError, match="equal length"):
        sr.MarketSlice(strikes, prices[:2], [True] * 3, tau)


CALL_PUT_QUERY = dict(s=100.0, k=105.0, r=0.05, tau=0.75, q=0.02)
HESTON_MODEL = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)


@pytest.mark.parametrize(
    "pricer, pair",
    [
        (lambda: sr.BSMPricer(v=0.25, **CALL_PUT_QUERY), (7.356284457997674, 9.980504358377694)),
        (lambda: sr.BachelierPricer(v=20.0, **CALL_PUT_QUERY), (5.425620481356887, 8.049840381736892)),
        (lambda: sr.AsianPricer(v=0.25, **CALL_PUT_QUERY), (3.2776906533992545, 7.067347664422634)),
        (lambda: sr.BjerksundStensland2002Pricer(v=0.25, **CALL_PUT_QUERY), (7.35628449811837, 10.292928328280922)),
        (
            lambda: sr.QuantoPricer(v=0.2, r_f=0.05, v_fx=0.12, rho=0.3, fixed_rate=1.5, **CALL_PUT_QUERY),
            (8.111646452569918, 12.84376640716852),
        ),
        (lambda: sr.HestonPricer(**HESTON_MODEL, **CALL_PUT_QUERY), (5.164008255606987, 7.788228155987014)),
        (lambda: sr.HestonPricer(lambda_=0.1, **HESTON_MODEL, **CALL_PUT_QUERY), (5.081933443027005, 7.706153343407038)),
        (
            lambda: sr.SabrPricer(alpha=0.2, beta=1.0, nu=0.4, rho=-0.3, **CALL_PUT_QUERY),
            (5.633121359954366, 8.257341260334393),
        ),
        (
            lambda: sr.Merton1976Pricer(v=0.25, lambda_=1.0, gamma=0.4, **{**CALL_PUT_QUERY, "q": None}),
            (7.905150880530156, 9.040564741216432),
        ),
    ],
)
def test_call_put_returns_each_pricers_pinned_pair(pricer, pair):
    assert pricer().call_put() == pytest.approx(pair, rel=1e-12)


def test_merton_pricer_prices_with_its_dividend_yield():
    query = dict(s=100.0, v=0.25, k=105.0, r=0.05, lambda_=1.0, gamma=0.4, tau=0.75)
    with_yield = sr.Merton1976Pricer(q=0.02, **query)
    assert with_yield.call_put() == pytest.approx((7.15321711897312, 9.777437019353142), rel=1e-12)
    assert with_yield.price() == with_yield.call_put()[0]
    assert sr.Merton1976Pricer(q=0.02, option_type="put", **query).price() == with_yield.call_put()[1]


def test_a_degenerate_sabr_vol_and_an_expired_option_keep_their_pairs():
    degenerate = sr.SabrPricer(s=100.0, k=105.0, r=0.05, alpha=0.2, beta=1.0, nu=3.0, rho=-0.9, tau=10.0)
    assert all(math.isnan(leg) for leg in degenerate.call_put())
    expired = dict(s=100.0, k=105.0, r=0.05, tau=0.0)
    assert sr.BSMPricer(v=0.25, **expired).call_put() == (0.0, 5.0)
    assert sr.HestonPricer(**HESTON_MODEL, **expired).call_put() == (0.0, 5.0)
    assert sr.Merton1976Pricer(v=0.25, lambda_=1.0, gamma=0.4, **{**expired, "k": 95.0}).call_put() == (5.0, 0.0)


def test_the_jump_fourier_models_keep_their_keywords_and_prices():
    # The pins are Lewis's integral by mpmath at 40 digits.
    bates = dict(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7, mu_j=-0.05, sigma_j=0.1)
    kw = sr.BatesFourier(**{"lambda": 0.5}, **bates)
    assert kw.price_call(100.0, 105.0, 0.05, 0.02, 0.75) == pytest.approx(5.6848297048328975, rel=1e-12)
    positional = sr.BatesFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, -0.05, 0.1)
    assert positional.price_call(100.0, 95.0, 0.05, 0.02, 0.5) == pytest.approx(9.64659612731218, rel=1e-12)
    hkde = dict(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7, lam=0.5, p_up=0.4, eta1=10.0, eta2=5.0)
    assert sr.HKDEFourier(**hkde).price_call(100.0, 105.0, 0.05, 0.02, 0.75) == pytest.approx(6.559356772752303, rel=1e-12)
    positional = sr.HKDEFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, 0.4, 10.0, 5.0)
    assert positional.price_call(100.0, 95.0, 0.05, 0.02, 0.5) == pytest.approx(10.316255704617463, rel=1e-12)
    with pytest.raises(ValueError, match=r"lambda must satisfy `lambda\.is_finite\(\) && lambda >= 0\.0`, got lambda = -0\.5"):
        sr.HKDEFourier(**{**hkde, "lam": -0.5})
    with pytest.raises(ValueError, match=r"sigma must satisfy `sigma\.is_finite\(\) && sigma > 0\.0`, got sigma = -0\.3"):
        sr.HKDEFourier(**{**hkde, "sigma_v": -0.3})


def test_the_svi_and_ssvi_calibrators_carry_their_fits_stop():
    truth = sr.SviRawParams(0.02, 0.1, -0.4, 0.05, 0.2)
    ks = [-0.5 + 0.05 * i for i in range(21)]
    fit = sr.SviCalibrator(ks, [truth.total_variance(k) for k in ks])
    assert fit.converged() and fit.message() == "converged" and fit.rmse() < 1e-9
    # eta = 2.5 breaks the calendar bound over these thetas, so the projection caps it.
    capped = sr.SsviParams(-0.3, 2.5, 0.5)
    grid = [-0.3 + 0.05 * i for i in range(13)]
    slices = [
        (grid, [capped.total_variance(k, theta) for k in grid], theta)
        for theta in (0.02, 0.04, 0.08)
    ]
    projected = sr.SsviCalibrator(slices)
    assert not projected.converged() and projected.rmse() > 1e-3
    assert projected.message() == "stopped at the bound of eta"

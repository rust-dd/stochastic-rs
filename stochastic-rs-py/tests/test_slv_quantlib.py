"""Cross-check of the Heston SLV leverage calibration against QuantLib.

QuantLib is not a dependency of the suite: the module is skipped unless it
is installed (`pip install QuantLib`). QuantLib's `HestonSLVFDMModel` solves
the same forward Kolmogorov equation by its own finite-difference scheme,
and its `NoExceptLocalVolSurface` reads the Dupire local volatility off the
analytic Heston Black surface. That local volatility is handed to both of
our routes, so what is compared is the leverage calibration alone; the
bounds are the level at which QuantLib's own FDM and MC models agree with
each other (mean 0.01, worst 0.025 on the same probes), and both of our
routes sit well inside them (mean 0.001-0.006, worst 0.003-0.012).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import stochastic_rs as sr

ql = pytest.importorskip("QuantLib")

S0, R, Q = 100.0, 0.02, 0.005
V0, KAPPA, THETA, SIGMA, RHO = 0.04, 2.0, 0.05, 0.4, -0.6
STRIKES = np.linspace(60.0, 150.0, 91)
MATURITIES = [0.1 * k for k in range(1, 11)]
PROBES = [(t, s) for t in (0.25, 0.5, 0.75, 1.0) for s in (80.0, 90.0, 100.0, 110.0, 120.0)]


def _quantlib_setup():
    today = ql.Date(22, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, R, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, Q, dc))
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    model = ql.HestonModel(ql.HestonProcess(r_ts, q_ts, spot, V0, KAPPA, THETA, SIGMA, RHO))
    black = ql.HestonBlackVolSurface(ql.HestonModelHandle(model))
    black.enableExtrapolation()
    local_vol = ql.NoExceptLocalVolSurface(ql.BlackVolTermStructureHandle(black), r_ts, q_ts, spot, math.sqrt(V0))
    local_vol.enableExtrapolation()
    analytic = ql.AnalyticHestonEngine(model)

    def call(k, tau):
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, k),
            ql.EuropeanExercise(today + ql.Period(int(round(tau * 365)), ql.Days)),
        )
        option.setPricingEngine(analytic)
        return option.NPV()

    return today, model, local_vol, call


def _quantlib_leverage(today, model, local_vol, eta):
    params = ql.HestonSLVFokkerPlanckFdmParams(
        201, 101, 400, 30, 2.0, 2, 2, 0.1, 1e-4, 10000, 1e-8, 1e-5, 0.00001, 0.1, 0.1, 0.9, 1e-4,
        ql.FdmHestonGreensFct.Gaussian, ql.FdmSquareRootFwdOp.Log, ql.FdmSchemeDesc.ModifiedCraigSneyd(),
    )
    slv = ql.HestonSLVFDMModel(local_vol, model, today + ql.Period(1, ql.Years), params, False, [], eta)
    leverage = slv.leverageFunction()
    leverage.enableExtrapolation()
    return leverage


def _deviation(ours, theirs):
    diffs = [abs(ours.interpolate(s, t) - theirs.localVol(t, s)) for t, s in PROBES]
    return float(np.mean(diffs)), float(np.max(diffs))


@pytest.mark.parametrize("eta", [1.0, 0.5])
def test_both_leverage_routes_agree_with_quantlib_fdm(eta):
    today, model, local_vol, call = _quantlib_setup()
    calls = np.array([[call(k, t) for k in STRIKES] for t in MATURITIES])
    ql_local = np.array([[local_vol.localVol(t, k) for k in STRIKES] for t in MATURITIES])
    theirs = _quantlib_leverage(today, model, local_vol, eta)
    heston = (V0, KAPPA, THETA, SIGMA, RHO)
    for method, kwargs in [
        ("fokker_planck", dict(log_spot_nodes=201, variance_nodes=100, steps_per_year=200)),
        ("particle", dict(n_particles=100_000, steps_per_year=200, seed=7)),
    ]:
        result = sr.HestonSlvCalibrator(
            S0, R, Q, list(STRIKES), MATURITIES, calls, eta=eta, heston=heston, method=method,
            local_vol=ql_local, **kwargs,
        ).calibrate()
        assert result.converged
        assert result.rmse < 0.05, f"{method}: in-sample rmse {result.rmse}"
        mean, worst = _deviation(result.leverage(), theirs)
        assert mean < 0.01, f"{method}: mean |L - L_quantlib| = {mean}"
        assert worst < 0.025, f"{method}: worst |L - L_quantlib| = {worst}"
        if eta == 1.0:
            assert abs(result.leverage().interpolate(100.0, 0.5) - 1.0) < 0.02


def test_the_dupire_read_of_the_call_grid_stays_close_to_quantlib():
    today, model, local_vol, call = _quantlib_setup()
    calls = np.array([[call(k, t) for k in STRIKES] for t in MATURITIES])
    theirs = _quantlib_leverage(today, model, local_vol, 0.5)
    result = sr.HestonSlvCalibrator(
        S0, R, Q, list(STRIKES), MATURITIES, calls, eta=0.5, heston=(V0, KAPPA, THETA, SIGMA, RHO),
        method="fokker_planck", log_spot_nodes=201, variance_nodes=100, steps_per_year=200,
    ).calibrate()
    mean, worst = _deviation(result.leverage(), theirs)
    assert mean < 0.03, f"finite-difference Dupire: mean |L - L_quantlib| = {mean}"
    assert worst < 0.06, f"finite-difference Dupire: worst |L - L_quantlib| = {worst}"

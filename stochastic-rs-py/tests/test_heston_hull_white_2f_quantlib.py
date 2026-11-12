"""Cross-check of `PyHestonHullWhite2F` against QuantLib in the one-factor limit.

QuantLib is not a dependency of the suite: the module is skipped unless it
is installed (`pip install QuantLib`). With `sigma2 = 0` the hybrid is Heston
on a one-factor Hull-White rate, which QuantLib's
`FdHestonHullWhiteVanillaEngine` prices by solving the three-dimensional
pricing equation, so the reference is exact up to its grid and is not the
affine approximation of the equity-rate covariance. At these parameters that
approximation (`AnalyticH1HWEngine`) is 3.7e-4 and 9.6e-4 from the
finite-difference price at the two wing strikes, 0.6 to 0.7 Monte Carlo
standard errors here.

The grid error is measured, not assumed: at zero equity-rate correlation
`AnalyticHestonHullWhiteEngine` is exact, and the finite-difference price on
the same grid has to match it within `FD_ERROR`. On QuantLib 1.44 the grid
below is 8e-5 from the exact price there, and refining it twice at
`rho_xr = 0.5` moves the price by 2.7e-5 and then 9.3e-6.

The comparison also has to be able to fail: the simulated prices must
reject the reference with the equity-rate correlation removed and with its
sign flipped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import stochastic_rs as sr

ql = pytest.importorskip("QuantLib")

S0, FLAT_RATE = 1.0, 0.03
V0, KAPPA, THETA, SIGMA, RHO_XV = 0.04, 1.0, 0.04, 0.25, -0.6
A, SIGMA1, RHO_XR = 0.3, 0.02, 0.5
YEARS, STEPS_PER_YEAR, PATHS, SEED = 5, 12, 40_000, 2011
STRIKES = (0.8, 1.0, 1.3)
GRID = (50, 100, 40, 20)  # time, spot, variance and rate nodes
FD_ERROR = 1.5e-4
Z_MAX, Z_POWER = 3.0, 5.0


def _theta_r(t):
    """Drift target that fits the flat curve: `a f + sigma1^2 / (2 a) * (1 - exp(-2 a t))`."""
    return A * FLAT_RATE + SIGMA1 * SIGMA1 / (2.0 * A) * (1.0 - math.exp(-2.0 * A * t))


def _simulated_puts():
    rho = np.array(
        [
            [1.0, RHO_XV, RHO_XR, 0.0],
            [RHO_XV, 1.0, 0.0, 0.0],
            [RHO_XR, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    paths = sr.PyHestonHullWhite2F(
        KAPPA, THETA, SIGMA, _theta_r, A, SIGMA1, 0.0, 0.1, rho, YEARS * STEPS_PER_YEAR + 1,
        s0=S0, v0=V0, r0=FLAT_RATE, t=float(YEARS), seed=SEED, scheme="qe",
    ).sample_par(PATHS)
    spot, discount = paths[:, 0, -1], np.exp(-paths[:, 4, -1])
    out = []
    for strike in STRIKES:
        payoff = np.maximum(strike - spot, 0.0) * discount
        out.append((payoff.mean(), payoff.std(ddof=1) / math.sqrt(PATHS)))
    return out


def _quantlib_puts():
    """Finite-difference puts at `RHO_XR`, `0` and `-RHO_XR`, and the exact zero-correlation puts."""
    today = ql.Date(10, 11, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    maturity = today + ql.Period(365 * YEARS, ql.Days)
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, FLAT_RATE, dc, ql.Continuous))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    model = ql.HestonModel(ql.HestonProcess(r_ts, q_ts, spot, V0, KAPPA, THETA, SIGMA, RHO_XV))
    hw_process = ql.HullWhiteProcess(r_ts, A, SIGMA1)

    def puts(engine):
        out = []
        for strike in STRIKES:
            option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, strike), ql.EuropeanExercise(maturity))
            option.setPricingEngine(engine)
            out.append(option.NPV())
        return out

    def fd(rho_xr):
        return puts(ql.FdHestonHullWhiteVanillaEngine(model, hw_process, rho_xr, *GRID, 0, True))

    exact_uncorrelated = puts(ql.AnalyticHestonHullWhiteEngine(model, ql.HullWhite(r_ts, A, SIGMA1), 164))
    return fd(RHO_XR), fd(0.0), fd(-RHO_XR), exact_uncorrelated


def test_one_factor_limit_prices_quantlib_finite_difference_puts():
    reference, uncorrelated, flipped, exact_uncorrelated = _quantlib_puts()
    simulated = _simulated_puts()

    grid_error = max(abs(a - b) for a, b in zip(uncorrelated, exact_uncorrelated))
    assert grid_error < FD_ERROR, f"finite-difference grid error {grid_error:.2e} at zero correlation"

    def worst_z(prices):
        return max(abs(mean - price) / se for (mean, se), price in zip(simulated, prices))

    for strike, (mean, se), price in zip(STRIKES, simulated, reference):
        z = (mean - price) / se
        print(f"K = {strike}: {mean:.6f} +/- {se:.6f} against {price:.6f} (z = {z:+.2f})")
        assert abs(mean - price) < Z_MAX * se + FD_ERROR, f"K = {strike}: {z:+.2f} standard errors from QuantLib"

    assert worst_z(uncorrelated) > Z_POWER, "the reference without the equity-rate correlation is not rejected"
    assert worst_z(flipped) > Z_POWER, "the reference with the equity-rate correlation flipped is not rejected"

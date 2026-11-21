"""`PyHestonHullWhite2F` at `sigma2 = 0` against QuantLib's finite-difference Heston Hull-White
puts, their grid error measured on the exact zero-correlation engine; needs QuantLib."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stochastic_rs as sr

ql = pytest.importorskip("QuantLib")

S0, FLAT_RATE = 1.0, 0.03
V0, KAPPA, THETA, SIGMA, RHO_XV = 0.04, 1.0, 0.04, 0.25, -0.6
A, SIGMA1, RHO_XR = 0.3, 0.02, 0.5
YEARS, STEPS_PER_YEAR, PATHS, SEEDS = 5, 12, 40_000, (2011, 1647, 42)
STRIKES = (0.8, 1.0, 1.3)
GRID = (50, 100, 40, 20)  # time, spot, variance and rate nodes
FD_ERROR = 1.5e-4
Z_MAX, Z_POWER = 3.0, 5.0


def _theta_r(t):
    """Drift target that fits the flat curve: `a f + sigma1^2 / (2 a) * (1 - exp(-2 a t))`."""
    return A * FLAT_RATE + SIGMA1 * SIGMA1 / (2.0 * A) * (1.0 - math.exp(-2.0 * A * t))


def _simulated_puts(seed):
    rho = np.array(
        [
            [1.0, RHO_XV, RHO_XR, 0.0],
            [RHO_XV, 1.0, 0.0, 0.0],
            [RHO_XR, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    spot, _, _, _, integrated_rate = sr.PyHestonHullWhite2F(
        KAPPA, THETA, SIGMA, _theta_r, A, SIGMA1, 0.0, 0.1, rho, YEARS * STEPS_PER_YEAR + 1,
        s0=S0, v0=V0, r0=FLAT_RATE, t=float(YEARS), seed=seed, scheme="qe",
    ).sample_par(PATHS)
    spot, discount = spot[:, -1], np.exp(-integrated_rate[:, -1])
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
    grid_error = max(abs(a - b) for a, b in zip(uncorrelated, exact_uncorrelated))
    assert grid_error < FD_ERROR, f"finite-difference grid error {grid_error:.2e} at zero correlation"

    def worst_z(simulated, prices, slack=0.0):
        z = [max(abs(mean - price) - slack, 0.0) / se for (mean, se), price in zip(simulated, prices)]
        return float("nan") if any(math.isnan(x) for x in z) else max(z)

    runs = [_simulated_puts(seed) for seed in SEEDS]
    closest = [worst_z(simulated, reference, FD_ERROR) for simulated in runs]
    print("worst |z| per seed beyond the grid error:", [f"{z:.2f}" for z in closest])
    assert not any(math.isnan(z) for z in closest)
    assert min(closest) < Z_MAX, f"every seed has a put {min(closest):.2f} standard errors or more from QuantLib"
    for simulated in runs:
        assert worst_z(simulated, uncorrelated) > Z_POWER, "the reference without the equity-rate correlation is not rejected"
        assert worst_z(simulated, flipped) > Z_POWER, "the reference with the equity-rate correlation flipped is not rejected"

"""Stats-surface pytest coverage.

Exercises the Jarque-Bera normality test binding: statistic / p-value
ranges, and the qualitative behaviour on Gaussian vs heavy-tailed data.
"""

from __future__ import annotations

import numpy as np
import stochastic_rs as sr


def test_jarque_bera_ranges_on_normal():
    arr = np.random.default_rng(0).standard_normal(2000)
    jb = sr.JarqueBera(arr)
    assert jb.statistic >= 0.0
    assert 0.0 <= jb.p_value <= 1.0


def test_jarque_bera_does_not_reject_gaussian():
    arr = np.random.default_rng(1).standard_normal(5000)
    jb = sr.JarqueBera(arr)
    # Gaussian data: JB should not reject normality at the 1% level.
    assert jb.p_value > 0.01


def test_jarque_bera_rejects_heavy_tails():
    # Student-t(3) is heavy-tailed; JB should reject normality strongly.
    arr = np.random.default_rng(2).standard_t(3, size=5000)
    jb = sr.JarqueBera(arr)
    assert jb.statistic > 0.0
    assert jb.p_value < 0.05


def test_jarque_bera_statistic_finite():
    arr = np.random.default_rng(3).standard_normal(1000)
    jb = sr.JarqueBera(arr)
    assert np.isfinite(jb.statistic)
    assert np.isfinite(jb.p_value)


def test_lee_mykland_flags_a_planted_jump():
    returns = np.random.default_rng(4).standard_normal(3000) * 1e-3
    returns[1500] = 0.03
    window = sr.LeeMyklandJumpTest.recommended_window(78)
    assert window == 141
    test = sr.LeeMyklandJumpTest(returns, window, alpha=0.01)
    assert 1500 in test.jump_indices
    stats = test.statistics()
    assert stats.shape == (3000,)
    assert np.isnan(stats[: window - 1]).all()
    assert abs(stats[1500]) > test.threshold
    assert test.nobs == 3000 and test.window == window


def _cointegrated_pair(seed: int, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n))
    y = np.empty((n, 2))
    y[:, 0] = walk
    y[:, 1] = 0.7 * walk + 0.1 * rng.standard_normal(n)
    return y


def test_johansen_selects_rank_one_on_a_cointegrated_pair():
    test = sr.Johansen(_cointegrated_pair(5), lags=2)
    assert test.rank_trace == 1
    assert test.rank_max_eig == 1
    assert test.max_eig_statistics().shape == (2,)
    assert test.eigenvectors().shape == (2, 2)
    np.testing.assert_allclose(test.trace_critical_5pct(), [15.4943, 3.8415])
    np.testing.assert_allclose(test.max_eig_critical_5pct(), [14.2639, 3.8415])


def test_vecm_shapes_and_pi_factorisation():
    y = _cointegrated_pair(6)
    fit = sr.Vecm(y, lags=2, rank=1)
    assert fit.beta().shape == (2, 1)
    assert fit.alpha().shape == (2, 1)
    assert len(fit.gamma()) == 1 and fit.gamma()[0].shape == (2, 2)
    assert fit.residuals().shape == (fit.nobs, 2)
    np.testing.assert_allclose(fit.pi(), fit.alpha() @ fit.beta().T, atol=1e-12)
    ratio = fit.beta()[1, 0] / fit.beta()[0, 0]
    assert abs(ratio + 1.0 / 0.7) < 0.1


def test_garch_fit_recovers_persistence_below_one():
    rng = np.random.default_rng(7)
    n = 1500
    omega, alpha, beta = 0.05, 0.10, 0.85
    sigma2 = omega / (1.0 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        eps = np.sqrt(sigma2) * rng.standard_normal()
        r[t] = eps
        sigma2 = omega + alpha * eps * eps + beta * sigma2
    fit = sr.GarchFit(r, kind="garch", p=1, q=1, mean="zero")
    assert fit.converged
    assert fit.param_names() == ["omega", "alpha[1]", "beta[1]"]
    assert fit.params().shape == (3,)
    assert 0.0 < fit.persistence < 1.0
    assert abs(fit.alpha()[0] - alpha) < 3.0 * fit.robust_std_errors()[1]
    assert fit.conditional_variance().shape == (n,)
    assert fit.covariance().shape == (3, 3)
    assert fit.kind == "garch" and fit.mean == "zero"


def test_garch_fit_rejects_unknown_kind():
    r = np.random.default_rng(8).standard_normal(300)
    try:
        sr.GarchFit(r, kind="figarch")
    except ValueError as err:
        assert "kind must be one of" in str(err)
    else:
        raise AssertionError("expected a ValueError")


def test_evt_pipeline_on_a_pareto_tail():
    rng = np.random.default_rng(9)
    losses = (1.0 - rng.random(20000)) ** (-1.0 / 3.0)  # Pareto(alpha=3), xi = 1/3
    hill = sr.HillEstimator(losses, 500)
    assert abs(hill.xi - 1.0 / 3.0) < 3.0 * hill.std_error
    assert hill.k == 500 and hill.nobs == 20000
    pot = sr.PotFit(losses, 3.0)
    assert pot.converged and pot.n_exceedances >= 10
    var99 = pot.quantile(0.99)
    assert pot.expected_shortfall(0.99) > var99 > pot.threshold
    assert pot.std_errors().shape == (2,)
    maxima = sr.block_maxima(losses, 100)
    assert maxima.shape == (200,)
    gev = sr.GevFit(maxima)
    assert gev.converged and gev.std_errors().shape == (3,)
    assert gev.return_level(50.0) > gev.mu
    excess = sr.mean_excess(losses, np.array([1.0, 2.0, 1e9]))
    assert np.isfinite(excess[:2]).all() and np.isnan(excess[2])
    gpd = sr.GpdFit(losses[losses > 3.0] - 3.0)
    assert abs(gpd.xi - pot.xi) < 1e-12


def test_distribution_fits_run_and_rank():
    rng = np.random.default_rng(12)
    r = 0.3 * rng.standard_t(6, 3000) - 0.1 * np.abs(rng.standard_normal(3000))
    skt = sr.SkewTFit(r)
    assert skt.converged and skt.eta > 2.0 and abs(skt.lambda_) < 1.0
    assert skt.std_errors().shape == (4,) and skt.covariance().shape == (4, 4)
    jsu = sr.JohnsonSuFit(r)
    assert jsu.converged and jsu.delta > 0.0 and jsu.lambda_ > 0.0
    vg = sr.VarianceGammaFit(r)
    assert vg.converged and vg.sigma > 0.0 and vg.nu > 0.0
    assert np.isfinite([skt.aic, jsu.aic, vg.aic]).all()
    pwm = sr.GpdPwm(np.abs(r[np.abs(r) > 0.5]) - 0.5)
    assert np.isfinite(pwm.xi) and pwm.nobs >= 2

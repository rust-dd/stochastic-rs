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

"""Stochastic-process-surface pytest coverage.

Exercises the GBM process binding: seed determinism, path shape, the
parallel multi-path sampler, and basic sanity of the simulated levels.
"""

from __future__ import annotations

import numpy as np
import pytest
import stochastic_rs as sr


def test_gbm_seed_determinism():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample()
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample()
    assert np.allclose(a, b)


def test_gbm_path_length():
    p = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    assert p.shape[0] == 252


def test_gbm_starts_at_x0():
    p = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    assert abs(float(p[0]) - 100.0) < 1e-6


def test_gbm_strictly_positive():
    p = sr.PyGbm(0.03, 0.4, 500, x0=50.0, t=2.0, seed=8).sample()
    assert np.all(p > 0.0)


def test_gbm_distinct_seeds_differ():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=2).sample()
    assert not np.allclose(a, b)


def test_gbm_sample_par_determinism():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    assert np.allclose(a, b)
    assert a.shape == (8, 252)


def test_gbm_sample_par_shape():
    s = sr.PyGbm(0.05, 0.2, 128, x0=100.0, t=1.0, seed=3).sample_par(16)
    assert s.shape == (16, 128)


def test_gbm_paths_all_finite():
    s = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=4).sample_par(32)
    assert np.all(np.isfinite(s))


def _heston(**kwargs):
    return sr.PyHeston(1.5, 0.04, 0.5, -0.7, 0.02, 53, s0=100.0, v0=0.04, t=1.0, **kwargs)


def test_heston_qe_prices_the_atm_call_within_monte_carlo_error():
    s, v = _heston(seed=7, scheme="qe").sample_par(100_000)
    payoff = np.exp(-0.02) * np.maximum(s[:, -1] - 100.0, 0.0)
    exact = sr.HestonFourier(0.04, 1.5, 0.04, 0.5, -0.7, 0.02, 0.0).price_call(
        100.0, 100.0, 0.02, 0.0, 1.0
    )
    assert np.all(v >= 0.0)
    assert abs(payoff.mean() - exact) < 4.0 * payoff.std() / np.sqrt(payoff.size)


def test_heston_qe_is_seeded_and_validated():
    a = _heston(seed=3, scheme="qe").sample()
    b = _heston(seed=3, scheme="qe").sample()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], _heston(seed=3).sample()[0])
    for bad in ({"scheme": "milstein"}, {"scheme": "qe", "pow": "3/2"}):
        with pytest.raises(ValueError):
            _heston(**bad)
    with pytest.raises(ValueError):
        sr.PyHeston(0.0, 0.04, 0.5, -0.7, 0.02, 53, s0=100.0, v0=0.04, t=1.0, scheme="qe")

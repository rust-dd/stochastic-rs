"""Copula-surface pytest coverage.

Exercises the bivariate Clayton copula binding: theta-from-tau,
seed-deterministic sampling, sample shape, and that the samples are valid
uniforms on the unit square.
"""

from __future__ import annotations

import numpy as np
import stochastic_rs as sr


def _fitted_clayton(tau: float):
    c = sr.Clayton(tau=tau)
    c.compute_theta()
    return c


def test_clayton_seed_determinism():
    a = _fitted_clayton(0.5).sample(1000, seed=42)
    b = _fitted_clayton(0.5).sample(1000, seed=42)
    assert np.allclose(a, b)


def test_clayton_sample_shape():
    s = _fitted_clayton(0.5).sample(1000, seed=1)
    assert s.shape == (1000, 2)


def test_clayton_samples_are_uniform_marginals():
    s = _fitted_clayton(0.4).sample(20_000, seed=3)
    # Each margin should be ~Uniform(0,1): mean ≈ 0.5, all in (0,1).
    assert np.all(s > 0.0) and np.all(s < 1.0)
    assert abs(float(np.mean(s[:, 0])) - 0.5) < 0.02
    assert abs(float(np.mean(s[:, 1])) - 0.5) < 0.02


def test_clayton_distinct_seeds_differ():
    a = _fitted_clayton(0.5).sample(1000, seed=1)
    b = _fitted_clayton(0.5).sample(1000, seed=2)
    assert not np.allclose(a, b)


def test_clayton_positive_dependence():
    # Clayton with τ = 0.6 is strongly positively dependent: the empirical
    # Spearman correlation of the two margins must be clearly positive.
    s = _fitted_clayton(0.6).sample(20_000, seed=7)
    corr = np.corrcoef(s[:, 0], s[:, 1])[0, 1]
    assert corr > 0.5


def test_clayton_samples_finite():
    s = _fitted_clayton(0.3).sample(5000, seed=9)
    assert np.all(np.isfinite(s))

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


def test_bb1_bb7_fit_and_tails():
    truth = sr.Bb1(theta=0.8, delta=1.6)
    lower, upper = truth.tail_dependence()
    assert abs(lower - 2 ** (-1 / (0.8 * 1.6))) < 1e-12
    assert abs(upper - (2 - 2 ** (1 / 1.6))) < 1e-12
    raw = truth.sample(3000, seed=42)
    uv = sr.pseudo_observations(raw)
    assert uv.shape == (3000, 2) and uv.min() > 0 and uv.max() < 1
    assert np.allclose(np.sort(uv[:, 0]), np.arange(1, 3001) / 3001)
    fitted = sr.Bb1()
    fitted.fit(uv)
    assert abs(fitted.theta() - 0.8) < 0.25 and abs(fitted.delta() - 1.6) < 0.25
    assert np.all(fitted.pdf(uv[:10]) > 0) and np.all(np.diff(fitted.cdf(np.column_stack([np.linspace(0.1, 0.9, 5)] * 2))) > 0)
    bb7 = sr.Bb7(theta=1.7, delta=0.9)
    lo, up = bb7.tail_dependence()
    assert abs(up - (2 - 2 ** (1 / 1.7))) < 1e-12 and abs(lo - 2 ** (-1 / 0.9)) < 1e-12
    bb7.set_tau(0.4)
    theta = bb7.compute_theta()
    assert theta > 1.0


def test_copula_gof_separates_families():
    # The true-family p-value is uniform under the null: best of three data seeds.
    p_right = max(
        sr.copula_gof(
            "clayton", sr.pseudo_observations(_fitted_clayton(0.6).sample(600, seed=s)), replications=40, seed=100
        )[1]
        for s in (5, 6, 7)
    )
    uv = sr.pseudo_observations(_fitted_clayton(0.6).sample(600, seed=5))  # theta = 3
    stat_right, _ = sr.copula_gof("clayton", uv, replications=1, seed=100)
    stat_wrong, p_wrong = sr.copula_gof("gaussian", uv, replications=40, seed=100)
    assert p_right > 0.05 and p_wrong < 0.05
    assert stat_wrong > stat_right


def test_fit_vine_recovers_a_clayton_edge():
    uv = sr.pseudo_observations(_fitted_clayton(0.5).sample(2000, seed=7))  # theta = 2
    fit = sr.fit_vine(uv, structure="dvine", criterion="aic")
    assert fit["families"] == [["clayton"]]
    assert abs(fit["parameters"][0][0][0] - 2.0) < 0.3
    assert fit["parameter_count"] == 1 and fit["bic"] > fit["aic"]
    fit_c = sr.fit_vine(uv, structure="cvine", criterion="bic", families=["independence", "gaussian", "frank"])
    assert fit_c["families"][0][0] in ("gaussian", "frank")
    assert sorted(fit_c["order"]) == [0, 1]

"""Copula-surface pytest coverage.

Exercises the bivariate Clayton copula binding: theta-from-tau,
seed-deterministic sampling, sample shape, and that the samples are valid
uniforms on the unit square.
"""

from __future__ import annotations

import numpy as np
import pytest
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


def test_kendall_tau_b_matrix_and_empirical_copula_ranks():
    # (1,1,2,3) x (1,2,2,3): four concordant pairs and one tie per margin, so tau-b = 4 / sqrt(5 * 5) = 0.8 (tau-a is 2/3).
    tied = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]])
    tau = sr.kendall_tau_matrix(tied)
    assert abs(tau[0, 1] - 0.8) < 1e-15 and tau[0, 1] == tau[1, 0] and tau[0, 0] == tau[1, 1] == 1.0
    with pytest.raises(ValueError):
        sr.kendall_tau_matrix(tied[:1])

    ec = sr.EmpiricalCopula2D(np.array([3.0, 1.0, 2.0, 4.0]), np.array([10.0, 30.0, 20.0, 40.0]))
    ranks = ec.rank_data()
    assert np.array_equal(ranks, np.array([[0.6, 0.2], [0.2, 0.6], [0.4, 0.4], [0.8, 0.8]]))  # rank / (n + 1)
    # C_n(0.5, 0.5): of the rank pairs (3,1), (1,3), (2,2), (4,4) only (2,2) lies at or below (2.5, 2.5) of n + 1 = 5.
    assert np.mean((ranks[:, 0] <= 0.5) & (ranks[:, 1] <= 0.5)) == 0.25
    assert {tuple(row) for row in ec.sample(50, seed=3)} <= {tuple(row) for row in ranks}
    with pytest.raises(ValueError):
        sr.EmpiricalCopula2D(np.array([1.0, 2.0]), np.array([1.0]))


def test_pseudo_observations_share_mean_ranks_and_propagate_nan():
    # Tied values take the mean of their ranks (R's `rank`, scipy's `rankdata`), so a row order cannot change the result.
    a = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]])
    ua, ub = sr.pseudo_observations(a), sr.pseudo_observations(a[[1, 0, 2, 3]])
    assert np.array_equal(ua, np.array([[0.3, 0.2], [0.3, 0.5], [0.6, 0.5], [0.8, 0.8]]))
    assert np.array_equal(ub, ua[[1, 0, 2, 3]])
    assert abs(sr.kendall_tau_matrix(ua)[0, 1] - 0.8) < 1e-15 and abs(sr.kendall_tau_matrix(ub)[0, 1] - 0.8) < 1e-15
    # A column holding a NaN is all NaN, at any size, and the other columns keep their ranks.
    holed = np.column_stack([np.arange(100.0), np.arange(100.0)[::-1]])
    holed[50, 0] = np.nan
    out = sr.pseudo_observations(holed)
    assert np.isnan(out[:, 0]).all() and np.array_equal(out[:, 1], np.arange(100, 0, -1) / 101)


def test_set_theta_replaces_a_pending_tau():
    clayton = sr.Clayton(tau=0.5)
    clayton.set_theta(4.0)
    assert abs(clayton.tau() - 2.0 / 3.0) < 1e-15
    assert abs(clayton.compute_theta() - 4.0) < 1e-12
    bb1 = sr.Bb1(tau=0.5, delta=1.5)
    bb1.set_theta(0.8)
    assert abs(bb1.tau() - 0.5) > 0.01 and abs(bb1.compute_theta() - 0.8) < 1e-12

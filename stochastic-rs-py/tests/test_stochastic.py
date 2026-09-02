"""Stochastic-process-surface pytest coverage.

Exercises the GBM process binding: seed determinism, path shape, the
parallel multi-path sampler, and basic sanity of the simulated levels.
"""

from __future__ import annotations

import numpy as np
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


def test_sobol_and_bridge_qmc():
    plain = sr.PySobolSeq(3000)
    pts = plain.sample(64)
    assert pts.shape == (64, 3000) and (pts >= 0).all() and (pts < 1).all()
    assert not plain.is_scrambled
    scrambled = sr.PySobolSeq(8, seed=5)
    a, b = scrambled.sample(1024), sr.PySobolSeq(8, seed=5).sample(1024)
    assert scrambled.is_scrambled and np.array_equal(a, b)
    qmc = sr.PyBrownianBridgeQmc(32, 2.0, seed=3)
    w = qmc.paths(2048)
    assert w.shape == (2048, 32) and abs(np.var(w[:, -1]) - 2.0) < 0.15
    dw = qmc.increments(2048)
    assert np.allclose(np.cumsum(dw, axis=1), w)


def test_multi_gbm_and_correlated_driver():
    rho = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    dw = sr.PyMcgns(rho, 100_000, t=1.0, seed=3).sample()
    assert dw.shape == (3, 100_000)
    assert np.allclose(np.corrcoef(dw), rho, atol=0.02)
    model = sr.PyMultiGbm([0.05, 0.03, 0.01], [0.2, 0.3, 0.15], rho, 64, [100.0, 50.0, 10.0], t=1.0, seed=7)
    one = model.sample()
    assert one.shape == (3, 64) and np.allclose(one[:, 0], [100.0, 50.0, 10.0])
    paths = model.sample_par(2000)
    assert len(paths) == 2000 and paths[0].shape == (3, 64)
    assert not np.array_equal(paths[0], paths[1])


def test_wishart_process():
    b = np.array([[-0.5, 0.1], [0.05, -0.3]])
    a = np.array([[0.3, 0.1], [0.0, 0.2]])
    x0 = np.array([[1.0, 0.2], [0.2, 0.5]])
    process = sr.PyWishart(2.5, b, a, x0, 5, t=1.0, seed=7)
    path = process.sample()
    assert path.shape == (5, 2, 2)
    assert np.allclose(path[0], x0) and np.allclose(path, np.swapaxes(path, 1, 2))
    paths = process.sample_par(4000)
    assert len(paths) == 4000 and paths[0].shape == (5, 2, 2)
    terminal = np.mean([p[-1] for p in paths], axis=0)
    assert np.abs(terminal - process.mean(1.0)).max() < 0.05
    v = -0.4 * np.array([[1.0, 0.3], [0.3, 1.0]])
    lt = process.laplace_transform(v, 1.0)
    assert 0.0 < lt < 1.0
    mc = np.mean([np.exp(np.trace(v @ p[-1])) for p in paths])
    assert abs(mc - lt) < 0.03

"""Distribution-surface pytest coverage for the `stochastic_rs` extension.

Mirrors and extends the verified `python_smoke.py` checks: seed
determinism, sample shapes, statistical moments, and the parallel
sampling path. Run after `maturin develop` from the workspace root:

    maturin develop --release
    pytest stochastic-rs-py/tests/test_distributions.py
"""

from __future__ import annotations

import numpy as np
import pytest
import stochastic_rs as sr


def test_normal_seed_determinism():
    a = sr.PyNormal(0.0, 1.0, seed=42).sample(1024)
    b = sr.PyNormal(0.0, 1.0, seed=42).sample(1024)
    assert np.allclose(a, b)


def test_normal_distinct_seeds_differ():
    a = sr.PyNormal(0.0, 1.0, seed=1).sample(1024)
    b = sr.PyNormal(0.0, 1.0, seed=2).sample(1024)
    assert not np.allclose(a, b)


def test_normal_sample_shape():
    s = sr.PyNormal(0.0, 1.0, seed=7).sample(2048)
    assert s.shape == (2048,)


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (2.5, 0.5), (-1.0, 3.0)])
def test_normal_moments(mean, std):
    s = sr.PyNormal(mean, std, seed=123).sample(200_000)
    assert abs(float(np.mean(s)) - mean) < 0.05
    assert abs(float(np.std(s)) - std) < 0.05


def test_normal_seed_zero_is_valid():
    s = sr.PyNormal(0.0, 1.0, seed=0).sample(256)
    assert s.shape == (256,)
    assert np.all(np.isfinite(s))


def test_normal_unseeded_runs():
    s = sr.PyNormal(0.0, 1.0).sample(4096)
    assert s.shape == (4096,)
    assert abs(float(np.mean(s))) < 0.1


def test_normal_sample_par_determinism():
    a = sr.PyNormal(0.0, 1.0, seed=99).sample_par(64, 1024)
    b = sr.PyNormal(0.0, 1.0, seed=99).sample_par(64, 1024)
    assert np.allclose(a, b)
    assert a.shape == (64, 1024)


def test_normal_sample_par_shape():
    s = sr.PyNormal(0.0, 1.0, seed=5).sample_par(16, 256)
    assert s.shape == (16, 256)


def test_normal_all_finite():
    s = sr.PyNormal(0.0, 1.0, seed=11).sample(10_000)
    assert np.all(np.isfinite(s))


def test_normal_small_sample():
    s = sr.PyNormal(0.0, 1.0, seed=3).sample(8)
    assert s.shape == (8,)

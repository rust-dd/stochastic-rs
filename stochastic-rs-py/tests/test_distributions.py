"""Distribution-surface pytest coverage for the `stochastic_rs` extension.

Mirrors and extends the verified `python_smoke.py` checks: seed
determinism, sample shapes, statistical moments, and the parallel
sampling path. Run after `maturin develop` from the workspace root:

    maturin develop --release
    pytest stochastic-rs-py/tests/test_distributions.py
"""

from __future__ import annotations

import subprocess
import sys

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


def test_gpd_and_gev_samplers_are_seeded_and_bounded():
    gpd = sr.PyGpd(0.0, 1.0, -0.5, seed=3)
    a = gpd.sample(4096)
    b = sr.PyGpd(0.0, 1.0, -0.5, seed=3).sample(4096)
    assert np.allclose(a, b)
    assert a.min() >= 0.0 and a.max() <= 2.0 + 1e-9  # support [0, mu - sigma/xi]
    gev = sr.PyGev(0.0, 1.0, 0.2, seed=4)
    z = gev.sample_par(4, 1024)
    assert z.shape == (4, 1024)
    assert np.isfinite(z).all()


def test_a6_distributions_sample_and_standardise():
    vg = sr.PyVarianceGamma(0.2, 0.5, -0.1, 0.05, seed=1).sample(200_000)
    assert abs(vg.mean() - (-0.05)) < 2e-3
    jsu = sr.PyJohnsonSu(-0.5, 1.5, 0.2, 2.0, seed=2).sample(100_000)
    assert abs(jsu.mean() - 1.048069681819088) < 0.03
    skt = sr.PySkewT(5.0, -0.3, seed=3).sample(200_000)
    assert abs(skt.mean()) < 0.01 and abs(skt.var() - 1.0) < 0.03
    assert sr.PySkewT(8.0, 0.4, seed=4).sample_par(3, 512).shape == (3, 512)


def test_gig_gh_and_tempered_stable_samplers():
    gig = sr.PyGig(0.3, 2.0, 0.5, seed=5).sample(200_000)
    assert gig.min() > 0.0 and abs(gig.mean() - 3.51040667383761) < 0.05
    gh = sr.PyGeneralizedHyperbolic(1.0, 2.0, 0.5, 1.5, -0.2, seed=6).sample(200_000)
    assert abs(gh.mean() - 0.40032685290757525) < 0.02
    ts = sr.PyTemperedStable(0.6, 2.0, 1.5, seed=7).sample(200_000)
    assert ts.min() > 0.0 and abs(ts.mean() - 1.5 * 0.6 * 2.0 ** (-0.4)) < 0.02
    assert sr.PyGig(-0.5, 1.0, 4.0, seed=8).sample_par(2, 1024).shape == (2, 1024)


def test_normal_calls_continue_one_stream():
    d = sr.PyNormal(0.0, 1.0, seed=5)
    first, second = d.sample(64), d.sample(64)
    twin = sr.PyNormal(0.0, 1.0, seed=5)
    assert np.array_equal(first, twin.sample(64))
    assert np.array_equal(second, twin.sample(64))
    assert not np.array_equal(first, second)


_CONCURRENT_CALLERS = """
import sys
import threading

import stochastic_rs as sr

assert "numpy" not in sys.modules
d = sr.PyNormal(0.0, 1.0, seed=1)
barrier = threading.Barrier(8)
blocks = [[] for _ in range(8)]


def work(i):
    barrier.wait()
    for k in range(4):
        blocks[i].append(d.sample(257) if (i + k) % 2 else d.sample_par(4, 16384))


threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
got = [b for per_thread in blocks for b in per_thread]
vectors = sorted(b.tobytes() for b in got if b.ndim == 1)
matrices = sorted(b.tobytes() for b in got if b.ndim == 2)
twin_v, twin_m = sr.PyNormal(0.0, 1.0, seed=1), sr.PyNormal(0.0, 1.0, seed=1)
assert vectors == sorted(twin_v.sample(257).tobytes() for _ in vectors)
assert matrices == sorted(twin_m.sample_par(4, 16384).tobytes() for _ in matrices)
"""


def test_one_normal_survives_concurrent_callers():
    """A deadlock would hold the GIL, so the callers run in a fresh interpreter (numpy not yet
    imported, as on a cold start) that a timeout turns into a failure."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CONCURRENT_CALLERS],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("concurrent calls on one distribution deadlocked", pytrace=False)
    assert result.returncode == 0, result.stderr


def test_a_failed_call_leaves_the_stream_usable():
    d = sr.PyNormal(0.0, 1.0, seed=3)
    with pytest.raises(RuntimeError):
        d.sample_par(1, 2**63)
    twin = sr.PyNormal(0.0, 1.0, seed=3)
    assert np.array_equal(d.sample(16), twin.sample(16))

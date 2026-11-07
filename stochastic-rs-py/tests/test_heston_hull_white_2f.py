"""`PyHestonHullWhite2F`: construction, invalid input, path shape, seed reproducibility, `sample_par`."""

from __future__ import annotations

import numpy as np
import pytest
import stochastic_rs as sr

RHO = np.array(
    [
        [1.0, -0.3, 0.35, 0.08],
        [-0.3, 1.0, 0.0, 0.0],
        [0.35, 0.0, 1.0, -0.4],
        [0.08, 0.0, -0.4, 1.0],
    ]
)


def _hybrid(**kwargs):
    args = {
        "kappa": 0.8, "theta": 0.2, "sigma": 0.2, "theta_r": lambda t: 1.1 * 0.03, "a": 1.1,
        "sigma1": 0.01, "sigma2": 0.015, "b": 0.8, "rho": RHO, "n": 65, "s0": 1.0, "v0": 0.2,
        "r0": 0.03, "t": 1.0,
    }
    return sr.PyHestonHullWhite2F(**{**args, **kwargs})


@pytest.mark.parametrize("scheme", ["euler", "qe"])
def test_sample_shape_rows_and_initial_state(scheme):
    path = _hybrid(seed=42, scheme=scheme).sample()
    assert path.shape == (5, 65)
    assert path.dtype == np.float64
    np.testing.assert_array_equal(path[:, 0], [1.0, 0.2, 0.03, 0.0, 0.0])
    assert np.all(path[0] > 0.0) and np.all(path[1] >= 0.0)
    # The integrated rate row is the trapezoidal sum of the rate row.
    dt = 1.0 / 64
    trapezoid = np.concatenate([[0.0], np.cumsum(0.5 * (path[2, :-1] + path[2, 1:]) * dt)])
    np.testing.assert_allclose(path[4], trapezoid, atol=1e-12)


@pytest.mark.parametrize("scheme", ["euler", "qe"])
def test_seed_reproduces_and_seeds_differ(scheme):
    a = _hybrid(seed=7, scheme=scheme).sample()
    b = _hybrid(seed=7, scheme=scheme).sample()
    c = _hybrid(seed=8, scheme=scheme).sample()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_par_shape_and_reproducibility():
    paths = _hybrid(seed=3, scheme="qe").sample_par(16)
    assert paths.shape == (16, 5, 65)
    assert np.array_equal(paths, _hybrid(seed=3, scheme="qe").sample_par(16))
    assert not np.array_equal(paths[0], paths[1])


def test_f32_dtype_samples():
    path = _hybrid(seed=1, dtype="f32").sample()
    assert path.dtype == np.float32 and path.shape == (5, 65)


@pytest.mark.parametrize(
    "bad",
    [
        {"sigma": -0.1},
        {"n": 1},
        {"s0": 0.0},
        {"rho": np.eye(4) * 2.0},
        {"rho": np.array([[1.0, 0.9, 0.9, 0.0], [0.9, 1.0, -0.9, 0.0], [0.9, -0.9, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])},
        {"scheme": "milstein"},
        {"dtype": "f16"},
    ],
)
def test_invalid_input_is_a_value_error(bad):
    with pytest.raises(ValueError):
        _hybrid(**bad)


def test_qe_rejects_a_rate_variance_correlation():
    rho = RHO.copy()
    rho[1, 2] = rho[2, 1] = 0.2
    _hybrid(rho=rho)  # Euler takes any positive-definite matrix
    with pytest.raises(ValueError, match="rate-variance"):
        _hybrid(rho=rho, scheme="qe")


def test_callback_failure_is_a_runtime_error():
    def broken(t):
        raise RuntimeError("theta_r failed")

    with pytest.raises(RuntimeError, match="theta_r failed"):
        _hybrid(theta_r=broken, seed=5).sample()


_WORKER = """
import json
import sys
import numpy as np
import stochastic_rs as sr
rho = np.array([[1.0, -0.3, 0.35, 0.08], [-0.3, 1.0, 0.0, 0.0], [0.35, 0.0, 1.0, -0.4], [0.08, 0.0, -0.4, 1.0]])
p = sr.PyHestonHullWhite2F(0.8, 0.2, 0.2, lambda t: 0.033, 1.1, 0.01, 0.015, 0.8, rho, 16, s0=1.0, v0=0.2,
                           r0=0.03, t=1.0, seed=42, scheme=sys.argv[1])
print(json.dumps(p.sample_par(70).tolist()))
"""


@pytest.mark.parametrize("scheme", ["euler", "qe"])
def test_sample_par_is_thread_count_independent(scheme):
    """As `test_sample_par_thread_count.py`: rayon's global pool is sized once per interpreter, so each
    pool size runs in its own subprocess under `RAYON_NUM_THREADS`; 70 paths is above `MAX_CHUNKS`."""
    import json
    import os
    import subprocess
    import sys

    def run(threads):
        env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
        out = subprocess.run([sys.executable, "-c", _WORKER, scheme], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        return np.array(json.loads(out.stdout))

    one = run(1)
    assert one.shape == (70, 5, 16)
    assert np.array_equal(one, run(3))
    assert np.array_equal(one, run(8))

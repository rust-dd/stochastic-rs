"""`sample_par` on a Python-callable jump law must not deadlock.

Seven process classes take a Python callable as their jump law: `PyMerton`,
`PyKou`, `PyLevyDiffusion`, `PyJumpFou`, `PyJumpFOUCustom`, `PyBates` and
`PyCustomJt`. Their `sample_par` runs the paths on rayon's workers, and every
draw of the law re-attaches to the interpreter there. A wrapper that keeps the
GIL while it waits for those workers hangs for any `m >= 2`: the caller holds
the lock the workers are waiting for.

The watchdog has to sit outside the process under test. The hung thread holds
the GIL, so a `Thread.join(timeout)` inside the same interpreter could never
get back to the assertion that follows it. Each case therefore runs in its own
interpreter under `subprocess.run(timeout=...)`, which kills a hung child and
turns the hang into an ordinary test failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("stochastic_rs")

_TIMEOUT_SECONDS = 30
_PATHS = 8
_N = 64

# Constructor calls, evaluated in the child interpreter, where `sr` is the
# extension and `law()` builds a Python jump law. Bates' `alpha` and `beta` are
# the variance drift's intercept and slope, kappa * theta and kappa, so 0.08
# and 2.0 are kappa = 2.0 and theta = 0.04.
_CASES = {
    "merton": f"sr.PyMerton(alpha=0.05, sigma=0.2, lambda_=5.0, theta=0.0, distribution=law(), n={_N}, x0=1.0, t=1.0, seed=1)",
    "kou": f"sr.PyKou(alpha=0.05, sigma=0.2, lambda_=5.0, theta=0.0, distribution=law(), n={_N}, x0=1.0, t=1.0, seed=1)",
    "levy_diffusion": f"sr.PyLevyDiffusion(gamma_=0.05, sigma=0.2, distribution=law(), lambda_=5.0, n={_N}, x0=1.0, t=1.0, seed=1)",
    "jump_fou": f"sr.PyJumpFou(hurst=0.7, theta=1.0, mu=0.0, sigma=0.2, distribution=law(), lambda_=5.0, n={_N}, x0=0.0, t=1.0, seed=1)",
    "jump_fou_custom": f"sr.PyJumpFOUCustom(hurst=0.7, theta=1.0, mu=0.0, sigma=0.2, jump_times=lambda: 0.2, jump_sizes=law(), n={_N}, x0=0.0, t=1.0)",
    "bates": f"sr.PyBates(lambda_=5.0, k=0.0, alpha=0.08, beta=2.0, sigma=0.3, rho=-0.5, distribution=law(), n={_N}, mu=0.05, s0=1.0, v0=0.04, t=1.0, seed=1)",
    "customjt": f"sr.PyCustomJt(distribution=lambda: 0.1, n={_N})",
}

_WORKER = """
import json
import random

import numpy as np
import stochastic_rs as sr


def law():
    rng = random.Random(0)
    return lambda: rng.gauss(0.0, 0.1)


out = {ctor}.sample_par({m})
arrays = out if isinstance(out, tuple) else (out,)
print(json.dumps([list(np.asarray(a).shape) for a in arrays]))
"""


def _shapes(ctor: str, m: int) -> list[list[int]]:
    """Shapes of the arrays `sample_par(m)` returns, taken in a child
    interpreter that is killed if it does not finish."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _WORKER.format(ctor=ctor, m=m)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        result = None
    if result is None:
        pytest.fail(
            f"sample_par({m}) deadlocked on a Python-callable jump law", pytrace=False
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", _CASES)
def test_sample_par_with_python_law_finishes(name):
    shapes = _shapes(_CASES[name], _PATHS)
    assert shapes and all(shape == [_PATHS, _N] for shape in shapes)


@pytest.mark.parametrize("name", _CASES)
def test_sample_par_of_no_paths_is_empty(name):
    shapes = _shapes(_CASES[name], 0)
    assert shapes and all(shape[0] == 0 for shape in shapes)

"""Seeded `sample_par` is thread-count independent across PyO3; rayon sizes its global pool once, so
each size is a subprocess (Rust guard: `reproducibility_all_processes.rs`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
import pytest

pytest.importorskip("stochastic_rs")

_SEED = 42
_N = 32
# <= MAX_CHUNKS (`traits/process.rs`, 64): one path per chunk; above it several share a chunk,
# mirroring the Rust guard's `M_ONE_PER_CHUNK` / `M_MULTI_PER_CHUNK` split.
_M_VALUES = (64, 256)
_THREAD_COUNTS = (1, 2, 3, 8)

_WORKER = """
import json
import sys
import stochastic_rs as sr
m = int(sys.argv[1])
p = sr.PyGbm(0.05, 0.2, {n}, x0=100.0, t=1.0, seed={seed}).sample_par(m)
print(json.dumps(p.tolist()))
""".format(n=_N, seed=_SEED)


@lru_cache(maxsize=None)
def _sample_par_under(m: int, num_threads: int) -> np.ndarray:
    """Runs `_WORKER` in a fresh subprocess with a rayon pool of `num_threads`; cached, as the
    cases share the `num_threads=1` baseline."""
    env = dict(os.environ, RAYON_NUM_THREADS=str(num_threads))
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, str(m)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"worker(m={m}) under RAYON_NUM_THREADS={num_threads} failed:\n"
        f"{result.stderr}"
    )
    return np.array(json.loads(result.stdout))


@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("num_threads", _THREAD_COUNTS)
def test_gbm_sample_par_is_thread_count_independent(m, num_threads):
    baseline = _sample_par_under(m, 1)
    other = _sample_par_under(m, num_threads)
    assert baseline.shape == (m, _N)
    assert np.array_equal(baseline, other), (
        f"sample_par(seed={_SEED}, m={m}) differed between "
        f"RAYON_NUM_THREADS=1 and RAYON_NUM_THREADS={num_threads}"
    )


@pytest.mark.parametrize("m", _M_VALUES)
def test_gbm_sample_par_paths_are_distinct(m):
    """Guards against the matrix above passing because every row is the same degenerate value."""
    paths = _sample_par_under(m, 1)
    assert not np.allclose(paths[0], paths[1])

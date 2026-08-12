"""Thread-count independence of seeded `sample_par` across the PyO3 boundary.

`ProcessExt::sample_par`'s guarantee — same seed and same `m` give
bit-identical output on any rayon thread-pool size — is proven in-process by
`stochastic-rs-stochastic/tests/reproducibility_all_processes.rs`: that guard
builds a fresh `rayon::ThreadPool` per pool size and calls `.install(|| ...)`
around each `sample_par` call, all inside one Rust test binary.

Python has no equivalent of `.install()` reachable from the binding, and
rayon's *global* pool — the one every `py_process_*!`-generated `sample_par`
uses — is sized once, from `RAYON_NUM_THREADS` (or the machine's core count)
if unset, the first time any parallel iterator runs in the process, and
cannot be resized or rebuilt afterwards for that interpreter's remaining
lifetime. So the only way to observe two different pool sizes from Python is
two different interpreter processes, one per size, each making its first
`sample_par` call under a different `RAYON_NUM_THREADS`.

Why this must fail if `sample_par` were thread-count dependent: `chunk_count`
(`stochastic-rs-stochastic/src/traits/process.rs`) is documented to be a
pure function of `m` alone, and every chunk's sampler is built sequentially,
before any chunk reaches rayon. If a future change made `chunk_count` read
`rayon::current_num_threads()` instead — the exact regression class that
function's own doc comment warns against — `sample_par(seed=42, m=256)`
would split into a different number of chunks under `RAYON_NUM_THREADS=1`
than under `RAYON_NUM_THREADS=8`, so the sequence of `derive()` calls that
seeds each chunk would differ, and the two subprocesses below would produce
different arrays. This was verified directly: temporarily changing
`chunk_count` to `m.min(MAX_CHUNKS).min(rayon::current_num_threads())` and
rebuilding the extension made `test_gbm_sample_par_is_thread_count_independent`
fail with a shape/value mismatch between the `RAYON_NUM_THREADS=1` and `=8`
runs, at every `m` in `_M_VALUES`; reverting restored a pass. See
MIGRATION.md and the follow-up report for the exact before/after.
"""

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
# <= MAX_CHUNKS (`traits/process.rs`, currently 64): one path per chunk, the
# regime that cannot by itself expose cross-chunk correlation. > MAX_CHUNKS:
# several paths share a chunk. Both regimes are exercised, mirroring the
# Rust-side guard's own `M_ONE_PER_CHUNK` / `M_MULTI_PER_CHUNK` split.
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
    """Runs `_WORKER` in a fresh subprocess with `RAYON_NUM_THREADS` fixed to
    `num_threads`, so rayon's global pool is first-sized to exactly that
    count before this process's one and only `sample_par` call.

    Cached on `(m, num_threads)`: several test cases below share the
    `num_threads=1` baseline for a given `m`, and this avoids re-spawning an
    identical subprocess for each of them.
    """
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
    """Guards against the matrix above passing only because every row is the
    same degenerate value regardless of threading."""
    paths = _sample_par_under(m, 1)
    assert not np.allclose(paths[0], paths[1])

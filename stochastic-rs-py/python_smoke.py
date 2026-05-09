#!/usr/bin/env python3
"""In-process Python smoke test for the `stochastic_rs` extension module.

Run with:
    cd stochastic-rs-py
    maturin develop --release           # build + install editable wheel
    python python_smoke.py

The script exercises the 5 main API surfaces (distributions, stochastic
processes, copulas, stats, quant) and verifies seed-determinism wherever
the wrapper accepts a `seed=` keyword argument.

Exit code 0 = all checks pass; non-zero = first failure raises.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import stochastic_rs as sr


def check_distributions_seed():
    n1 = sr.PyNormal(0.0, 1.0, seed=42)
    n2 = sr.PyNormal(0.0, 1.0, seed=42)
    s1 = n1.sample(1024)
    s2 = n2.sample(1024)
    assert np.allclose(s1, s2), "PyNormal seed determinism failed"
    print("[OK]  PyNormal seed determinism (1024 samples)")

    n_unseeded = sr.PyNormal(0.0, 1.0)
    s3 = n_unseeded.sample(1024)
    assert s3.shape == (1024,)
    assert abs(float(np.mean(s3))) < 0.2
    print("[OK]  PyNormal unseeded sampling (mean ~ 0)")

    # sample_par must keep determinism on the seeded path even though the
    # underlying `sample_matrix` clones `self` per rayon worker.
    p1 = sr.PyNormal(0.0, 1.0, seed=99).sample_par(64, 1024)
    p2 = sr.PyNormal(0.0, 1.0, seed=99).sample_par(64, 1024)
    assert np.allclose(p1, p2), "PyNormal sample_par seed determinism failed"
    assert p1.shape == (64, 1024)
    print("[OK]  PyNormal sample_par seed determinism (64x1024)")


def check_stochastic_seed():
    g1 = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42)
    g2 = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42)
    p1 = g1.sample()
    p2 = g2.sample()
    assert np.allclose(p1, p2), "PyGbm seed determinism failed"
    assert p1.shape[0] == 252
    print("[OK]  PyGbm seed determinism (T=1y, n=252)")

    # sample_par determinism on the seeded path: the wrapper must serialize
    # because the default ProcessExt::sample_par would race on the shared
    # Deterministic atomic state.
    pp1 = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    pp2 = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    assert np.allclose(pp1, pp2), "PyGbm sample_par seed determinism failed"
    assert pp1.shape == (8, 252)
    print("[OK]  PyGbm sample_par seed determinism (8 paths)")


def check_copula_seed():
    c1 = sr.Clayton(tau=0.5)
    c2 = sr.Clayton(tau=0.5)
    c1.compute_theta()
    c2.compute_theta()
    s1 = c1.sample(1000, seed=42)
    s2 = c2.sample(1000, seed=42)
    assert np.allclose(s1, s2), "Clayton seed determinism failed"
    assert s1.shape == (1000, 2)
    print("[OK]  Clayton bivariate copula seed determinism")


def check_stats_jb():
    arr = np.random.default_rng(0).standard_normal(2000)
    jb = sr.JarqueBera(arr)
    stat = jb.statistic
    pv = jb.p_value
    assert stat >= 0.0
    assert 0.0 <= pv <= 1.0
    print(f"[OK]  JarqueBera N(0,1)·2000 → JB={stat:.3f}, p={pv:.3f}")


def check_quant_bsm():
    p = sr.BSMPricer(s=100.0, v=0.2, k=100.0, r=0.05, tau=1.0)
    price = p.price()
    assert 9.0 < price < 12.0, f"BSM ATM 1y 20% vol price out of range: {price}"
    print(f"[OK]  BSMPricer ATM 1y σ=20%, r=5% → {price:.4f}")


CHECKS = [
    ("distributions", check_distributions_seed),
    ("stochastic", check_stochastic_seed),
    ("copulas", check_copula_seed),
    ("stats", check_stats_jb),
    ("quant", check_quant_bsm),
]


def main() -> int:
    failed = []
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
            failed.append(name)
    print()
    if failed:
        print(f"smoke test FAILED ({len(failed)}/{len(CHECKS)}): {', '.join(failed)}")
        return 1
    print(f"stochastic-rs Python smoke test: ALL {len(CHECKS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

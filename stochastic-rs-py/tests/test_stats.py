"""Stats-surface pytest coverage.

Exercises the Jarque-Bera normality test binding: statistic / p-value
ranges, and the qualitative behaviour on Gaussian vs heavy-tailed data.
"""

from __future__ import annotations

import numpy as np
import stochastic_rs as sr


def test_jarque_bera_ranges_on_normal():
    arr = np.random.default_rng(0).standard_normal(2000)
    jb = sr.JarqueBera(arr)
    assert jb.statistic >= 0.0
    assert 0.0 <= jb.p_value <= 1.0


def test_jarque_bera_does_not_reject_gaussian():
    arr = np.random.default_rng(1).standard_normal(5000)
    jb = sr.JarqueBera(arr)
    # Gaussian data: JB should not reject normality at the 1% level.
    assert jb.p_value > 0.01


def test_jarque_bera_rejects_heavy_tails():
    # Student-t(3) is heavy-tailed; JB should reject normality strongly.
    arr = np.random.default_rng(2).standard_t(3, size=5000)
    jb = sr.JarqueBera(arr)
    assert jb.statistic > 0.0
    assert jb.p_value < 0.05


def test_jarque_bera_statistic_finite():
    arr = np.random.default_rng(3).standard_normal(1000)
    jb = sr.JarqueBera(arr)
    assert np.isfinite(jb.statistic)
    assert np.isfinite(jb.p_value)

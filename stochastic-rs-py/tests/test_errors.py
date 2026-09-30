"""A Rust panic reaches Python as an exception `except Exception` catches.

PyO3 raises an uncaught panic as `pyo3_runtime.PanicException`, a
`BaseException`. The bindings report a constructor's panic, an assertion on an
argument, as `ValueError`, and a sampling call's panic as `RuntimeError`.
"""

import numpy as np
import pytest
import stochastic_rs as sr


def test_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match="or mu must be provided"):
        sr.PyGbmLog(sigma=0.2, n=16)


def test_process_sampling_raises_runtime_error():
    gbm = sr.PyGbmLog(mu=0.05, sigma=0.2, n=16, s0=-1.0)
    with pytest.raises(RuntimeError, match="s0 must be > 0"):
        gbm.sample()


def test_distribution_constructor_raises_value_error():
    with pytest.raises(ValueError, match="std_dev must satisfy"):
        sr.PyNormal(0.0, -1.0)


def test_hand_written_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match="one entry per asset"):
        sr.PyMultiGbm([0.05, 0.03], [0.2], np.eye(2), 16, [100.0, 100.0])


def test_hand_written_pricer_constructor_raises_value_error():
    with pytest.raises(ValueError, match="is_power_of_two"):
        sr.CarrMadanPricer(n=12)

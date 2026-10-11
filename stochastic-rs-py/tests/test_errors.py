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


def test_distribution_constructor_names_the_rejected_argument():
    want = r"scale must satisfy `scale > T::zero\(\)`, got scale = 0\.0"
    with pytest.raises(ValueError, match=want):
        sr.PyGamma(2.0, 0.0)


def test_hand_written_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match="one entry per asset"):
        sr.PyMultiGbm([0.05, 0.03], [0.2], np.eye(2), 16, [100.0, 100.0])


def test_hand_written_pricer_constructor_raises_value_error():
    with pytest.raises(ValueError, match="is_power_of_two"):
        sr.CarrMadanPricer(n=12)


def _clayton_pseudo_observations():
    c = sr.Clayton(tau=0.5)
    c.compute_theta()
    return sr.pseudo_observations(c.sample(200, seed=1))


def test_copula_gof_alpha_outside_the_unit_interval_raises_value_error():
    with pytest.raises(ValueError, match="alpha must satisfy"):
        sr.copula_gof("clayton", _clayton_pseudo_observations(), replications=5, alpha=1.0)


def test_copula_gof_zero_replications_raises_value_error():
    with pytest.raises(ValueError, match="replications must satisfy"):
        sr.copula_gof("clayton", _clayton_pseudo_observations(), replications=0)


def test_copula_tail_dependence_before_parameters_raises_value_error():
    with pytest.raises(ValueError, match="no parameters yet"):
        sr.Bb1().tail_dependence()


def test_copula_tail_dependence_out_of_range_raises_value_error():
    with pytest.raises(ValueError, match="theta must satisfy"):
        sr.Bb7(theta=0.5, delta=1.0).tail_dependence()

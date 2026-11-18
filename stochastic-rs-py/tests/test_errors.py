"""Constructors, setters and calls report a rejected argument, input or state as `ValueError`,
which `except Exception` catches; PyO3's `PanicException` is a `BaseException` it would miss."""

import re

import numpy as np
import pytest
import stochastic_rs as sr


def test_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match="exactly one drift"):
        sr.PyGbmLog(sigma=0.2, n=16, s0=1.0)


def test_a_non_positive_spot_is_refused_at_construction():
    with pytest.raises(ValueError, match=r"s0 must satisfy `s0 > T::zero\(\)`, got s0 = -1\.0"):
        sr.PyGbmLog(mu=0.05, sigma=0.2, n=16, s0=-1.0)


def test_distribution_constructor_raises_value_error():
    with pytest.raises(ValueError, match="sigma must satisfy"):
        sr.PyNormal(0.0, -1.0)


def test_distribution_constructor_names_the_rejected_argument():
    want = r"scale must satisfy `scale > T::zero\(\)`, got scale = 0\.0"
    with pytest.raises(ValueError, match=want):
        sr.PyGamma(2.0, 0.0)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: sr.PyArch(0.1, [-0.5], 64), "alpha[0] must satisfy `alpha[i] >= T::zero()`, got alpha[0] = -0.5"),
        (lambda: sr.PyGarch(0.1, [0.1], [0.5, -0.2], 64), "beta[1] must satisfy `beta[i] >= T::zero()`, got beta[1] = -0.2"),
        (
            lambda: sr.PyGjrGarch(0.1, [0.1], [-0.3], [0.5], 64),
            "gamma[0] must satisfy `alpha[i] + gamma[i] >= T::zero()`, got alpha[0] = 0.1, gamma[0] = -0.3",
        ),
        (
            lambda: sr.PyAgarch(0.1, [0.1], [-0.3], [0.5], 64),
            "delta[0] must satisfy `alpha[i] + delta[i] >= T::zero()`, got alpha[0] = 0.1, delta[0] = -0.3",
        ),
        (
            lambda: sr.PyEgarch(1.0, [0.1], [0.0], [1.0], 64),
            "beta must satisfy `1 - Σ beta[j] x^(j+1) has every root outside the unit circle`, got beta = [1.0]",
        ),
    ],
    ids=["arch", "garch", "gjr_garch", "agarch", "egarch"],
)
def test_the_garch_family_refuses_its_preconditions_at_construction(build, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        build()


def test_hand_written_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match=r"sigma\.len\(\) must satisfy `sigma\.len\(\) == rho\.nrows\(\)`, got sigma\.len\(\) = 1"):
        sr.PyMultiGbm([0.05, 0.03], [0.2], np.eye(2), 16, [100.0, 100.0])


def test_hand_written_pricer_constructor_raises_value_error():
    with pytest.raises(ValueError, match="is_power_of_two"):
        sr.CarrMadanPricer(n=12)


def test_the_jump_fourier_models_refuse_parameters_outside_their_domain():
    with pytest.raises(ValueError, match=r"eta1 must satisfy `eta1\.is_finite\(\) && eta1 > 1\.0`, got eta1 = 0\.9"):
        sr.HKDEFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, 0.4, 0.9, 5.0)
    with pytest.raises(ValueError, match=r"sigma_v must satisfy `sigma_v\.is_finite\(\) && sigma_v > 0\.0`, got sigma_v = 0\.0"):
        sr.BatesFourier(0.04, 1.5, 0.04, 0.0, -0.7, 0.5, -0.05, 0.1)


def test_quant_data_error_raises_value_error_with_its_display():
    calibrator = sr.HestonSlvCalibrator(
        100.0, 0.02, 0.0, [90.0, float("nan"), 110.0], [0.5, 1.0], np.full((2, 3), 5.0)
    )
    want = r"invalid strikes: strike 1 must be finite and positive, got NaN$"
    with pytest.raises(ValueError, match=want):
        calibrator.calibrate()


def test_factors_data_error_raises_value_error_with_its_display():
    want = r"observations in a PCA decomposition: needs at least 2, got 1$"
    with pytest.raises(ValueError, match=want):
        sr.PCA(np.zeros((1, 3)))


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


def test_copula_bb7_below_its_theta_range_raises_value_error():
    with pytest.raises(ValueError, match=r"theta must satisfy `1\.0 <= theta < inf`, got theta = 0\.5"):
        sr.Bb7(theta=0.5, delta=1.0)


def test_copula_batch_methods_on_a_wrong_column_count_raise_value_error():
    clayton = sr.Clayton(theta=2.0)
    for columns in (1, 3):
        x = np.full((3, columns), 0.5)
        for method in (clayton.pdf, clayton.cdf):
            with pytest.raises(ValueError, match=f"the copula has dim 2, the input has {columns}"):
                method(x)


def test_copula_parameters_outside_the_domain_raise_value_error():
    with pytest.raises(ValueError, match=r"theta must satisfy `0\.0 <= theta < inf`, got theta = -1\.0"):
        sr.Clayton(theta=-1.0)
    with pytest.raises(ValueError, match="tau must satisfy"):
        sr.Clayton(tau=-0.2)
    clayton = sr.Clayton(theta=2.0)
    with pytest.raises(ValueError, match="theta must satisfy"):
        clayton.set_theta(float("nan"))
    assert clayton.theta() == 2.0
    with pytest.raises(ValueError, match=r"theta must satisfy `theta == 0\.0`"):
        sr.Independence(theta=0.5)
    bb1 = sr.Bb1()
    with pytest.raises(ValueError, match=r"delta must satisfy `1\.0 <= delta < inf`, got delta = 0\.5"):
        bb1.set_delta(0.5)
    assert bb1.delta() == 1.5
    with pytest.raises(ValueError, match="delta must satisfy"):
        sr.Bb7(theta=1.7, delta=0.0)


def test_copula_setter_rejections_print_no_rust_panic(capfd):
    clayton = sr.Clayton(theta=2.0)
    with pytest.raises(ValueError, match=r"theta must satisfy `0\.0 <= theta < inf`, got theta = -1\.0"):
        clayton.set_theta(-1.0)
    bb1 = sr.Bb1(theta=0.8, delta=1.6)
    with pytest.raises(ValueError, match="theta must satisfy"):
        bb1.set_theta(0.0)
    with pytest.raises(ValueError, match=r"delta must satisfy `1\.0 <= delta < inf`, got delta = 0\.5"):
        bb1.set_delta(0.5)
    with pytest.raises(ValueError, match=r"delta must satisfy `0\.0 < delta < inf`, got delta = 0\.0"):
        sr.Bb7(delta=0.0)
    assert (clayton.theta(), bb1.theta(), bb1.delta()) == (2.0, 0.8, 1.6)
    assert "panicked" not in capfd.readouterr().err


def test_hscm_quotes_must_share_one_rate():
    options = [sr.HscmMarketOption(95.0, 0.25, 8.0, 0.03), sr.HscmMarketOption(100.0, 0.5, 7.5, 0.04)]
    with pytest.raises(ValueError, match="HSCM quotes must share one rate; got 0.03 and 0.04"):
        sr.HscmCalibrator(100.0, options)


def test_a_regularization_anchor_that_is_not_finite_is_refused_at_construction():
    slices = [sr.MarketSlice([90.0, 100.0, 110.0], [12.0, 6.0, 2.5], [True] * 3, 0.5)]
    anchor = [0.04, float("nan"), 0.04, 0.5, -0.5]
    with pytest.raises(ValueError, match=r"anchor 1 must satisfy `anchor\.is_finite\(\)`, got anchor = NaN"):
        sr.HestonCalibrator(slices, 100.0, 0.02, regularization=(anchor, [1.0] * 5))

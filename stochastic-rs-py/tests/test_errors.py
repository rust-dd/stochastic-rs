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


@pytest.mark.parametrize(
    "beta, gamma, delta, message",
    [
        (0.0, 1.0, 0.0, "delta must satisfy `delta > T::zero()`, got delta = 0.0"),
        (0.0, 1.0, 1e-200, "delta must satisfy `delta * delta > T::zero()`, got delta = 1e-200"),
        (-0.5, 0.5, 1.0, "gamma must satisfy `gamma > beta.abs()`, got gamma = 0.5, beta = -0.5"),
    ],
    ids=["zero_delta", "delta_square_underflows", "gamma_at_most_the_skew"],
)
def test_the_hyperbolic_diffusion_refuses_parameters_outside_its_law(beta, gamma, delta, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        sr.PyHyperbolic2(beta, gamma, delta, 0.0, 0.2, 24)


def test_hand_written_process_constructor_raises_value_error():
    with pytest.raises(ValueError, match=r"sigma\.len\(\) must satisfy `sigma\.len\(\) == rho\.nrows\(\)`, got sigma\.len\(\) = 1"):
        sr.PyMultiGbm([0.05, 0.03], [0.2], np.eye(2), 16, [100.0, 100.0])


def test_hand_written_pricer_constructor_raises_value_error():
    with pytest.raises(ValueError, match="is_power_of_two"):
        sr.CarrMadanPricer(n=12)


def test_the_jump_fourier_models_refuse_parameters_outside_their_domain():
    with pytest.raises(ValueError, match=r"eta1 must satisfy `eta1\.is_finite\(\) && eta1 > 1\.0`, got eta1 = 0\.9"):
        sr.HKDEFourier(0.04, 1.5, 0.04, 0.3, -0.7, 0.5, 0.4, 0.9, 5.0)
    with pytest.raises(ValueError, match=r"sigma must satisfy `sigma\.is_finite\(\) && sigma > 0\.0`, got sigma = 0\.0"):
        sr.BatesFourier(0.04, 1.5, 0.04, 0.0, -0.7, 0.5, -0.05, 0.1)


def test_the_heston_family_refuses_a_negative_drift_at_zero_variance():
    negative = r"kappa must satisfy `kappa \* theta >= 0\.0`, got kappa = -1\.5"
    with pytest.raises(ValueError, match=negative):
        sr.HestonFourier(0.04, -1.5, 0.04, 0.3, -0.7)
    with pytest.raises(ValueError, match=negative):
        sr.HestonPricer(100.0, 0.04, 100.0, 0.05, -1.5, 0.04, 0.3, -0.7, 1.0)
    assert sr.HestonFourier(0.04, -1.5, 0.0, 0.3, -0.7).price_call(100.0, 100.0, 0.05, 0.0, 1.0) > 0.0


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


def test_a_swaption_quote_outside_its_domain_names_its_tau():
    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0]), np.full(3, 0.03), interp="log_df")
    calibrator = sr.HullWhiteSwaptionCalibrator([(-1.0, 2.0, 0.2, 0.5, "payer")], curve, initial_rate=0.03, theta=0.03)
    with pytest.raises(ValueError, match=r"quote 0 cannot be used: tau must satisfy `tau\.is_finite\(\) && tau > 0\.0`, got tau = -1\.0$"):
        calibrator.calibrate()


SINGLE_VOL_QUERY = dict(s=100.0, k=105.0, r=0.05, tau=0.75)


@pytest.mark.parametrize(
    "build",
    [
        lambda v: sr.AsianPricer(v=v, **SINGLE_VOL_QUERY),
        lambda v: sr.BjerksundStensland2002Pricer(v=v, **SINGLE_VOL_QUERY),
        lambda v: sr.Merton1976Pricer(v=v, lambda_=1.0, gamma=0.4, **SINGLE_VOL_QUERY),
    ],
)
def test_a_single_vol_pricer_refuses_an_infinite_sigma(build):
    with pytest.raises(ValueError, match=r"sigma must satisfy `sigma\.is_finite\(\) && sigma >= 0\.0`, got sigma = inf$"):
        build(float("inf"))


def test_kirk_checks_each_volatility_and_the_correlation_in_one_wording():
    query = dict(f1=100.0, f2=90.0, x=5.0, r=0.05, tau=0.5)
    with pytest.raises(ValueError, match=r"sigma1 must satisfy `sigma1\.is_finite\(\) && sigma1 >= 0\.0`, got sigma1 = -0\.3$"):
        sr.KirkSpreadPricer(v1=-0.3, v2=0.25, corr=0.7, **query)
    with pytest.raises(ValueError, match=r"sigma2 must satisfy `sigma2\.is_finite\(\) && sigma2 >= 0\.0`, got sigma2 = inf$"):
        sr.KirkSpreadPricer(v1=0.3, v2=float("inf"), corr=0.7, **query)
    with pytest.raises(ValueError, match=r"corr must satisfy `\(-1\.0\.\.=1\.0\)\.contains\(&corr\)`, got corr = 1\.7$"):
        sr.KirkSpreadPricer(v1=0.3, v2=0.25, corr=1.7, **query)
    assert sr.KirkSpreadPricer(v1=0.3, v2=0.25, corr=0.7, **query).price() == pytest.approx(8.547483131757428, rel=1e-12)


def test_the_hull_white_tree_names_a_broken_precondition():
    want = r"mean_reversion must satisfy `mean_reversion\.is_finite\(\) && mean_reversion >= T::zero\(\)`, got mean_reversion = -0\.1"
    with pytest.raises(ValueError, match=want):
        sr.HullWhiteCallableBond(0.04, -0.1, 0.04, 0.01, 3.0, 36)
    with pytest.raises(ValueError, match=r"horizon must satisfy `horizon\.is_finite\(\) && horizon > T::zero\(\)`, got horizon = 0\.0"):
        sr.HullWhiteCallableBond(0.04, 0.3, 0.04, 0.01, 0.0, 36)

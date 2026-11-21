"""The swap exposure's fixed rate: unset for a swap at par, finite otherwise."""

import numpy as np
import pytest
import stochastic_rs as sr


def test_a_swap_exposure_without_a_fixed_rate_is_at_par_and_a_nan_one_raises():
    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 5.0, 10.0]), np.full(4, 0.03), interp="log_df")
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    at_par = sr.HullWhiteSwapExposure(0.1, 0.01, 1e6, times)
    fixed = sr.HullWhiteSwapExposure(0.1, 0.01, 1e6, times, fixed_rate=at_par.par_rate(curve))
    epe = [swap.profile(curve, paths=64, seed=3).epe() for swap in (at_par, fixed)]
    assert np.array_equal(epe[0], epe[1]) and max(epe[0]) > 0.0
    with pytest.raises(ValueError, match="fixed_rate must satisfy `fixed_rate.is_finite\\(\\)`, got fixed_rate = NaN"):
        sr.HullWhiteSwapExposure(0.1, 0.01, 1e6, times, fixed_rate=float("nan"))
    with pytest.raises(ValueError, match="payment_times\\[0\\].is_finite"):
        sr.HullWhiteSwapExposure(0.1, 0.01, 1e6, [float("inf")])

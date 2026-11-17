"""Microstructure bindings against the values their Rust schedules give."""

import pytest
import stochastic_rs as sr

AC_ARGS = (1e6, 5.0, 5, 0.95, 2.5e-7, 2.5e-6, 1e-6)
# The pins go through the platform's sinh and ln: glibc gives values up to 2 ulp (3.4e-16) from
# these macOS ones. abs=0 keeps the terminal zero exact.
LIBM = {"rel": 1e-12, "abs": 0.0}


def test_almgren_chriss_plan_buys_the_sell_schedule_sign_reversed():
    sell = sr.AlmgrenChrissPlan(*AC_ARGS, epsilon=0.0625, direction="sell")
    buy = sr.AlmgrenChrissPlan(*AC_ARGS, epsilon=0.0625, direction="buy")
    inventory = [1000000.0, 541955.5543739223, 289854.2194099351, 147897.48782172316, 62141.801605766035, 0.0]
    trades = [458044.4456260777, 252101.33496398717, 141956.73158821196, 85755.68621595713, 62141.801605766035]
    assert sell.inventory().tolist() == pytest.approx(inventory, **LIBM)
    assert sell.trades().tolist() == pytest.approx(trades, **LIBM)
    assert sell.rates().tolist() == sell.trades().tolist()
    assert buy.inventory().tolist() == [-x for x in sell.inventory().tolist()]
    assert buy.trades().tolist() == buy.rates().tolist() == [-x for x in sell.trades().tolist()]
    moments = (0.6070761632470627, 911226.9863037937, 364128572058.141)
    assert (sell.kappa, sell.expected_cost, sell.variance) == pytest.approx(moments, **LIBM)
    assert (buy.kappa, buy.expected_cost, buy.variance) == (sell.kappa, sell.expected_cost, sell.variance)
    assert sell.risk_adjusted_cost(1e-6) == sell.expected_cost + 1e-6 * sell.variance
    assert sell.risk_adjusted_cost(1e-6) == pytest.approx(1275355.5583619347, **LIBM)
    with pytest.raises(ValueError, match="direction must be 'sell' or 'buy', got 'hold'"):
        sr.AlmgrenChrissPlan(*AC_ARGS, direction="hold")

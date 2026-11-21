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


def test_order_book_answers_with_the_same_tuples():
    book = sr.OrderBook()
    assert book.add_order("sell", 101.0, 5.0) == (1, [])
    assert book.add_order("sell", 102.0, 3.0) == (2, [])
    assert book.add_order("buy", 99.0, 1.0) == (3, [])
    assert book.add_order("buy", 98.5, 2.0) == (4, [])
    assert book.depth() == ([(99.0, 1.0), (98.5, 2.0)], [(101.0, 5.0), (102.0, 3.0)])
    assert (book.best_bid(), book.best_ask()) == ((99.0, 1.0), (101.0, 5.0))
    assert (book.mid(), book.spread()) == (100.0, 2.0)
    assert book.add_order("buy", 101.5, 6.0) == (5, [(101.0, 5.0, 5, 1)])
    assert book.execute_order("sell", 9.0) == (
        6,
        [(101.5, 1.0, 6, 5), (99.0, 1.0, 6, 3), (98.5, 2.0, 6, 4)],
        5.0,
    )
    assert book.execute_order("buy", 100.0) == (7, [(102.0, 3.0, 7, 2)], 97.0)
    assert book.depth() == ([], [])
    assert (book.best_bid(), book.best_ask(), book.mid(), book.spread()) == (None, None, None, None)

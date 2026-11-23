"""Calendar bindings (DayCount, Calendar factories and joint mode, ScheduleBuilder, conventions):
`python_smoke.py::check_calendar_layer` plus factory and convention edge cases."""

from __future__ import annotations

import datetime

import pytest

import stochastic_rs as sr

# par_rate = (1 - D) / annuity: 1 ulp on D (0.16 numerator, 4.4 annuity) moves a 0.035 rate by ~6e-17;
# 1e-14 leaves room for another libm.
PAR_RATE_ABS = 1e-14


def test_daycount_act365f_leap_year():
    dcc = sr.DayCount("Act365F")
    yf = dcc.year_fraction(datetime.date(2024, 1, 1), datetime.date(2025, 1, 1))
    assert abs(yf - 366.0 / 365.0) < 1e-12


def test_daycount_act360():
    dcc = sr.DayCount("Act360")
    yf = dcc.year_fraction(datetime.date(2024, 1, 1), datetime.date(2025, 1, 1))
    assert abs(yf - 366.0 / 360.0) < 1e-12


def test_us_settlement_july4():
    us = sr.Calendar.us_settlement()
    assert us.is_holiday(datetime.date(2024, 7, 4))


def test_target_christmas():
    target = sr.Calendar.target()
    assert target.is_holiday(datetime.date(2024, 12, 25))


def test_calendar_factories_construct():
    for factory in (
        sr.Calendar.us_settlement,
        sr.Calendar.united_kingdom,
        sr.Calendar.target,
        sr.Calendar.tokyo,
    ):
        cal = factory()
        # A regular mid-week non-holiday must be a business day.
        assert cal.is_business_day(datetime.date(2024, 3, 6))


def test_joint_any_holiday_union():
    joint = sr.Calendar.joint(["US", "TARGET"], mode="AnyHoliday")
    assert joint.is_holiday(datetime.date(2024, 7, 4))  # US-only
    assert joint.is_holiday(datetime.date(2024, 5, 1))  # TARGET-only


def test_schedule_builder_semi_annual_2y():
    sb = sr.ScheduleBuilder(datetime.date(2024, 1, 15), datetime.date(2026, 1, 15))
    sb.frequency("SemiAnnual")
    sched = sb.build()
    dates = sched.dates
    assert len(dates) == 5
    assert dates[0] == datetime.date(2024, 1, 15)
    assert dates[-1] == datetime.date(2026, 1, 15)


def test_schedule_builder_quarterly_1y():
    sb = sr.ScheduleBuilder(datetime.date(2024, 1, 15), datetime.date(2025, 1, 15))
    sb.frequency("Quarterly")
    sched = sb.build()
    assert len(sched.dates) == 5


def test_business_day_convention_nearest_sunday():
    us = sr.Calendar.us_settlement()
    bdc = sr.BusinessDayConvention("Nearest")
    sun = datetime.date(2024, 1, 7)
    assert bdc.adjust(sun, us) == datetime.date(2024, 1, 8)


def test_business_day_convention_following():
    us = sr.Calendar.us_settlement()
    bdc = sr.BusinessDayConvention("Following")
    # New Year's Day 2024 (Mon holiday) → next business day Jan 2.
    adj = bdc.adjust(datetime.date(2024, 1, 1), us)
    assert adj == datetime.date(2024, 1, 2)


def _dates(builder, how=None, calendar=None, convention=None):
    if how is not None:
        getattr(builder, how)(*(() if how != "imm" else (True,)))
    if calendar is not None:
        builder.calendar(calendar)
    if convention is not None:
        builder.convention(sr.BusinessDayConvention(convention))
    schedule = builder.build()
    return [str(d) for d in schedule.dates], [str(d) for d in schedule.adjusted_dates]


def _builder(effective, termination, frequency):
    builder = sr.ScheduleBuilder(effective, termination)
    builder.frequency(frequency)
    return builder


def test_schedule_builder_directions_imm_and_calendars_keep_their_dates():
    d = datetime.date
    forward = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15", "2025-01-15", "2025-03-01"]
    assert _dates(_builder(d(2024, 1, 15), d(2025, 3, 1), "Quarterly"), "forward") == (forward, forward)
    backward = ["2024-01-15", "2024-03-01", "2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01"]
    assert _dates(_builder(d(2024, 1, 15), d(2025, 3, 1), "Quarterly"), "backward") == (backward, backward)
    imm = ["2024-03-20", "2024-06-19", "2024-09-18", "2024-12-18", "2025-03-19"]
    assert _dates(_builder(d(2024, 3, 1), d(2025, 3, 1), "Quarterly"), "imm") == (imm, imm)
    imm_semi = ["2024-03-20", "2024-06-19", "2024-12-18", "2025-06-18", "2025-12-17", "2026-06-17"]
    assert _dates(_builder(d(2024, 2, 10), d(2026, 5, 1), "SemiAnnual"), "imm") == (imm_semi, imm_semi)
    target = _dates(_builder(d(2023, 12, 31), d(2025, 12, 31), "SemiAnnual"), None, sr.Calendar.target(), "ModifiedFollowing")
    assert target == (
        ["2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"],
        ["2023-12-29", "2024-06-28", "2024-12-31", "2025-06-30", "2025-12-31"],
    )
    joint = _dates(_builder(d(2024, 1, 1), d(2025, 1, 1), "Quarterly"), "forward", sr.Calendar.joint(["US", "UK"]), "Following")
    assert joint == (
        ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01"],
        ["2024-01-02", "2024-04-02", "2024-07-01", "2024-10-01", "2025-01-02"],
    )


def test_par_rate_takes_one_two_four_or_twelve_payments_a_year():
    import numpy as np

    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 2.0, 5.0, 10.0]), np.array([0.02, 0.025, 0.03, 0.035, 0.04]), interp="log_df")
    assert [curve.par_rate(5.0, f) for f in (1, 2, 4, 12)] == pytest.approx(
        [0.03539979136987701, 0.03508251360718011, 0.034930018735942335, 0.03482883900774336], rel=0, abs=PAR_RATE_ABS
    )
    assert [curve.par_rate(2.5, f) for f in (1, 2, 4, 12)] == pytest.approx(
        [0.033170135130105284, 0.03178438160960795, 0.031659114989185194, 0.031575961456323146], rel=0, abs=PAR_RATE_ABS
    )
    with pytest.raises(ValueError, match="frequency must be 1, 2, 4 or 12 payments a year, got 3"):
        curve.par_rate(5.0, 3)


def test_schedule_builder_keeps_its_direction_under_imm_false_and_refuses_forward_imm():
    d = datetime.date

    def build(*calls):
        builder = _builder(d(2024, 2, 10), d(2026, 5, 1), "SemiAnnual")
        for name, *args in calls:
            getattr(builder, name)(*args)
        return [str(x) for x in builder.build().dates]

    forward = ["2024-02-10", "2024-08-10", "2025-02-10", "2025-08-10", "2026-02-10", "2026-05-01"]
    backward = ["2024-02-10", "2024-05-01", "2024-11-01", "2025-05-01", "2025-11-01", "2026-05-01"]
    imm = ["2024-03-20", "2024-06-19", "2024-12-18", "2025-06-18", "2025-12-17", "2026-06-17"]
    assert build(("forward",), ("imm", False)) == forward
    assert build(("imm", False), ("forward",)) == forward
    assert build(("imm", False)) == backward
    assert build(("imm", True), ("imm", False)) == backward
    assert build(("backward",), ("imm", True)) == imm
    assert build(("imm", True), ("backward",)) == imm
    assert build(("forward",), ("imm", True), ("backward",)) == imm
    for calls in ((("forward",), ("imm", True)), (("imm", True), ("forward",))):
        with pytest.raises(ValueError, match="IMM schedules generate backward"):
            build(*calls)


def test_business_252_counts_its_calendars_business_days_and_icma_needs_a_reference_period():
    import math

    d = datetime.date
    bus = sr.DayCount("BUS/252")
    assert (bus.day_count(d(2026, 1, 5), d(2026, 1, 9)), bus.year_fraction(d(2026, 1, 5), d(2026, 1, 9))) == (4, 4 / 252)
    assert (bus.day_count(d(2026, 4, 20), d(2026, 4, 22)), repr(bus)) == (1, "DayCount(BUS/252/Brazil)")
    assert sr.DayCount("BUS/252/US").day_count(d(2026, 7, 2), d(2026, 7, 6)) == 1
    assert math.isnan(sr.DayCount("ACT/ACT ICMA").year_fraction(d(2026, 1, 1), d(2026, 7, 1)))


def test_business_252_defaults_to_the_brazil_settlement_calendar_of_quantlib():
    d = datetime.date
    # QuantLib 1.43, Business252(Brazil(Settlement)).dayCount.
    counts = [
        (d(2026, 12, 23), d(2027, 1, 4), 6),
        (d(2020, 1, 1), d(2030, 12, 31), 2755),
        (d(2024, 11, 18), d(2024, 11, 22), 3),
        (d(2023, 11, 17), d(2023, 11, 22), 3),
        (d(2025, 2, 28), d(2025, 3, 6), 2),
        (d(2027, 5, 24), d(2027, 5, 31), 4),
        (d(2030, 12, 20), d(2030, 12, 31), 6),
        (d(2022, 1, 3), d(2023, 1, 2), 251),
        (d(2027, 1, 4), d(2026, 12, 23), -6),
    ]
    for name in ("BUS/252", "Bus252", "Business252", "BUS/252/Brazil"):
        bus = sr.DayCount(name)
        assert [bus.day_count(d1, d2) for d1, d2, _ in counts] == [count for _, _, count in counts]
        assert bus.year_fraction(d(2020, 1, 1), d(2030, 12, 31)) == 10.932539682539682
    # B3, the exchange, also closes on 24 and 31 December.
    assert sr.DayCount("BUS/252/B3").day_count(d(2026, 12, 23), d(2027, 1, 4)) == 4
    brazil, b3 = sr.Calendar.brazil(), sr.Calendar.b3()
    assert [brazil.is_holiday(d(2026, 12, day)) for day in (24, 25, 31)] == [False, True, False]
    assert [b3.is_holiday(d(2026, 12, day)) for day in (24, 25, 31)] == [True, True, True]

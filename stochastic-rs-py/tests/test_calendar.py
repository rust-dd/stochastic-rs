"""Calendar-layer pytest coverage.

Exercises the PyO3 calendar bindings (DayCount, Calendar factories +
joint mode, ScheduleBuilder, BusinessDayConvention) mirroring the
verified `python_smoke.py::check_calendar_layer` and extending with
factory + convention edge cases.
"""

from __future__ import annotations

import datetime

import stochastic_rs as sr


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
    import pytest

    curve = sr.DiscountCurve.from_zero_rates(np.array([0.5, 1.0, 2.0, 5.0, 10.0]), np.array([0.02, 0.025, 0.03, 0.035, 0.04]), interp="log_df")
    assert [curve.par_rate(5.0, f) for f in (1, 2, 4, 12)] == [
        0.03539979136987701,
        0.03508251360718011,
        0.03493945015025661,
        0.03484447860149439,
    ]
    assert [curve.par_rate(2.5, f) for f in (1, 2, 4, 12)] == [
        0.033170135130105284,
        0.03178438160960795,
        0.03167546093401026,
        0.03160308615699292,
    ]
    with pytest.raises(ValueError, match="frequency must be 1, 2, 4 or 12 payments a year, got 3"):
        curve.par_rate(5.0, 3)


def test_business_252_counts_its_calendars_business_days_and_icma_needs_a_reference_period():
    import math

    d = datetime.date
    b3 = sr.DayCount("BUS/252")
    assert (b3.day_count(d(2026, 1, 5), d(2026, 1, 9)), b3.year_fraction(d(2026, 1, 5), d(2026, 1, 9))) == (4, 4 / 252)
    assert (b3.day_count(d(2026, 4, 20), d(2026, 4, 22)), repr(b3)) == (1, "DayCount(BUS/252/B3)")
    assert sr.DayCount("BUS/252/US").day_count(d(2026, 7, 2), d(2026, 7, 6)) == 1
    assert math.isnan(sr.DayCount("ACT/ACT ICMA").year_fraction(d(2026, 1, 1), d(2026, 7, 1)))

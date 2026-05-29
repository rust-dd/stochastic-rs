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

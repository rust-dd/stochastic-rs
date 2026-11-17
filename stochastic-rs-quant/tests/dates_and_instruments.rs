//! Market dates and instrument shapes: the schedules the market uses, the enums that name conventions.

use std::sync::Arc;

use chrono::Datelike;
use chrono::Days;
use chrono::NaiveDate;
use chrono::Weekday;
use ndarray::array;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calendar::BusinessDayConvention;
use stochastic_rs_quant::calendar::Calendar;
use stochastic_rs_quant::calendar::CalendarExt;
use stochastic_rs_quant::calendar::DateGenerationRule;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::HolidayCalendar;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::calendar::SharedCalendar;
use stochastic_rs_quant::calendar::date_math::cds_roll_on_or_before;
use stochastic_rs_quant::calendar::date_math::imm_date;
use stochastic_rs_quant::calendar::date_math::next_cds_roll;
use stochastic_rs_quant::calendar::date_math::snap_to_imm;
use stochastic_rs_quant::cashflows::FloatingIndex;
use stochastic_rs_quant::cashflows::IborIndex;
use stochastic_rs_quant::cashflows::OvernightIndex;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::credit::CdsPosition;
use stochastic_rs_quant::credit::CreditDefaultSwap;
use stochastic_rs_quant::credit::HazardInterpolation;
use stochastic_rs_quant::credit::SurvivalCurve;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::MultiCurve;
use stochastic_rs_quant::fx::currency::EUR;
use stochastic_rs_quant::fx::currency::USD;
use stochastic_rs_quant::inflation::InflationIndex;
use stochastic_rs_quant::inflation::PriceIndex;
use stochastic_rs_quant::instruments::CrossCurrencyBasisSwap;
use stochastic_rs_quant::instruments::CrossCurrencySwapDirection;
use stochastic_rs_quant::instruments::CurrencyLeg;
use stochastic_rs_quant::instruments::SwaptionDirection;
use stochastic_rs_quant::instruments::VanillaInterestRateSwap;
use stochastic_rs_quant::instruments::option::cms::hagan_linear_tsr_convexity_factor;
use stochastic_rs_quant::market::Handle;
use stochastic_rs_quant::market::NamedOvernightIndex;
use stochastic_rs_quant::market::Quote;
use stochastic_rs_quant::market::SimpleQuote;
use stochastic_rs_quant::market::SwapRateHelper;
use stochastic_rs_quant::microstructure::AlmgrenChrissParams;
use stochastic_rs_quant::microstructure::optimal_execution;
use stochastic_rs_quant::order_book::Side;
use stochastic_rs_quant::strategies::delta_hedge::DeltaHedge;
use stochastic_rs_quant::strategies::delta_hedge::DeltaHedgeSpec;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat(rate: f64) -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &array![0.5, 1.0, 2.0, 5.0],
    &array![rate, rate, rate, rate],
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn ibor_3m() -> FloatingIndex<f64> {
  FloatingIndex::Ibor(IborIndex::new(
    "IBOR-3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ))
}

fn currency_leg(
  currency: stochastic_rs_quant::fx::Currency,
  notional: f64,
  end: NaiveDate,
) -> CurrencyLeg<f64> {
  CurrencyLeg {
    currency,
    notional,
    index: ibor_3m(),
    spread: 0.0,
    day_count: DayCountConvention::Actual360,
    schedule: ScheduleBuilder::new(d(2024, 1, 15), end)
      .frequency(Frequency::Quarterly)
      .build(),
  }
}

#[test]
fn a_swap_and_a_swaption_share_one_direction_and_an_order_side_drives_execution() {
  let fixed = ScheduleBuilder::new(d(2024, 1, 15), d(2026, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .build();
  let float = ScheduleBuilder::new(d(2024, 1, 15), d(2026, 1, 15))
    .frequency(Frequency::Quarterly)
    .build();
  let mut curves = MultiCurve::new(flat(0.03));
  curves.add_forecast(RateTenor::ThreeMonths, flat(0.03));
  let npv = |direction| {
    VanillaInterestRateSwap::new(
      direction,
      &fixed,
      &float,
      1e6,
      0.04,
      DayCountConvention::Thirty360,
      ibor_3m(),
      0.0,
      DayCountConvention::Actual360,
    )
    .npv(d(2024, 1, 15), DayCountConvention::Actual365Fixed, &curves)
  };
  let (payer, receiver) = (
    npv(SwaptionDirection::Payer),
    npv(SwaptionDirection::Receiver),
  );
  assert!(
    payer < 0.0 && payer == -receiver,
    "payer {payer}, receiver {receiver}"
  );

  let sell = AlmgrenChrissParams::new(Side::Sell, 1.0e6, 1.0, 10, 0.5, 0.0, 0.02, 0.0);
  assert_eq!(sell.side, Side::Sell);
  let buy = AlmgrenChrissParams {
    side: Side::Buy,
    ..sell.clone()
  };
  assert_eq!(
    optimal_execution(&buy).inventory,
    -optimal_execution(&sell).inventory
  );
}

#[test]
fn the_inflation_index_reads_its_prints_from_the_market_fixing_history() {
  let cpi = InflationIndex::<f64>::new(PriceIndex::CpiUsAllItems);
  cpi.fixings.add_fixing(d(2025, 1, 1), 300.0);
  cpi.fixings.add_fixing(d(2025, 2, 1), 303.0);
  assert_eq!(cpi.fixings.get(d(2025, 1, 15)), None);
  assert_eq!(
    cpi.fixings.last_before(d(2025, 1, 20)),
    Some((d(2025, 1, 1), 300.0))
  );
  let ratio = cpi.reference_ratio(d(2025, 1, 16), 300.0).unwrap();
  assert!((ratio - (300.0 + 3.0 * 15.0 / 31.0) / 300.0).abs() < 1e-12);
  assert_eq!(
    cpi.reference_ratio(d(2025, 2, 1), 300.0),
    Some(303.0 / 300.0)
  );
  assert_eq!(cpi.reference_ratio(d(2025, 3, 1), 300.0), None);
}

#[test]
fn a_delta_hedge_and_a_cross_currency_swap_take_spec_structs() {
  let hedge = DeltaHedge::new(DeltaHedgeSpec {
    c: 6.0,
    c_premium: 5.0,
    c_delta: 0.6,
    k: 100.0,
    s: 102.0,
    s0: 100.0,
    contract_size: 100.0,
    option_type: OptionType::Call,
    dotm_threshold: 0.9,
    ditm_threshold: 1.1,
    atm_threshold: 0.99,
    itm_threshold: 1.0,
  });
  assert_eq!(hedge.hedge_shares(), 0.6 * 100.0);
  assert_eq!(
    hedge.current_hedge_profit(),
    (5.0 - 6.0) * 100.0 + 2.0 * 60.0
  );

  let xccy = CrossCurrencyBasisSwap::new(
    CrossCurrencySwapDirection::PayDomesticReceiveForeign,
    1.2,
    currency_leg(USD, 1.2e6, d(2025, 1, 15)),
    currency_leg(EUR, 1.0e6, d(2025, 1, 15)),
    true,
  );
  assert_eq!((xccy.domestic_currency, xccy.foreign_currency), (USD, EUR));
  assert_eq!((xccy.domestic_leg.len(), xccy.foreign_leg.len()), (5, 5));
}

#[test]
#[should_panic(expected = "foreign.schedule must satisfy `adjusted_dates.len() >= 2`")]
fn a_cross_currency_leg_needs_two_schedule_dates() {
  CrossCurrencyBasisSwap::new(
    CrossCurrencySwapDirection::ReceiveDomesticPayForeign,
    1.2,
    currency_leg(USD, 1.2e6, d(2025, 1, 15)),
    currency_leg(EUR, 1.0e6, d(2024, 1, 15)),
    false,
  );
}

fn cds_rolls() -> [NaiveDate; 6] {
  [
    d(2026, 3, 20),
    d(2026, 6, 20),
    d(2026, 9, 20),
    d(2026, 12, 20),
    d(2027, 3, 20),
    d(2027, 6, 20),
  ]
}

#[test]
fn the_cds_rule_accrues_from_the_roll_on_or_before_the_effective_date() {
  let s = ScheduleBuilder::new(d(2026, 5, 10), d(2027, 6, 20))
    .frequency(Frequency::Quarterly)
    .rule(DateGenerationRule::Cds)
    .build();
  assert_eq!(s.dates, cds_rolls());
  assert_eq!(
    [
      d(2026, 5, 10),
      d(2026, 3, 20),
      d(2026, 3, 19),
      d(2026, 1, 5)
    ]
    .map(cds_roll_on_or_before),
    [
      d(2026, 3, 20),
      d(2026, 3, 20),
      d(2025, 12, 20),
      d(2025, 12, 20)
    ]
  );
  assert_eq!(
    [d(2026, 3, 20), d(2025, 12, 25)].map(next_cds_roll),
    [d(2026, 6, 20), d(2026, 3, 20)]
  );
}

#[test]
fn a_standard_cds_pays_on_the_rolls_and_a_vanilla_one_keeps_its_plain_schedule() {
  let standard = CreditDefaultSwap::<f64>::standard(
    CdsPosition::Buyer,
    1.0,
    0.01,
    0.4,
    d(2026, 5, 10),
    d(2027, 6, 20),
    DayCountConvention::Actual360,
  );
  assert_eq!(standard.premium_schedule.dates, cds_rolls());
  assert_eq!(standard.effective_date, d(2026, 5, 10));
  let vanilla = CreditDefaultSwap::<f64>::vanilla(
    CdsPosition::Buyer,
    1.0,
    0.01,
    0.4,
    d(2026, 5, 10),
    d(2027, 6, 20),
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );
  assert_eq!(vanilla.premium_schedule.dates[0], d(2026, 5, 10));
}

#[test]
fn a_standard_cds_and_a_vanilla_one_from_one_step_in_agree_on_the_fair_spread() {
  let survival = SurvivalCurve::from_hazard_rates(
    &array![1.0, 3.0, 5.0, 10.0],
    &array![0.02, 0.02, 0.02, 0.02],
    HazardInterpolation::PiecewiseConstantHazard,
  );
  let (discount, step_in) = (flat(0.03), d(2026, 5, 10));
  for maturity in [d(2027, 6, 20), d(2031, 6, 20)] {
    let standard = CreditDefaultSwap::<f64>::standard(
      CdsPosition::Buyer,
      1.0,
      0.012,
      0.4,
      step_in,
      maturity,
      DayCountConvention::Actual360,
    )
    .valuation(
      step_in,
      DayCountConvention::Actual365Fixed,
      &discount,
      &survival,
    );
    let vanilla = CreditDefaultSwap::<f64>::vanilla(
      CdsPosition::Buyer,
      1.0,
      0.012,
      0.4,
      step_in,
      maturity,
      Frequency::Quarterly,
      DayCountConvention::Actual360,
    )
    .valuation(
      step_in,
      DayCountConvention::Actual365Fixed,
      &discount,
      &survival,
    );
    assert_eq!(standard.fair_spread, vanilla.fair_spread, "{maturity}");
    assert_eq!(standard.risky_annuity, vanilla.risky_annuity, "{maturity}");
    // A flat hazard's par spread is λ(1 − R) = 0.012, less the coupon's discrete-payment bias.
    assert!(
      (standard.fair_spread - 0.012).abs() < 2.5e-4,
      "{}",
      standard.fair_spread
    );
  }
}

#[test]
fn a_cds_skips_a_schedule_period_that_ends_before_its_effective_date() {
  let survival = SurvivalCurve::from_hazard_rates(
    &array![1.0, 3.0, 5.0, 10.0],
    &array![0.02, 0.02, 0.02, 0.02],
    HazardInterpolation::PiecewiseConstantHazard,
  );
  let (discount, valuation_date, effective) = (flat(0.03), d(2026, 5, 1), d(2026, 8, 1));
  let rolls = cds_rolls().to_vec();
  let early = CreditDefaultSwap::<f64>::new(
    CdsPosition::Buyer,
    1.0,
    0.012,
    0.4,
    stochastic_rs_quant::calendar::Schedule {
      dates: rolls.clone(),
      adjusted_dates: rolls,
    },
    DayCountConvention::Actual360,
    effective,
    d(2027, 6, 20),
  )
  .valuation(
    valuation_date,
    DayCountConvention::Actual365Fixed,
    &discount,
    &survival,
  );
  let vanilla = CreditDefaultSwap::<f64>::vanilla(
    CdsPosition::Buyer,
    1.0,
    0.012,
    0.4,
    effective,
    d(2027, 6, 20),
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  )
  .valuation(
    valuation_date,
    DayCountConvention::Actual365Fixed,
    &discount,
    &survival,
  );
  assert_eq!(early.fair_spread, vanilla.fair_spread);
  assert_eq!(early.risky_annuity, vanilla.risky_annuity);
}

#[test]
#[should_panic(
  expected = "frequency must satisfy `frequency.months() % 3 == 0`, got frequency = Monthly"
)]
fn the_cds_rule_steps_by_whole_quarters() {
  ScheduleBuilder::new(d(2026, 5, 10), d(2027, 6, 20))
    .frequency(Frequency::Monthly)
    .rule(DateGenerationRule::Cds)
    .build();
}

#[test]
#[should_panic(
  expected = "termination must satisfy `termination == cds_roll_on_or_before(termination)`"
)]
fn a_standard_cds_matures_on_a_roll_date() {
  CreditDefaultSwap::<f64>::standard(
    CdsPosition::Buyer,
    1.0,
    0.01,
    0.4,
    d(2026, 5, 10),
    d(2027, 6, 21),
    DayCountConvention::Actual360,
  );
}

#[test]
fn the_imm_rule_lands_on_third_wednesdays() {
  assert_eq!(
    [3, 6, 9, 12].map(|m| imm_date(2026, m)),
    [
      d(2026, 3, 18),
      d(2026, 6, 17),
      d(2026, 9, 16),
      d(2026, 12, 16)
    ]
  );
  assert_eq!(snap_to_imm(d(2026, 5, 10)), d(2026, 6, 17));
  let s = ScheduleBuilder::new(d(2026, 3, 1), d(2027, 3, 1))
    .frequency(Frequency::Quarterly)
    .rule(DateGenerationRule::Imm)
    .build();
  assert!(
    s.dates.iter().all(|x| x == &snap_to_imm(*x)),
    "{:?}",
    s.dates
  );
}

#[test]
fn business_252_counts_the_named_calendars_business_days_and_icma_needs_a_reference_period() {
  let bus = DayCountConvention::Business252(HolidayCalendar::B3);
  // 2026-01-05 (Mon) to 2026-01-09 (Fri): four business days, no B3 holiday.
  assert_eq!(bus.day_count(d(2026, 1, 5), d(2026, 1, 9)), 4);
  // 20 to 22 April 2026 spans Tiradentes: one B3 business day, two weekdays.
  assert_eq!(bus.day_count(d(2026, 4, 20), d(2026, 4, 22)), 1);
  assert!((bus.year_fraction::<f64>(d(2026, 1, 5), d(2026, 1, 9)) - 4.0 / 252.0).abs() < 1e-15);
  assert_eq!(bus.to_string(), "BUS/252/B3");
  let icma = DayCountConvention::ActualActualIcma;
  assert!(
    icma
      .year_fraction::<f64>(d(2026, 1, 1), d(2026, 7, 1))
      .is_nan()
  );
  let half = DayCountConvention::year_fraction_icma::<f64>(
    d(2026, 1, 1),
    d(2026, 7, 1),
    d(2026, 1, 1),
    d(2026, 7, 1),
    Frequency::SemiAnnual,
  );
  assert!((half - 0.5).abs() < 1e-15);
  assert_eq!(
    Frequency::from_periods_per_year(4),
    Some(Frequency::Quarterly)
  );
  assert_eq!(Frequency::from_periods_per_year(3), None);
}

/// QuantLib 1.43 `Brazil(Settlement)`: its weekday holidays of 2020 to 2030, as `mmdd` per year.
const BRAZIL_SETTLEMENT_HOLIDAYS: [(i32, &str); 11] = [
  (
    2020,
    "0101 0224 0225 0410 0421 0501 0611 0907 1012 1102 1225",
  ),
  (2021, "0101 0215 0216 0402 0421 0603 0907 1012 1102 1115"),
  (2022, "0228 0301 0415 0421 0616 0907 1012 1102 1115"),
  (
    2023,
    "0220 0221 0407 0421 0501 0608 0907 1012 1102 1115 1225",
  ),
  (2024, "0101 0212 0213 0329 0501 0530 1115 1120 1225"),
  (2025, "0101 0303 0304 0418 0421 0501 0619 1120 1225"),
  (
    2026,
    "0101 0216 0217 0403 0421 0501 0604 0907 1012 1102 1120 1225",
  ),
  (2027, "0101 0208 0209 0326 0421 0527 0907 1012 1102 1115"),
  (
    2028,
    "0228 0229 0414 0421 0501 0615 0907 1012 1102 1115 1120 1225",
  ),
  (
    2029,
    "0101 0212 0213 0330 0501 0531 0907 1012 1102 1115 1120 1225",
  ),
  (2030, "0101 0304 0305 0419 0501 0620 1115 1120 1225"),
];

#[test]
fn the_brazil_settlement_calendar_and_its_business_252_count_match_quantlib() {
  let brazil = Calendar::new(HolidayCalendar::Brazil);
  let expected: Vec<NaiveDate> = BRAZIL_SETTLEMENT_HOLIDAYS
    .iter()
    .flat_map(|&(year, days)| {
      days.split(' ').map(move |x| {
        let mmdd = x.parse::<u32>().unwrap();
        d(year, mmdd / 100, mmdd % 100)
      })
    })
    .collect();
  let closed: Vec<NaiveDate> = d(2020, 1, 1)
    .iter_days()
    .take_while(|x| *x <= d(2030, 12, 31))
    .filter(|x| !brazil.is_weekend(*x) && brazil.is_holiday(*x))
    .collect();
  assert_eq!(closed, expected);
  let bus = DayCountConvention::Business252(HolidayCalendar::Brazil);
  let counts = [
    (d(2026, 12, 23), d(2027, 1, 4), 6),
    (d(2026, 4, 20), d(2026, 4, 22), 1),
    (d(2020, 1, 1), d(2030, 12, 31), 2755),
    (d(2024, 11, 18), d(2024, 11, 22), 3),
    (d(2023, 11, 17), d(2023, 11, 22), 3),
    (d(2025, 2, 28), d(2025, 3, 6), 2),
    (d(2027, 5, 24), d(2027, 5, 31), 4),
    (d(2030, 12, 20), d(2030, 12, 31), 6),
    (d(2022, 1, 3), d(2023, 1, 2), 251),
    (d(2027, 1, 4), d(2026, 12, 23), -6),
  ];
  for (d1, d2, count) in counts {
    assert_eq!(bus.day_count(d1, d2), count, "{d1} to {d2}");
  }
  assert_eq!(
    bus.year_fraction::<f64>(d(2020, 1, 1), d(2030, 12, 31)),
    10.932539682539682
  );
  assert_eq!(bus.to_string(), "BUS/252/Brazil");
  // B3, the exchange, also closes on 24 and 31 December.
  let b3 = DayCountConvention::Business252(HolidayCalendar::B3);
  assert_eq!(b3.day_count(d(2026, 12, 23), d(2027, 1, 4)), 4);
}

#[test]
fn a_par_rate_and_a_cms_convexity_take_a_frequency() {
  let annual = flat(0.03).par_rate(5.0, Frequency::Annual);
  assert!((annual - (0.03_f64.exp() - 1.0)).abs() < 1e-12, "{annual}");
  let lambda = hagan_linear_tsr_convexity_factor::<f64>(0.03, 10.0, Frequency::SemiAnnual);
  assert!(lambda.is_finite() && lambda < 0.0, "{lambda}");
}

#[derive(Debug)]
struct ClosedOnWednesdays;

impl CalendarExt for ClosedOnWednesdays {
  fn is_business_day(&self, date: NaiveDate) -> bool {
    !matches!(date.weekday(), Weekday::Sat | Weekday::Sun | Weekday::Wed)
  }
}

#[test]
fn a_calendar_of_the_callers_plugs_into_schedules_indices_and_the_swap_helper() {
  let closed: SharedCalendar = Arc::new(ClosedOnWednesdays);
  let s = ScheduleBuilder::new(d(2026, 3, 1), d(2027, 3, 1))
    .frequency(Frequency::Quarterly)
    .rule(DateGenerationRule::Imm)
    .calendar(Arc::clone(&closed))
    .convention(BusinessDayConvention::Following)
    .build();
  assert!(
    s.dates
      .iter()
      .zip(&s.adjusted_dates)
      .all(|(imm, adjusted)| *adjusted == *imm + Days::new(1)),
    "{:?}",
    s.adjusted_dates
  );
  let index = NamedOvernightIndex::<f64>::new(
    OvernightIndex::new("XOIS", DayCountConvention::Actual360),
    USD,
    Arc::clone(&closed),
  );
  assert!(!index.calendar.is_business_day(d(2026, 3, 18)));
  let quote: Arc<dyn Quote<f64>> = Arc::new(SimpleQuote::new(0.03));
  let helper = SwapRateHelper::new(
    Handle::new(quote),
    d(2026, 1, 5),
    d(2028, 1, 5),
    Frequency::Quarterly,
    DayCountConvention::Actual365Fixed,
  )
  .with_calendar(closed, BusinessDayConvention::Following);
  let schedule = helper.built_schedule();
  assert!(
    schedule.iter().all(|x| x.weekday() != Weekday::Wed),
    "{schedule:?}"
  );
}

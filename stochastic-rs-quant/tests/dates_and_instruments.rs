//! Market dates and instrument shapes: the schedules the market uses, the enums that name conventions.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::cashflows::FloatingIndex;
use stochastic_rs_quant::cashflows::IborIndex;
use stochastic_rs_quant::cashflows::RateTenor;
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
  curves.add_forecast("3M", flat(0.03));
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

//! A bond price that no yield or spread reproduces inverts to `NaN`, never a
//! finite stand-in.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::cashflows::NotionalSchedule;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::AmortizingFixedRateBond;
use stochastic_rs_quant::instruments::FixedRateBond;
use stochastic_rs_quant::instruments::ZeroCouponBond;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn bond() -> FixedRateBond<f64> {
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2027, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  FixedRateBond::new(
    &schedule,
    100.0,
    0.05,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  )
}

fn flat_curve(rate: f64) -> DiscountCurve<f64> {
  let times = array![0.0, 1.0, 2.0, 3.0, 4.0];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

#[test]
fn yield_of_an_impossible_price_is_nan() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let dcc = DayCountConvention::Actual365Fixed;
  // 1e40 and 1e-5 lie beyond the prices at both ends of the search bracket. A
  // merely absurd price such as 1e12 still has a yield (near -197%) and keeps it.
  for price in [f64::NAN, 0.0, -5.0, f64::INFINITY, 1.0e40, 1.0e-5] {
    let y = b.yield_to_maturity_from_dirty_price(d(2024, 4, 15), price, dcc, comp);
    assert!(y.is_nan(), "price {price} gave yield {y}");
  }
}

#[test]
fn yield_still_inverts_a_fair_price() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let dcc = DayCountConvention::Actual365Fixed;
  let price = b.dirty_price_from_yield(d(2024, 4, 15), 0.04, dcc, comp);
  let y = b.yield_to_maturity_from_dirty_price(d(2024, 4, 15), price, dcc, comp);
  assert!((y - 0.04).abs() < 1e-10);
}

#[test]
fn z_spread_of_an_impossible_price_is_nan() {
  let b = bond();
  let curve = flat_curve(0.04);
  for price in [f64::NAN, 0.0, -5.0, f64::INFINITY] {
    let s = b.z_spread_from_dirty_price(
      d(2024, 4, 15),
      price,
      DayCountConvention::Actual365Fixed,
      &curve,
    );
    assert!(s.is_nan(), "price {price} gave spread {s}");
  }
}

#[test]
fn option_adjusted_spread_without_a_solution_is_nan() {
  let b = bond();
  let curve = flat_curve(0.04);
  let dcc = DayCountConvention::Actual365Fixed;
  // A price of 100 against an option worth -150 asks for a model price of -50,
  // which no spread produces; a NaN option value leaves no target at all.
  for option_value in [-150.0, f64::NAN] {
    let s =
      b.option_adjusted_spread_from_dirty_price(d(2024, 4, 15), 100.0, dcc, &curve, option_value);
    assert!(s.is_nan(), "option value {option_value} gave spread {s}");
  }
}

#[test]
fn clean_price_wrappers_apply_the_same_policy() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let dcc = DayCountConvention::Actual365Fixed;
  let curve = flat_curve(0.04);
  let settlement = d(2024, 4, 15);
  let accrued = b.accrued_interest(settlement);
  assert!(accrued > 0.0);
  // A clean price of minus the accrued interest is a dirty price of zero.
  for clean in [f64::NAN, -accrued] {
    let y = b.yield_to_maturity_from_clean_price(settlement, clean, dcc, comp);
    assert!(y.is_nan(), "clean price {clean} gave yield {y}");
    let s = b.z_spread_from_clean_price(settlement, clean, dcc, &curve);
    assert!(s.is_nan(), "clean price {clean} gave spread {s}");
    let oas = b.option_adjusted_spread_from_clean_price(settlement, clean, dcc, &curve, 0.0);
    assert!(oas.is_nan(), "clean price {clean} gave OAS {oas}");
  }
}

#[test]
fn amortizing_bond_z_spread_of_an_impossible_price_is_nan() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2025, 1, 15))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let bond = AmortizingFixedRateBond::new(
    &schedule,
    NotionalSchedule::from_array(array![100.0, 80.0, 60.0, 40.0]),
    0.05,
    Frequency::Quarterly,
    DayCountConvention::Actual365Fixed,
  );
  let curve = flat_curve(0.04);
  for price in [f64::NAN, 0.0, -5.0, f64::INFINITY] {
    let s = bond.z_spread_from_dirty_price(
      d(2024, 4, 15),
      price,
      DayCountConvention::Actual365Fixed,
      &curve,
    );
    assert!(s.is_nan(), "price {price} gave spread {s}");
  }
}

#[test]
fn analytics_of_an_impossible_price_are_nan() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let a = b.analytics_from_clean_price(
    d(2024, 4, 15),
    f64::NAN,
    DayCountConvention::Actual365Fixed,
    comp,
  );
  assert!(a.yield_to_maturity.is_nan());
  assert!(a.macaulay_duration.is_nan());
  assert!(a.modified_duration.is_nan());
  assert!(a.convexity.is_nan());
}

#[test]
fn analytics_still_report_a_fair_price() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let dcc = DayCountConvention::Actual365Fixed;
  let clean = b.clean_price_from_yield(d(2024, 4, 15), 0.04, dcc, comp);
  let a = b.analytics_from_clean_price(d(2024, 4, 15), clean, dcc, comp);
  assert!((a.yield_to_maturity - 0.04).abs() < 1e-10);
  assert!(a.macaulay_duration > 0.0 && a.macaulay_duration < 3.0);
  assert!(a.modified_duration > 0.0);
  assert!(a.convexity > 0.0);
}

const COMPOUNDINGS: [Compounding; 3] = [
  Compounding::Continuous,
  Compounding::Simple,
  Compounding::Periodic(2),
];

#[test]
fn zero_coupon_yield_of_an_impossible_price_is_nan() {
  let zcb = ZeroCouponBond::<f64>::new(100.0, d(2027, 1, 15));
  let dcc = DayCountConvention::Actual365Fixed;
  for compounding in COMPOUNDINGS {
    for price in [f64::NAN, 0.0, -5.0, f64::INFINITY] {
      let y = zcb.yield_to_maturity(d(2024, 4, 15), price, dcc, compounding);
      assert!(y.is_nan(), "{compounding} price {price} gave yield {y}");
    }
  }
}

#[test]
fn zero_coupon_yield_of_a_matured_bond_is_nan() {
  let zcb = ZeroCouponBond::<f64>::new(100.0, d(2027, 1, 15));
  let dcc = DayCountConvention::Actual365Fixed;
  for compounding in COMPOUNDINGS {
    for settlement in [d(2027, 1, 15), d(2027, 6, 1)] {
      let y = zcb.yield_to_maturity(settlement, 100.0, dcc, compounding);
      assert!(
        y.is_nan(),
        "{compounding} settlement {settlement} gave yield {y}"
      );
    }
  }
}

#[test]
fn zero_coupon_yield_still_inverts_a_fair_price() {
  let zcb = ZeroCouponBond::<f64>::new(100.0, d(2027, 1, 15));
  let dcc = DayCountConvention::Actual365Fixed;
  for compounding in COMPOUNDINGS {
    let price = zcb.price_from_yield(d(2024, 4, 15), 0.04, dcc, compounding);
    let y = zcb.yield_to_maturity(d(2024, 4, 15), price, dcc, compounding);
    assert!((y - 0.04).abs() < 1e-12, "{compounding} gave yield {y}");
  }
}

#[test]
fn duration_and_convexity_of_a_matured_bond_are_nan() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let dcc = DayCountConvention::Actual365Fixed;
  let settlement = d(2027, 6, 1);
  let macaulay = b.macaulay_duration(settlement, 0.04, dcc, comp);
  assert!(macaulay.is_nan(), "macaulay duration {macaulay}");
  let modified = b.modified_duration(settlement, 0.04, dcc, comp);
  assert!(modified.is_nan(), "modified duration {modified}");
  let convexity = b.convexity(settlement, 0.04, dcc, comp);
  assert!(convexity.is_nan(), "convexity {convexity}");
}

#[test]
fn duration_and_convexity_at_a_non_finite_yield_are_nan() {
  let b = bond();
  let dcc = DayCountConvention::Actual365Fixed;
  for compounding in COMPOUNDINGS {
    for y in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let macaulay = b.macaulay_duration(d(2024, 4, 15), y, dcc, compounding);
      assert!(
        macaulay.is_nan(),
        "{compounding} yield {y} gave macaulay duration {macaulay}"
      );
      let modified = b.modified_duration(d(2024, 4, 15), y, dcc, compounding);
      assert!(
        modified.is_nan(),
        "{compounding} yield {y} gave modified duration {modified}"
      );
      let convexity = b.convexity(d(2024, 4, 15), y, dcc, compounding);
      assert!(
        convexity.is_nan(),
        "{compounding} yield {y} gave convexity {convexity}"
      );
    }
  }
}

#[test]
fn analytics_of_a_matured_bond_are_nan() {
  let b = bond();
  let comp = b.standard_yield_compounding();
  let a = b.analytics_from_clean_price(
    d(2027, 6, 1),
    100.0,
    DayCountConvention::Actual365Fixed,
    comp,
  );
  assert!(a.yield_to_maturity.is_nan());
  assert!(a.macaulay_duration.is_nan());
  assert!(a.modified_duration.is_nan());
  assert!(a.convexity.is_nan());
}

#[test]
fn asset_swap_spread_of_a_matured_bond_is_nan() {
  let b = bond();
  let curve = flat_curve(0.04);
  let dcc = DayCountConvention::Actual365Fixed;
  let settlement = d(2027, 6, 1);
  let dirty = b.asset_swap_spread_from_dirty_price(settlement, 100.0, dcc, &curve);
  assert!(dirty.is_nan(), "dirty price spread {dirty}");
  let clean = b.asset_swap_spread_from_clean_price(settlement, 100.0, dcc, &curve);
  assert!(clean.is_nan(), "clean price spread {clean}");
}

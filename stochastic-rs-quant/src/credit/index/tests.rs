//! Index identities: a homogeneous index is its single name, the fair spread
//! is bracketed by the names', and the ISDA upfront is zero at the coupon.

use chrono::NaiveDate;
use ndarray::Array1;

use super::*;
use crate::curves::DiscountCurve;
use crate::curves::InterpolationMethod;

fn discount() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

#[test]
fn homogeneous_index_is_its_single_name() {
  let (start, end) = (date(2026, 3, 20), date(2031, 6, 20));
  let survivals = vec![flat_survival(0.02); 5];
  let index = CdsIndex::homogeneous(survivals, 0.4, 0.01, 10_000_000.0, start, end);
  let single = CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    1.0,
    0.01,
    0.4,
    start,
    end,
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );
  let d = discount();
  let sv = single.valuation(
    start,
    DayCountConvention::Actual365Fixed,
    &d,
    &flat_survival(0.02),
  );
  let iv = index.valuation(start, &d);
  assert!((iv.fair_spread - sv.fair_spread).abs() < 1e-12);
  assert!((iv.net_npv - 10_000_000.0 * sv.net_npv).abs() < 1e-6);
  assert!(iv.risky_annuity > 0.0 && iv.protection_leg_npv > 0.0);
}

#[test]
fn heterogeneous_fair_spread_is_bracketed_by_the_names() {
  let (start, end) = (date(2026, 3, 20), date(2031, 6, 20));
  let d = discount();
  let hazards = [0.005, 0.02, 0.08];
  let names: Vec<IndexName> = hazards
    .iter()
    .map(|&h| IndexName {
      weight: 1.0 / 3.0,
      recovery: 0.4,
      survival: flat_survival(h),
    })
    .collect();
  let index = CdsIndex::new(names, 0.01, 1.0, start, end);
  let fair = index.fair_spread(start, &d);
  let single = |h: f64| {
    CreditDefaultSwap::vanilla(
      CdsPosition::Buyer,
      1.0,
      0.01,
      0.4,
      start,
      end,
      Frequency::Quarterly,
      DayCountConvention::Actual360,
    )
    .valuation(
      start,
      DayCountConvention::Actual365Fixed,
      &d,
      &flat_survival(h),
    )
    .fair_spread
  };
  assert!(fair > single(0.005) && fair < single(0.08), "fair {fair}");
}

#[test]
fn isda_upfront_is_zero_at_the_coupon_and_monotone_in_the_quote() {
  let (start, end) = (date(2026, 3, 20), date(2031, 6, 20));
  let d = discount();
  let index = CdsIndex::homogeneous(
    vec![flat_survival(0.02); 3],
    0.4,
    0.01,
    1_000_000.0,
    start,
    end,
  );
  let at_coupon = index.isda_upfront(start, &d, 0.01, 0.4);
  assert!(at_coupon.abs() < 1e-3, "upfront at the coupon {at_coupon}");
  let wide = index.isda_upfront(start, &d, 0.02, 0.4);
  let tight = index.isda_upfront(start, &d, 0.005, 0.4);
  assert!(wide > 0.0 && tight < 0.0 && wide > -tight);
  let hazard = index.flat_hazard_for(start, &d, 0.02, 0.4);
  assert!(
    (hazard - 0.02 / 0.6).abs() < 0.2 * hazard,
    "flat hazard {hazard} vs the credit-triangle 0.033"
  );
}

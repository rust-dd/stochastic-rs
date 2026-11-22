//! Brent's `zero` under quant's solvers: a yield within its bound of mpmath's root, and a CDS quote
//! no hazard rate reprices as an error at the caller's index, never the initial guess.

use chrono::NaiveDate;
use ndarray::Array1;
use stochastic_rs_distributions::brent::MACHEPS;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calendar::DateGenerationRule;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::calendar::ScheduleBuilder;
use stochastic_rs_quant::credit::CdsQuote;
use stochastic_rs_quant::credit::bootstrap_hazard;
use stochastic_rs_quant::curves::Compounding;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::FixedRateBond;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]).view(),
    Array1::from_vec(vec![0.03; 4]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn quote(years: i32, recovery_rate: Option<f64>) -> CdsQuote {
  CdsQuote {
    maturity: d(2026 + years, 6, 20),
    spread: 0.005,
    recovery_rate,
    frequency: Frequency::Quarterly,
    premium_day_count: DayCountConvention::Actual360,
  }
}

#[test]
fn a_yield_lands_within_brents_bound_of_the_mpmath_root() {
  let schedule = ScheduleBuilder::new(d(2026, 6, 20), d(2036, 6, 20))
    .frequency(Frequency::Annual)
    .rule(DateGenerationRule::Forward)
    .build();
  let bond = FixedRateBond::<f64>::new(
    &schedule,
    100.0,
    0.05,
    Frequency::Annual,
    DayCountConvention::Thirty360,
  );
  let y = bond.yield_to_maturity_from_dirty_price(
    d(2026, 6, 20),
    103.0,
    DayCountConvention::Thirty360,
    Compounding::Periodic(1),
  );
  // mpmath's root of the ten annual coupons of 5 and the face of 100 priced at 103.
  let exact = 0.046_186_485_721_353_86;
  assert!((y - exact).abs() <= 6.0 * MACHEPS * y + 2e-12, "{y}");
}

#[test]
fn a_cds_spread_no_hazard_reprices_is_an_invalid_quote_at_the_callers_index() {
  // At a recovery of 0.999999 the protection leg is worth almost nothing: no hazard reaches 50 bp.
  let quotes = [quote(5, None), quote(1, Some(0.999_999))];
  let valuation = d(2026, 6, 20);
  let dcc = DayCountConvention::Actual365Fixed;
  let err = bootstrap_hazard(valuation, valuation, &quotes, 0.4, &curve(), dcc).unwrap_err();
  assert!(
    matches!(err, QuantError::InvalidQuote { index: 1, .. }),
    "{err}"
  );
}

#[test]
fn no_cds_quote_is_insufficient_data() {
  let valuation = d(2026, 6, 20);
  let dcc = DayCountConvention::Actual365Fixed;
  let err = bootstrap_hazard(valuation, valuation, &[], 0.4, &curve(), dcc).unwrap_err();
  let expected = QuantError::InsufficientData {
    what: "cds quotes",
    needed: 1,
    got: 0,
  };
  assert_eq!(err, expected);
}

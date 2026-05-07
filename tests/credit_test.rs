//! Integration tests for the credit module.
//!
//! Validates:
//! - Merton structural model against closed-form reference values.
//! - Survival curve against the hazard-rate → survival identity.
//! - CDS valuation against a flat-hazard closed-form approximation.
//! - Hazard bootstrap repricing consistency (each CDS reprices to par).
//! - Continuous-time generator → transition matrix via matrix exponential
//!   against a 2-state analytic solution, and Moody's S&P-style one-year
//!   transition matrix invariants.

use chrono::NaiveDate;
use ndarray::Array1;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::credit::CdsPosition;
use stochastic_rs::quant::credit::CdsQuote;
use stochastic_rs::quant::credit::CreditDefaultSwap;
use stochastic_rs::quant::credit::GeneratorMatrix;
use stochastic_rs::quant::credit::HazardInterpolation;
use stochastic_rs::quant::credit::HazardRateCurve;
use stochastic_rs::quant::credit::MertonStructural;
use stochastic_rs::quant::credit::SurvivalCurve;
use stochastic_rs::quant::credit::TransitionMatrix;
use stochastic_rs::quant::credit::bootstrap_hazard;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat_discount_curve(rate: f64) -> DiscountCurve<f64> {
  let times = array![0.25_f64, 1.0, 3.0, 5.0, 10.0, 30.0];
  let rates = array![rate, rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

#[test]
fn merton_matches_closed_form_reference() {
  let merton = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);

  let pd = merton.risk_neutral_default_probability(1.0);
  let dd = merton.distance_to_default(1.0);
  let equity = merton.equity_value(1.0);
  let debt = merton.debt_value(1.0);
  let spread = merton.credit_spread(1.0);

  assert!((dd - 0.76048).abs() < 1e-4, "DD = {dd}");
  assert!((pd - 0.22353).abs() < 1e-4, "PD = {pd}");
  assert!((equity - 26.459).abs() < 1e-2, "E0 = {equity}");
  assert!((debt - 73.541).abs() < 1e-2, "B0 = {debt}");
  assert!((spread - 0.03417).abs() < 1e-3, "spread = {spread}");
}

#[test]
fn merton_leverage_zero_produces_riskfree_debt() {
  let merton = MertonStructural::new(1_000.0, 50.0, 0.2, 0.04, 0.0);
  let spread = merton.credit_spread(1.0);
  assert!(
    spread.abs() < 1e-4,
    "low-leverage spread too high: {spread}"
  );
  assert!(merton.risk_neutral_default_probability(1.0) < 1e-12);
}

#[test]
fn survival_curve_reconstructs_hazard_input() {
  let times = array![1.0_f64, 3.0, 5.0];
  let hazards = array![0.01_f64, 0.02, 0.03];
  let curve = SurvivalCurve::from_hazard_rates(
    &times,
    &hazards,
    HazardInterpolation::PiecewiseConstantHazard,
  );

  assert!((curve.survival_probability(1.0) - (-0.01_f64).exp()).abs() < 1e-12);
  let q3_expected = (-(0.01 + 0.02 * 2.0_f64)).exp();
  assert!((curve.survival_probability(3.0) - q3_expected).abs() < 1e-12);
  let q5_expected = (-(0.01 + 0.02 * 2.0_f64 + 0.03 * 2.0_f64)).exp();
  assert!((curve.survival_probability(5.0) - q5_expected).abs() < 1e-12);

  let h13 = curve.forward_hazard(1.0, 3.0);
  assert!((h13 - 0.02).abs() < 1e-12, "forward hazard = {h13}");
}

#[test]
fn hazard_rate_curve_matches_underlying_survival() {
  let times = array![1.0_f64, 3.0];
  let hazards = array![0.02_f64, 0.02];
  let hcurve = HazardRateCurve::from_hazard_rates(
    &times,
    &hazards,
    HazardInterpolation::PiecewiseConstantHazard,
  );
  assert!((hcurve.average_hazard(1.0) - 0.02).abs() < 1e-12);
  assert!((hcurve.forward_hazard(1.0, 3.0) - 0.02).abs() < 1e-12);
  assert!((hcurve.survival_probability(3.0) - (-0.06_f64).exp()).abs() < 1e-12);
}

#[test]
fn cds_fair_spread_reproduces_hazard_times_loss_given_default() {
  // With r ≈ 0, flat hazard h, and recovery R, the ISDA-style fair spread
  // converges to h*(1-R) as the premium frequency → 0 and δ → 0.
  let val_date = d(2025, 1, 1);
  let maturity = d(2030, 1, 1);

  let discount = flat_discount_curve(0.0);
  let hazard = 0.02_f64;
  let times = Array1::from_vec(vec![5.0_f64]);
  let hazards = Array1::from_vec(vec![hazard]);
  let survival = SurvivalCurve::from_hazard_rates(
    &times,
    &hazards,
    HazardInterpolation::PiecewiseConstantHazard,
  );

  let cds = CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    1_000_000.0,
    0.012,
    0.4,
    val_date,
    maturity,
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );

  let v = cds.valuation(
    val_date,
    DayCountConvention::Actual365Fixed,
    &discount,
    &survival,
  );
  let fair = v.fair_spread;
  let lgd_hazard = hazard * (1.0 - 0.4);
  assert!(
    (fair - lgd_hazard).abs() < 1.5e-3,
    "fair spread = {fair}, expected ≈ {lgd_hazard}"
  );
  assert!(v.protection_leg_npv > 0.0);
  assert!(v.premium_leg_npv > 0.0);
}

#[test]
fn cds_at_par_spread_has_zero_npv() {
  let val_date = d(2025, 1, 1);
  let maturity = d(2030, 1, 1);

  let discount = flat_discount_curve(0.03);
  let hazard = 0.015_f64;
  let times = Array1::from_vec(vec![5.0_f64]);
  let hazards = Array1::from_vec(vec![hazard]);
  let survival = SurvivalCurve::from_hazard_rates(
    &times,
    &hazards,
    HazardInterpolation::PiecewiseConstantHazard,
  );

  let probe = CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    1.0,
    0.01,
    0.4,
    val_date,
    maturity,
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );

  let probe_val = probe.valuation(
    val_date,
    DayCountConvention::Actual365Fixed,
    &discount,
    &survival,
  );
  let par = probe_val.fair_spread;

  let atpar_cds = CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    10_000_000.0,
    par,
    0.4,
    val_date,
    maturity,
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );
  let v = atpar_cds.valuation(
    val_date,
    DayCountConvention::Actual365Fixed,
    &discount,
    &survival,
  );
  assert!(v.net_npv.abs() < 1e-4, "at-par NPV = {}", v.net_npv);
}

#[test]
fn bootstrap_hazard_reprices_quotes_at_par() {
  let val_date = d(2025, 1, 1);
  let discount = flat_discount_curve(0.03);

  let quotes = vec![
    CdsQuote::isda(d(2026, 1, 1), 0.0080),
    CdsQuote::isda(d(2028, 1, 1), 0.0105),
    CdsQuote::isda(d(2030, 1, 1), 0.0130),
    CdsQuote::isda(d(2032, 1, 1), 0.0150),
    CdsQuote::isda(d(2035, 1, 1), 0.0175),
  ];

  let recovery = 0.4;
  let survival = bootstrap_hazard(
    val_date,
    val_date,
    &quotes,
    recovery,
    &discount,
    DayCountConvention::Actual365Fixed,
  );

  for quote in &quotes {
    let cds = CreditDefaultSwap::vanilla(
      CdsPosition::Buyer,
      1.0,
      quote.spread,
      recovery,
      val_date,
      quote.maturity,
      quote.frequency,
      quote.premium_day_count,
    );
    let v = cds.valuation(
      val_date,
      DayCountConvention::Actual365Fixed,
      &discount,
      &survival,
    );
    assert!(
      (v.fair_spread - quote.spread).abs() < 1e-6,
      "bootstrap fair {} vs quote {} at {:?}",
      v.fair_spread,
      quote.spread,
      quote.maturity
    );
    assert!(v.net_npv.abs() < 1e-6, "residual NPV = {}", v.net_npv);
  }

  // Survival probabilities must be strictly decreasing with maturity.
  let q1 = survival.survival_probability(1.0);
  let q3 = survival.survival_probability(3.0);
  let q5 = survival.survival_probability(5.0);
  let q10 = survival.survival_probability(10.0);
  assert!(1.0 > q1 && q1 > q3 && q3 > q5 && q5 > q10 && q10 > 0.0);
}

#[test]
fn generator_matrix_two_state_analytic() {
  let lambda = 0.03_f64;
  let q = ndarray::arr2(&[[-lambda, lambda], [0.0, 0.0]]);
  let gen_ = GeneratorMatrix::new(q);
  gen_.check_generator(1e-12).expect("valid generator");

  let p = gen_.transition_at(2.0);
  let m = p.matrix();
  assert!((m[[0, 0]] - (-2.0 * lambda).exp()).abs() < 1e-10);
  assert!((m[[0, 1]] - (1.0 - (-2.0 * lambda).exp())).abs() < 1e-10);
  assert!((m[[1, 0]] - 0.0).abs() < 1e-10);
  assert!((m[[1, 1]] - 1.0).abs() < 1e-10);
  p.check_row_stochastic(1e-10)
    .expect("row-stochastic output");
}

#[test]
fn transition_matrix_power_default_probability_monotone() {
  // Small illustrative 3-state matrix (A, B, Default).
  let m = ndarray::arr2(&[
    [0.90_f64, 0.09, 0.01],
    [0.10, 0.85, 0.05],
    [0.00, 0.00, 1.00],
  ]);
  let tm = TransitionMatrix::new(m);
  tm.check_row_stochastic(1e-12).expect("row stochastic");

  let p1 = tm.default_probabilities(1);
  let p5 = tm.default_probabilities(5);
  let p10 = tm.default_probabilities(10);

  for i in 0..2 {
    assert!(p1[i] <= p5[i] + 1e-12);
    assert!(p5[i] <= p10[i] + 1e-12);
  }
  // Default state remains at probability 1 of being defaulted.
  assert!((p1[2] - 1.0).abs() < 1e-12);
  assert!((p10[2] - 1.0).abs() < 1e-12);
}

#[test]
fn generator_from_yearly_transition_is_close_to_jlt_embedding() {
  // Small 3-state matrix with weak off-diagonal mass — suitable for JLT.
  let p = ndarray::arr2(&[
    [0.95_f64, 0.04, 0.01],
    [0.06, 0.90, 0.04],
    [0.00, 0.00, 1.00],
  ]);
  let tm = TransitionMatrix::new(p.clone());
  let q_hat = GeneratorMatrix::from_yearly_transition(&tm);
  let q_proj = q_hat.project_to_generator();
  q_proj
    .check_generator(1e-9)
    .expect("projected generator is valid");

  let p1 = q_proj.transition_at(1.0);
  // Reconstructed one-year probabilities are within 0.01 of the input.
  for i in 0..3 {
    for j in 0..3 {
      let diff = (p1.matrix()[[i, j]] - p[[i, j]]).abs();
      assert!(diff < 1e-2, "P({i},{j}) mismatch = {diff}");
    }
  }
}

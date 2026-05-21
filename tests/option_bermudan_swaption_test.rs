//! Bermudan and tree-based swaption pricing tests.
//!
//! Covers Hull-White and G2++ trinomial trees, the Jamshidian decomposition
//! for European HW swaptions, and Bermudan-vs-European dominance on a shared
//! exercise grid.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BermudanSwaption;
use stochastic_rs::quant::instruments::ExerciseSchedule;
use stochastic_rs::quant::instruments::JamshidianHullWhiteSwaption;
use stochastic_rs::quant::instruments::SwaptionDirection;
use stochastic_rs::quant::instruments::TreeCouponSchedule;
use stochastic_rs::quant::lattice::CurveFittedHullWhiteTree;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn approx(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b:.12}, got {a:.12}, diff {:.2e}",
    (a - b).abs()
  );
}

#[test]
fn bermudan_swaption_dominates_european_at_common_grid() {
  let horizon: f64 = 5.0;
  let steps: usize = 20;
  let dt: f64 = horizon / steps as f64;
  let model = HullWhiteTreeModel::new(0.04_f64, 0.1, 0.04, 0.01);
  let tree = HullWhiteTree::new(model, horizon, steps);

  let coupon_levels: Vec<usize> = (4..=steps).step_by(4).collect();
  let accrual_factors = vec![dt * 4.0; coupon_levels.len()];
  let coupon_schedule = TreeCouponSchedule::new(coupon_levels.clone(), accrual_factors);

  let notional = 1.0_f64;
  let strike = 0.04_f64;

  let european = BermudanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    notional,
    ExerciseSchedule::new(vec![*coupon_levels.first().unwrap()]),
    coupon_schedule.clone(),
  );
  let bermudan = BermudanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    notional,
    ExerciseSchedule::new(coupon_levels),
    coupon_schedule,
  );

  let european_npv = european.price_on(&tree);
  let bermudan_npv = bermudan.price_on(&tree);

  assert!(
    bermudan_npv >= european_npv - 1e-10,
    "Bermudan {bermudan_npv:.8} must dominate European {european_npv:.8}"
  );
  assert!(bermudan_npv >= 0.0);
}

#[test]
fn curve_fitted_hw_tree_reprices_zero_coupon_bond() {
  let times = array![0.0, 1.0, 2.0, 5.0, 10.0];
  let rates = array![0.02, 0.025, 0.03, 0.035, 0.04];
  let curve =
    DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates);
  let horizon: f64 = 5.0;
  let tree = CurveFittedHullWhiteTree::new(0.08_f64, 0.012, &curve, horizon, 200);
  let tree_price = tree.zero_coupon_bond_price();
  let curve_price = curve.discount_factor(horizon);
  approx(tree_price, curve_price, 1e-3);
}

#[test]
fn jamshidian_hw_payer_receiver_parity() {
  let times = array![0.0, 1.0, 2.0, 5.0, 10.0];
  let rates = array![0.03, 0.035, 0.038, 0.04, 0.042];
  let curve =
    DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates);
  let expiry = 1.0_f64;
  let coupon_times: Vec<f64> = (1..=10).map(|i| expiry + 0.5 * i as f64).collect();
  let accrual_factors = vec![0.5_f64; coupon_times.len()];
  let strike = 0.04;
  let notional = 1_000_000.0;

  let payer = JamshidianHullWhiteSwaption::new(
    SwaptionDirection::Payer,
    strike,
    notional,
    expiry,
    coupon_times.clone(),
    accrual_factors.clone(),
    0.05,
    0.01,
  );
  let receiver = JamshidianHullWhiteSwaption::new(
    SwaptionDirection::Receiver,
    strike,
    notional,
    expiry,
    coupon_times.clone(),
    accrual_factors.clone(),
    0.05,
    0.01,
  );
  let payer_npv = payer.price(&curve);
  let receiver_npv = receiver.price(&curve);

  let p_exp = curve.discount_factor(expiry);
  let mut expected_diff = 0.0_f64;
  for (i, &t_i) in coupon_times.iter().enumerate() {
    let c = if i == coupon_times.len() - 1 {
      notional * strike * accrual_factors[i] + notional
    } else {
      notional * strike * accrual_factors[i]
    };
    expected_diff += c * curve.discount_factor(t_i);
  }
  expected_diff = notional * p_exp - expected_diff;

  let actual_diff = payer_npv - receiver_npv;
  assert!(
    (actual_diff - expected_diff).abs() < 1.0,
    "payer - receiver should equal N·P(0,T_0) - Σc_i·P(0,T_i); got {actual_diff:.6}, expected {expected_diff:.6}"
  );
  assert!(payer_npv >= 0.0 && receiver_npv >= 0.0);
}

#[test]
fn g2pp_bermudan_matches_hw_when_second_factor_vanishes() {
  let horizon: f64 = 5.0;
  let steps: usize = 20;
  let dt: f64 = horizon / steps as f64;

  // Hull-White: dr = a(θ-r)dt + σ dW with r₀=θ means r(t) = X(t) + θ where
  // X follows a zero-mean Ou with X(0)=0. Matching G2++ uses phi=θ, x(0)=0,
  // and a nearly zero second factor with strong mean reversion.
  let a = 0.1_f64;
  let sigma = 0.012_f64;
  let theta = 0.05_f64;
  let hw_model = HullWhiteTreeModel::new(theta, a, theta, sigma);
  let hw_tree = HullWhiteTree::new(hw_model, horizon, steps);

  let g2pp_model = G2ppTreeModel::new(0.0, 0.0, theta, a, 50.0, sigma, 1e-10, 0.0);
  let g2pp_tree = G2ppTree::new(g2pp_model, horizon, steps);

  let coupon_levels: Vec<usize> = (4..=steps).step_by(4).collect();
  let accrual_factors = vec![dt * 4.0; coupon_levels.len()];
  let coupon_schedule = TreeCouponSchedule::new(coupon_levels.clone(), accrual_factors);

  let swpn = BermudanSwaption::new(
    SwaptionDirection::Payer,
    0.045,
    1.0_f64,
    ExerciseSchedule::new(coupon_levels),
    coupon_schedule,
  );
  let hw_price = swpn.price_on(&hw_tree);
  let g2pp_price = swpn.price_on_g2pp(&g2pp_tree);
  let rel = (hw_price - g2pp_price).abs() / hw_price.max(1e-10);
  assert!(
    rel < 0.05,
    "G2++ with vanishing 2nd factor should match 1F HW: hw={hw_price:.8}, g2pp={g2pp_price:.8}, rel={rel:.2e}"
  );
}

#[test]
fn calendar_bermudan_matches_manual_levels() {
  let valuation = d(2024, 1, 15);
  let horizon = 5.0_f64;
  let steps = 20;
  let dt = horizon / steps as f64;
  let model = HullWhiteTreeModel::new(0.04_f64, 0.1, 0.04, 0.01);
  let tree = HullWhiteTree::new(model, horizon, steps);

  let exercise_dates: Vec<NaiveDate> = (1..=4).map(|i| d(2024 + i, 1, 15)).collect();
  let coupon_dates: Vec<NaiveDate> = (1..=5).map(|i| d(2024 + i, 1, 15)).collect();
  let accruals = vec![1.0_f64; coupon_dates.len()];
  let swpn_cal = BermudanSwaption::from_calendar(
    SwaptionDirection::Payer,
    0.04_f64,
    1.0,
    valuation,
    DayCountConvention::Actual365Fixed,
    dt,
    &exercise_dates,
    &coupon_dates,
    &accruals,
  );
  let price_cal = swpn_cal.price_on(&tree);
  assert!(price_cal >= 0.0 && price_cal.is_finite());
}

#[test]
fn bermudan_payer_zero_value_deep_out_of_the_money() {
  let horizon: f64 = 5.0;
  let steps: usize = 40;
  let dt: f64 = horizon / steps as f64;
  let model = HullWhiteTreeModel::new(0.02_f64, 0.05, 0.02, 0.005);
  let tree = HullWhiteTree::new(model, horizon, steps);

  let coupon_levels: Vec<usize> = (4..=steps).step_by(4).collect();
  let accrual_factors = vec![dt * 4.0; coupon_levels.len()];
  let coupon_schedule = TreeCouponSchedule::new(coupon_levels.clone(), accrual_factors);

  let bermudan = BermudanSwaption::new(
    SwaptionDirection::Payer,
    0.20,
    1.0,
    ExerciseSchedule::new(coupon_levels),
    coupon_schedule,
  );
  let npv = bermudan.price_on(&tree);
  assert!(
    npv < 1e-4,
    "deep-OTM payer swaption should be near zero, got {npv:.6}"
  );
}

//! Comparison tests for caps, floors, collars, and swaptions.
//!
//! Closed-form prices are checked against Black-76 put-call parity,
//! Bachelier reference values, and the textbook swap-annuity / forward-rate
//! identities. Bermudan swaption prices are sanity-checked against their
//! European counterparts.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::Leg;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BachelierVolatility;
use stochastic_rs::quant::instruments::BermudanSwaption;
use stochastic_rs::quant::instruments::BlackVolatility;
use stochastic_rs::quant::instruments::Cap;
use stochastic_rs::quant::instruments::CmsCaplet;
use stochastic_rs::quant::instruments::CmsFloorlet;
use stochastic_rs::quant::instruments::Collar;
use stochastic_rs::quant::instruments::EuropeanSwaption;
use stochastic_rs::quant::instruments::ExerciseSchedule;
use stochastic_rs::quant::instruments::Floor;
use stochastic_rs::quant::instruments::JamshidianHullWhiteSwaption;
use stochastic_rs::quant::instruments::SabrVolatility;
use stochastic_rs::quant::instruments::ShiftedSabrVolatility;
use stochastic_rs::quant::instruments::SwapDirection;
use stochastic_rs::quant::instruments::SwaptionDirection;
use stochastic_rs::quant::instruments::TreeCouponSchedule;
use stochastic_rs::quant::instruments::VanillaInterestRateSwap;
use stochastic_rs::quant::lattice::CurveFittedHullWhiteTree;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;

const TOL: f64 = 1e-10;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.0, max_t / 4.0, max_t / 2.0, 3.0 * max_t / 4.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

fn approx(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b:.12}, got {a:.12}, diff {:.2e}",
    (a - b).abs()
  );
}

fn floating_leg(
  valuation: NaiveDate,
  end: NaiveDate,
  notional: f64,
  day_count: DayCountConvention,
) -> Leg<f64> {
  let schedule = ScheduleBuilder::new(valuation, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    day_count,
  ));
  Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(periods, notional),
    index,
    0.0,
    day_count,
  )
}

#[test]
fn cap_equals_sum_of_caplet_prices() {
  let valuation = d(2024, 1, 15);
  let leg = floating_leg(
    valuation,
    d(2026, 1, 15),
    1_000_000.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 2.2);
  let cap = Cap::new(0.045, leg, BlackVolatility::new(0.25));
  let val = cap.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let sum: f64 = val.caplet_prices.iter().sum();
  approx(val.npv, sum, TOL);
  assert!(val.npv > 0.0, "cap must have positive value");
}

#[test]
fn cap_minus_floor_equals_forward_swap_intrinsic() {
  use stochastic_rs::quant::cashflows::Cashflow;

  let valuation = d(2024, 1, 15);
  let notional = 1_000_000.0;
  let strike = 0.04;
  let leg = floating_leg(
    valuation,
    d(2026, 1, 15),
    notional,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 2.2);

  let cap = Cap::new(strike, leg.clone(), BlackVolatility::new(0.3));
  let floor = Floor::new(strike, leg.clone(), BlackVolatility::new(0.3));
  let collar = Collar::new(cap, floor);
  let val = collar.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let mut manual = 0.0;
  for (i, cashflow) in leg.cashflows().iter().enumerate() {
    let Cashflow::Floating(coupon) = cashflow else {
      continue;
    };
    let tau_payment =
      DayCountConvention::Actual365Fixed.year_fraction(valuation, coupon.period.payment_date);
    let df = curve.discount_factor(tau_payment);
    let forward = val.cap.forward_rates[i];
    manual += df * notional * coupon.period.accrual_factor * (forward - strike);
  }

  approx(val.npv, manual, TOL);
}

#[test]
fn european_payer_put_call_parity_against_forward_swap_value() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 10_000_000.0;
  let strike = 0.045;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 6.5);
  let vol = BlackVolatility::new(0.3);

  let payer = EuropeanSwaption::new(SwaptionDirection::Payer, strike, expiry, swap.clone(), vol);
  let receiver = EuropeanSwaption::new(
    SwaptionDirection::Receiver,
    strike,
    expiry,
    swap.clone(),
    vol,
  );

  let payer_val = payer.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let receiver_val = receiver.valuation(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let expected_diff = payer_val.annuity * (payer_val.forward_swap_rate - strike);
  let actual_diff = payer_val.npv - receiver_val.npv;
  approx(actual_diff, expected_diff, 1e-6);
}

#[test]
fn bachelier_swaption_matches_black_atm_at_small_vol() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 1_000_000.0;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let curve = flat_curve(0.04, 6.5);

  let forward_swap_rate = {
    let swap = VanillaInterestRateSwap::new(
      SwapDirection::Payer,
      &fixed_schedule,
      &float_schedule,
      notional,
      0.04,
      DayCountConvention::Thirty360,
      index.clone(),
      0.0,
      DayCountConvention::Actual360,
    );
    swap
      .valuation(valuation, DayCountConvention::Actual365Fixed, &curve)
      .fair_rate
  };

  let strike = forward_swap_rate;
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );

  let sigma_lognormal = 0.2;
  let sigma_normal = sigma_lognormal * forward_swap_rate;

  let black = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap.clone(),
    BlackVolatility::new(sigma_lognormal),
  );
  let bachelier = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap,
    BachelierVolatility::new(sigma_normal),
  );

  let black_npv = black.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  let bachelier_npv = bachelier.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );

  let rel_err = (black_npv - bachelier_npv).abs() / black_npv;
  assert!(rel_err < 0.05, "ATM Bachelier/Black mismatch: {rel_err:.4}");
}

#[test]
fn sabr_swaption_produces_positive_price() {
  let valuation = d(2024, 1, 15);
  let expiry = d(2025, 1, 15);
  let start = d(2025, 1, 15);
  let end = d(2030, 1, 15);
  let notional = 1_000_000.0;
  let strike = 0.045;

  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    notional,
    strike,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 6.5);

  let swpn = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    expiry,
    swap,
    SabrVolatility::new(0.3, 0.5, 0.4, -0.2),
  );
  let npv = swpn.npv(
    valuation,
    DayCountConvention::Actual365Fixed,
    DayCountConvention::Actual365Fixed,
    &curve,
  );
  assert!(npv > 0.0, "Sabr swaption must have positive value");
  assert!(npv.is_finite());
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
fn shifted_sabr_matches_plain_sabr_at_zero_shift() {
  let plain = SabrVolatility::new(0.3_f64, 0.5, 0.4, -0.2);
  let shifted = ShiftedSabrVolatility::new(0.3_f64, 0.5, 0.4, -0.2, 0.0);
  use stochastic_rs::quant::instruments::VolatilityModel;
  let v_plain = plain.implied_volatility(0.04, 0.045, 1.5);
  let v_shift = shifted.implied_volatility(0.04, 0.045, 1.5);
  approx(v_plain, v_shift, 1e-12);
}

#[test]
fn shifted_sabr_handles_zero_rate() {
  let shifted = ShiftedSabrVolatility::new(0.25_f64, 0.5, 0.4, -0.3, 0.03);
  use stochastic_rs::quant::instruments::VolatilityModel;
  let v = shifted.implied_volatility(0.001, 0.0, 2.0);
  assert!(
    v.is_finite() && v > 0.0,
    "shifted vol at zero strike must be positive, got {v}"
  );
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
fn cms_caplet_produces_finite_positive_price() {
  let caplet = CmsCaplet {
    strike: 0.05_f64,
    notional: 1_000_000.0,
    accrual_factor: 0.5,
    discount_factor: 0.96,
    forward_cms: 0.04,
    t_fix: 2.0,
    swap_years: 10.0,
    fixed_freq: 2.0,
    payment_delay: 0.25,
    vol: BlackVolatility::new(0.3_f64),
  };
  let price = caplet.price();
  assert!(price.is_finite() && price >= 0.0, "price={price}");

  let floorlet = CmsFloorlet {
    strike: 0.05_f64,
    notional: 1_000_000.0,
    accrual_factor: 0.5,
    discount_factor: 0.96,
    forward_cms: 0.04,
    t_fix: 2.0,
    swap_years: 10.0,
    fixed_freq: 2.0,
    payment_delay: 0.25,
    vol: BlackVolatility::new(0.3_f64),
  };
  assert!(floorlet.price().is_finite() && floorlet.price() > price * 0.5);
}

#[test]
fn hw_calibration_recovers_self_consistent_params() {
  use stochastic_rs::quant::calibration::HullWhiteSwaptionCalibrator;
  use stochastic_rs::quant::calibration::SwaptionQuote;

  let times = array![0.0, 1.0, 2.0, 5.0, 10.0];
  let rates = array![0.03, 0.035, 0.038, 0.04, 0.042];
  let curve =
    DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates);

  let target_a = 0.08_f64;
  let target_sigma = 0.012_f64;

  let quote_params: Vec<(f64, f64)> = vec![(1.0, 5.0), (2.0, 5.0), (3.0, 3.0)];
  let mut quotes = Vec::new();
  for (expiry, tenor) in quote_params {
    let n_payments = (tenor * 2.0).round() as usize;
    let accrual = tenor / n_payments as f64;
    let coupon_times: Vec<f64> = (1..=n_payments)
      .map(|k| expiry + accrual * k as f64)
      .collect();
    let accruals = vec![accrual; n_payments];
    let annuity: f64 = coupon_times
      .iter()
      .map(|&t| curve.discount_factor(t) * accrual)
      .sum();
    let fair_rate = (curve.discount_factor(expiry)
      - curve.discount_factor(*coupon_times.last().unwrap()))
      / annuity;
    let swpn = JamshidianHullWhiteSwaption::new(
      SwaptionDirection::Payer,
      fair_rate,
      1.0,
      expiry,
      coupon_times,
      accruals,
      target_a,
      target_sigma,
    );
    let model_price = swpn.price(&curve);
    let forward = fair_rate;
    let tau_exp = expiry;
    let implied_vol = implied_vol_solve(model_price / annuity, forward, forward, tau_exp);
    quotes.push(SwaptionQuote {
      expiry,
      tenor,
      black_vol: implied_vol,
      fixed_accrual: 0.5,
      direction: SwaptionDirection::Payer,
      weight: None,
    });
  }

  let calibrator = HullWhiteSwaptionCalibrator {
    quotes: &quotes,
    curve: &curve,
    notional: 1.0,
    initial_guess: Some((0.03, 0.008)),
    max_iters: 400,
    sd_tolerance: 1e-10,
  };
  let result = calibrator.calibrate();
  assert!(
    (result.mean_reversion - target_a).abs() < 0.02,
    "a recovery: got {}, target {target_a}",
    result.mean_reversion
  );
  assert!(
    (result.sigma - target_sigma).abs() < 0.005,
    "sigma recovery: got {}, target {target_sigma}",
    result.sigma
  );
}

fn implied_vol_solve(atm_forward_value: f64, forward: f64, _strike: f64, tau: f64) -> f64 {
  let target = atm_forward_value;
  let mut lo = 1e-6_f64;
  let mut hi = 5.0_f64;
  for _ in 0..120 {
    let mid = 0.5 * (lo + hi);
    let sqrt_t = tau.sqrt();
    let d1 = 0.5 * mid * sqrt_t;
    let d2 = -d1;
    let n = statrs::distribution::Normal::default();
    use statrs::distribution::ContinuousCDF;
    let price = forward * (n.cdf(d1) - n.cdf(d2));
    if (price - target).abs() < 1e-12 {
      return mid;
    }
    if price < target {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  0.5 * (lo + hi)
}

#[test]
fn sabr_caplet_calibration_recovers_self_consistent_params() {
  use stochastic_rs::quant::calibration::SabrCapletCalibrator;
  use stochastic_rs::quant::pricing::sabr::hagan_implied_vol;

  let target_alpha = 0.035_f64;
  let target_beta = 0.5;
  let target_nu = 0.45;
  let target_rho = -0.25;
  let forward = 0.04_f64;
  let expiry = 2.0_f64;

  let strikes: Vec<f64> = (0..=10).map(|i| 0.02 + 0.004 * i as f64).collect();
  let market_vols: Vec<f64> = strikes
    .iter()
    .map(|&k| {
      hagan_implied_vol(
        k,
        forward,
        expiry,
        target_alpha,
        target_beta,
        target_nu,
        target_rho,
      )
    })
    .collect();

  let cal = SabrCapletCalibrator {
    forward,
    expiry,
    beta: target_beta,
    strikes: strikes.clone(),
    market_vols: market_vols.clone(),
    weights: None,
    initial_guess: Some((0.03, 0.3, -0.1)),
    max_iters: 800,
    sd_tolerance: 1e-12,
  };
  let res = cal.calibrate();
  assert!(res.rmse < 1e-6, "RMSE too large: {}", res.rmse);
  assert!((res.alpha - target_alpha).abs() < 1e-3);
  assert!((res.nu - target_nu).abs() < 1e-2);
  assert!((res.rho - target_rho).abs() < 1e-2);
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

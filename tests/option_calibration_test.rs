//! CMS caplet/floorlet pricing and Hull-White / SABR calibration tests.
//!
//! Includes finite-positive sanity checks on CMS instruments and round-trip
//! consistency tests where calibrators recover the parameters used to
//! generate synthetic market quotes.

use ndarray::array;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BlackVolatility;
use stochastic_rs::quant::instruments::CmsCaplet;
use stochastic_rs::quant::instruments::CmsFloorlet;
use stochastic_rs::quant::instruments::JamshidianHullWhiteSwaption;
use stochastic_rs::quant::instruments::SwaptionDirection;

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
  use stochastic_rs::traits::Calibrator;

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
    regularization: None,
  };
  let result = calibrator.calibrate(None).unwrap();
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
    use stochastic_rs::distributions::special::norm_cdf;
    let price = forward * (norm_cdf(d1) - norm_cdf(d2));
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
  use stochastic_rs::traits::Calibrator;

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
    shift: 0.0,
    expiry,
    beta: target_beta,
    strikes: strikes.clone(),
    market_vols: market_vols.clone(),
    weights: None,
    initial_guess: Some((0.03, 0.3, -0.1)),
    max_iters: 800,
    sd_tolerance: 1e-12,
  };
  let res = cal.calibrate(None).unwrap();
  assert!(res.rmse < 1e-6, "RMSE too large: {}", res.rmse);
  assert!((res.alpha - target_alpha).abs() < 1e-3);
  assert!((res.nu - target_nu).abs() < 1e-2);
  assert!((res.rho - target_rho).abs() < 1e-2);
  assert_eq!(res.shift, 0.0);
}

/// Reviewer repro: negative forward and negative strikes, no shift
/// configured (the default). `hagan_implied_vol` now asserts `k > 0` /
/// `f > 0`, so this used to panic inside the Nelder-Mead cost callback
/// instead of surfacing through `Calibrator::calibrate`'s `Result`.
#[test]
fn sabr_caplet_calibration_rejects_negative_forward_without_shift() {
  use stochastic_rs::quant::calibration::SabrCapletCalibrator;
  use stochastic_rs::traits::Calibrator;

  let cal = SabrCapletCalibrator::new(
    -0.002,
    0.5,
    1.0,
    vec![-0.005, -0.002, 0.0, 0.005],
    vec![0.5, 0.5, 0.5, 0.5],
  );
  let err = cal.calibrate(None).unwrap_err();
  let msg = err.to_string();
  assert!(
    msg.contains("-0.002"),
    "error should name the offending forward: {msg}"
  );
}

/// A shift that fixes the forward but is still too small for the lowest
/// strike must be reported by value, not silently truncated or panicked
/// through.
#[test]
fn sabr_caplet_calibration_rejects_insufficient_shift() {
  use stochastic_rs::quant::calibration::SabrCapletCalibrator;
  use stochastic_rs::traits::Calibrator;

  // shift=0.003 makes the forward positive (-0.002 + 0.003 = 0.001) but
  // not the lowest strike (-0.005 + 0.003 = -0.002 <= 0).
  let cal = SabrCapletCalibrator::new(
    -0.002,
    0.5,
    1.0,
    vec![-0.005, -0.002, 0.0, 0.005],
    vec![0.5, 0.5, 0.5, 0.5],
  )
  .with_shift(0.003);
  let err = cal.calibrate(None).unwrap_err();
  let msg = err.to_string();
  assert!(
    msg.contains("-0.005"),
    "error should name the offending strike: {msg}"
  );
}

/// Round-trip recovery on a negative forward with strikes spanning
/// negative, zero and positive values — the realistic EUR/JPY/CHF caplet
/// regime from roughly 2015 to 2022. `shift = 0.02` is the market
/// convention mentioned for EUR caplets; every shifted strike/forward is
/// then comfortably positive (0.0125 to 0.03).
#[test]
fn sabr_caplet_calibration_recovers_self_consistent_params_negative_forward() {
  use stochastic_rs::quant::calibration::SabrCapletCalibrator;
  use stochastic_rs::quant::pricing::sabr::hagan_implied_vol;
  use stochastic_rs::traits::Calibrator;

  let shift = 0.02_f64;
  let forward = -0.0025_f64;
  let expiry = 1.5_f64;
  let target_beta = 0.5;
  let target_nu = 0.4;
  let target_rho = -0.3;

  let strikes: Vec<f64> = vec![-0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075, 0.01];
  let shifted_forward = forward + shift;
  // sigma_ATM ~= 38% on the shifted forward, chosen so the shifted-coordinate
  // smile stays in Hagan's well-behaved regime (log-moneyness < 1 in
  // magnitude across the whole strike grid).
  let target_alpha = 0.38 * shifted_forward.powf(1.0 - target_beta);

  let market_vols: Vec<f64> = strikes
    .iter()
    .map(|&k| {
      hagan_implied_vol(
        k + shift,
        shifted_forward,
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
    shift,
    expiry,
    beta: target_beta,
    strikes: strikes.clone(),
    market_vols: market_vols.clone(),
    weights: None,
    initial_guess: Some((0.03, 0.3, -0.1)),
    max_iters: 800,
    sd_tolerance: 1e-12,
  };
  let res = cal.calibrate(None).unwrap();

  assert_eq!(res.shift, shift);
  assert!(res.rmse < 1e-6, "RMSE too large: {}", res.rmse);
  assert!(
    (res.alpha - target_alpha).abs() < 1e-3,
    "alpha recovery: got {}, target {target_alpha}",
    res.alpha
  );
  assert!(
    (res.nu - target_nu).abs() < 1e-2,
    "nu recovery: got {}, target {target_nu}",
    res.nu
  );
  assert!(
    (res.rho - target_rho).abs() < 1e-2,
    "rho recovery: got {}, target {target_rho}",
    res.rho
  );
}

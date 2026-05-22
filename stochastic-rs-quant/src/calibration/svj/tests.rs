use super::SVJCalibrator;
use super::SVJParams;
use super::loss::bates_call_price;
use crate::LossMetric;
use crate::OptionType;
use crate::traits::Calibrator;

const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

#[test]
fn bates_pricer_matches_heston_reference() {
  let p = SVJParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma_v: 0.3,
    rho: -0.7,
    lambda: 0.0,
    mu_j: 0.0,
    sigma_j: 0.01,
  };
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = bates_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 0.1,
      "Heston K={k}: got {price:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn svj_calibrate_recovers_heston_prices() {
  let n = STRIKES.len();
  let calibrator = SVJCalibrator::new(
    Some(SVJParams {
      v0: 0.06,
      kappa: 2.0,
      theta: 0.06,
      sigma_v: 0.4,
      rho: -0.5,
      lambda: 0.1,
      mu_j: 0.0,
      sigma_j: 0.1,
    }),
    HESTON_REF.to_vec().into(),
    vec![100.0; n].into(),
    STRIKES.to_vec().into(),
    0.05,
    Some(0.0),
    1.0,
    OptionType::Call,
    false,
  );

  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.5,
    "SVJ→Heston RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  println!(
    "SVJ→Heston: v0={:.4}, kappa={:.4}, theta={:.4}, sigma_v={:.4}, rho={:.4}, lambda={:.4}",
    result.v0, result.kappa, result.theta, result.sigma_v, result.rho, result.lambda
  );
}

/// Long-maturity / high-|ρ| regression: `bates_cf` must use the
/// Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap" form
/// (`g̃ = 1/g_original`, `exp(-d·τ)`). Original Heston (1993) form develops a
/// branch-cut discontinuity at T = 5y, ρ = -0.9; the Trap form does not.
#[test]
fn svj_bates_little_trap_long_maturity_high_rho() {
  let p = SVJParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma_v: 0.3,
    rho: -0.9,
    lambda: 0.3,
    mu_j: -0.05,
    sigma_j: 0.15,
  };
  let call = bates_call_price(&p, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "SVJ bates_cf Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

#[test]
fn test_svj_calibrate() {
  let s = vec![
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
  ];

  let k = vec![
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
  ];

  let c_market = vec![
    21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2, 1.6, 1.2, 0.9,
  ];

  let r = 0.01;
  let q = Some(0.0);
  let tau = 0.5;
  let option_type = OptionType::Call;

  let calibrator = SVJCalibrator::new(
    Some(SVJParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.5,
      rho: -0.7,
      lambda: 0.5,
      mu_j: -0.05,
      sigma_j: 0.1,
    }),
    c_market.into(),
    s.into(),
    k.into(),
    r,
    q,
    tau,
    option_type,
    true,
  );

  let result = calibrator.calibrate(None).unwrap();
  println!(
    "SVJ result: v0={}, kappa={}, theta={}, sigma_v={}, rho={}, lambda={}, mu_j={}, sigma_j={}",
    result.v0,
    result.kappa,
    result.theta,
    result.sigma_v,
    result.rho,
    result.lambda,
    result.mu_j,
    result.sigma_j,
  );
  println!("Loss: {:?}", result.loss);
}

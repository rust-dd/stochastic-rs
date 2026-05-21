use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::heston_nml_cekf::HestonNMLECEKFConfig;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston as HestonProcess;

use super::*;
use crate::OptionType;
use crate::pricing::heston::HestonPricer;
use crate::traits::Calibrator;
use crate::traits::PricerExt;
use crate::traits::ProcessExt;

const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const REF_STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

#[test]
fn heston_pricer_matches_reference() {
  for (i, &k) in REF_STRIKES.iter().enumerate() {
    let pricer = HestonPricer::new(
      100.0,
      0.04,
      k,
      0.05,
      Some(0.0),
      -0.7,
      1.5,
      0.04,
      0.3,
      Some(0.0),
      Some(1.0),
      None,
      None,
    );
    let (call, _) = pricer.calculate_call_put();
    assert!(
      (call - HESTON_REF[i]).abs() < 0.15,
      "Heston K={k}: got {call:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn heston_calibrate_reference_prices() {
  let n = REF_STRIKES.len();
  let calibrator = HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.06,
      kappa: 2.0,
      theta: 0.06,
      sigma: 0.4,
      rho: -0.5,
    }),
    HESTON_REF.to_vec().into(),
    vec![100.0; n].into(),
    REF_STRIKES.to_vec().into(),
    0.05,
    Some(0.0),
    1.0,
    OptionType::Call,
    None,
    None,
    None,
    true,
  );

  let true_params = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let model_prices = calibrator.compute_model_prices_for_numeric(&true_params);
  for i in 0..n {
    assert!(
      (model_prices[i] - HESTON_REF[i]).abs() < 0.15,
      "Heston cal K={}: model={:.6}, ref={:.6}",
      REF_STRIKES[i],
      model_prices[i],
      HESTON_REF[i]
    );
  }
}

#[test]
fn test_heston_calibrate() {
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

  let calibrator = HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.7,
    }),
    c_market.clone().into(),
    s.clone().into(),
    k.clone().into(),
    r,
    q,
    tau,
    option_type,
    None,
    None,
    None,
    true,
  );

  calibrator.calibrate(None).unwrap();
}

#[test]
fn test_heston_calibrate_with_mle_seed() {
  let s0 = 100.0;
  let v0 = 0.04;
  let true_params = HestonParams {
    v0,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.6,
  };
  let n = 256usize;
  let t = 1.0;
  let mu = 0.0;

  let process = HestonProcess::new(
    Some(s0),
    Some(v0),
    true_params.kappa,
    true_params.theta,
    true_params.sigma,
    true_params.rho,
    mu,
    n,
    Some(t),
    HestonPow::Sqrt,
    Some(true),
    Deterministic::new(42),
  );

  let [s_ts, v_ts] = process.sample();

  let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let s_grid = vec![s0; strikes.len()];
  let r = 0.01;
  let q = Some(0.0);
  let tau = 0.5;

  let mut c_market = Vec::with_capacity(strikes.len());
  for &kk in &strikes {
    let pr = HestonPricer::new(
      s0,
      true_params.v0,
      kk,
      r,
      q,
      true_params.rho,
      true_params.kappa,
      true_params.theta,
      true_params.sigma,
      Some(0.0),
      Some(tau),
      None,
      None,
    );
    let (call, _) = pr.calculate_call_put();
    c_market.push(call);
  }

  let calibrator = HestonCalibrator::new(
    None,
    c_market.clone().into(),
    s_grid.clone().into(),
    strikes.clone().into(),
    r,
    q,
    tau,
    OptionType::Call,
    Some(s_ts),
    Some(v_ts),
    Some(r),
    true,
  );

  calibrator.calibrate(None).unwrap();
}

#[test]
fn test_heston_calibrate_with_pmle_seed() {
  let s0 = 100.0;
  let v0 = 0.04;
  let true_params = HestonParams {
    v0,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.6,
  };
  let n = 256usize;
  let t = 1.0;
  let mu = 0.0;

  let process = HestonProcess::new(
    Some(s0),
    Some(v0),
    true_params.kappa,
    true_params.theta,
    true_params.sigma,
    true_params.rho,
    mu,
    n,
    Some(t),
    HestonPow::Sqrt,
    Some(true),
    Deterministic::new(42),
  );
  let [s_ts, v_ts] = process.sample();

  let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let s_grid = vec![s0; strikes.len()];
  let r = 0.01;
  let q = Some(0.0);
  let tau = 0.5;

  let mut c_market = Vec::with_capacity(strikes.len());
  for &kk in &strikes {
    let pr = HestonPricer::new(
      s0,
      true_params.v0,
      kk,
      r,
      q,
      true_params.rho,
      true_params.kappa,
      true_params.theta,
      true_params.sigma,
      Some(0.0),
      Some(tau),
      None,
      None,
    );
    let (call, _) = pr.calculate_call_put();
    c_market.push(call);
  }

  let mut calibrator = HestonCalibrator::new(
    None,
    c_market.clone().into(),
    s_grid.clone().into(),
    strikes.clone().into(),
    r,
    q,
    tau,
    OptionType::Call,
    Some(s_ts),
    Some(v_ts),
    Some(r),
    true,
  );
  calibrator.set_mle_seed_method(HestonMleSeedMethod::Pmle);
  calibrator.set_mle_delta(Some(t / (n - 1) as f64));

  calibrator.calibrate(None).unwrap();
}

#[test]
fn test_heston_calibrate_with_nmle_cekf_seed() {
  let s0 = 100.0;
  let v0 = 0.04;
  let true_params = HestonParams {
    v0,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.6,
  };
  let n = 256usize;
  let t = 1.0;
  let mu = 0.0;

  let process = HestonProcess::new(
    Some(s0),
    Some(v0),
    true_params.kappa,
    true_params.theta,
    true_params.sigma,
    true_params.rho,
    mu,
    n,
    Some(t),
    HestonPow::Sqrt,
    Some(true),
    Deterministic::new(42),
  );
  let [s_ts, _v_ts] = process.sample();

  let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let s_grid = vec![s0; strikes.len()];
  let r = 0.01;
  let q = Some(0.0);
  let tau = 0.5;

  let mut c_market = Vec::with_capacity(strikes.len());
  for &kk in &strikes {
    let pr = HestonPricer::new(
      s0,
      true_params.v0,
      kk,
      r,
      q,
      true_params.rho,
      true_params.kappa,
      true_params.theta,
      true_params.sigma,
      Some(0.0),
      Some(tau),
      None,
      None,
    );
    let (call, _) = pr.calculate_call_put();
    c_market.push(call);
  }

  let mut calibrator = HestonCalibrator::new(
    None,
    c_market.clone().into(),
    s_grid.clone().into(),
    strikes.clone().into(),
    r,
    q,
    tau,
    OptionType::Call,
    Some(s_ts),
    None,
    Some(r),
    true,
  );
  calibrator.set_mle_seed_method(HestonMleSeedMethod::NmleCekf);
  calibrator.set_mle_delta(Some(t / (n - 1) as f64));
  calibrator.set_nmle_cekf_config(HestonNMLECEKFConfig {
    max_iters: 6,
    tol: 1e-5,
    param_damping: 0.6,
    initial_v0: v0,
    ..HestonNMLECEKFConfig::default()
  });

  calibrator.calibrate(None).unwrap();
}

#[test]
fn test_heston_calibrate_ls_dataset() {
  let r = 0.04;
  let q = Some(0.06);
  let tau = 0.083;

  let strikes_ls: Vec<f64> = vec![
    5220.318, 6090.371, 6960.424, 7830.477, 8265.5035, 8483.01675, 8700.53, 8918.04325, 9135.5565,
    9570.583, 10440.636, 11310.689,
  ];
  let spots_ls: Vec<f64> = vec![
    8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53,
    8700.53, 8700.53,
  ];
  let vol_ls: Vec<f64> = vec![
    0.3669, 0.3082, 0.2218, 0.1799, 0.1393, 0.1156, 0.1019, 0.0923, 0.0915, 0.1086, 0.1237, 0.136,
  ];
  let markets_ls: Vec<f64> = vec![
    3499.564, 2632.74, 1765.933, 901.846, 479.055, 279.313, 118.848, 28.79, 4.23, 0.143, 1.799e-5,
    1.259e-9,
  ];

  let s_ts = Array1::from(spots_ls.clone());
  let v_ts = Array1::from(vol_ls.iter().map(|x| x * x).collect::<Vec<f64>>());

  let calibrator = HestonCalibrator::new(
    None,
    markets_ls.clone().into(),
    spots_ls.clone().into(),
    strikes_ls.clone().into(),
    r,
    q,
    tau,
    OptionType::Call,
    Some(s_ts),
    Some(v_ts),
    Some(r),
    true,
  );

  calibrator.calibrate(None).unwrap();
  let history = calibrator.history();
  println!("{:?}", history);
}

#[test]
fn test_heston_cui_price_and_jacobian_finite() {
  let params = HestonParams {
    v0: 0.04,
    kappa: 1.8,
    theta: 0.05,
    sigma: 0.45,
    rho: -0.55,
  };
  let s = vec![100.0; 6];
  let k = vec![80.0, 90.0, 95.0, 100.0, 110.0, 120.0];
  let r = 0.01;
  let q = Some(0.0);
  let tau = 0.75;
  let option_type = OptionType::Call;

  let market = vec![22.0, 14.8, 11.9, 9.5, 6.2, 4.0];
  let mut calibrator = HestonCalibrator::new(
    Some(params.clone()),
    market.into(),
    s.clone().into(),
    k.clone().into(),
    r,
    q,
    tau,
    option_type,
    None,
    None,
    None,
    false,
  );
  calibrator.set_jacobian_method(HestonJacobianMethod::CuiAnalytic);

  let (c_model, jac) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .expect("Cui model/jacobian should be computable");
  assert_eq!(c_model.len(), k.len());
  assert_eq!(jac.nrows(), k.len());
  assert_eq!(jac.ncols(), 5);
  assert!(c_model.iter().all(|x| x.is_finite()));
  assert!(jac.iter().all(|x| x.is_finite()));

  for (i, &strike) in k.iter().enumerate() {
    let pr = HestonPricer::new(
      s[i],
      params.v0,
      strike,
      r,
      q,
      params.rho,
      params.kappa,
      params.theta,
      params.sigma,
      Some(0.0),
      Some(tau),
      None,
      None,
    );
    let (call_ref, _) = pr.calculate_call_put();
    let rel = ((c_model[i] - call_ref).abs()) / (1.0 + call_ref.abs());
    assert!(rel < 5e-2, "quote {} relative gap too large: {}", i, rel);
  }
}

#[test]
fn test_heston_cui_jacobian_matches_numeric() {
  let params = HestonParams {
    v0: 0.05,
    kappa: 1.4,
    theta: 0.06,
    sigma: 0.35,
    rho: -0.45,
  };
  let s = vec![100.0; 5];
  let k = vec![85.0, 95.0, 100.0, 105.0, 115.0];
  let r = 0.015;
  let q = Some(0.0);
  let tau = 0.6;

  let market = vec![18.0, 11.0, 8.5, 6.4, 3.7];
  let calibrator = HestonCalibrator::new(
    Some(params.clone()),
    market.into(),
    s.into(),
    k.into(),
    r,
    q,
    tau,
    OptionType::Call,
    None,
    None,
    None,
    false,
  );

  let jac_num = calibrator.numeric_jacobian(&params);
  let (_, jac_cui) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .expect("Cui Jacobian should be computable");

  for row in 0..jac_num.nrows() {
    for col in 0..jac_num.ncols() {
      let n = jac_num[(row, col)];
      let a = jac_cui[(row, col)];
      let rel = (a - n).abs() / (1.0 + n.abs());
      assert!(
        rel < 5e-3,
        "Jacobian mismatch at ({}, {}): analytic={}, numeric={}, rel={}",
        row,
        col,
        a,
        n,
        rel
      );
    }
  }
}

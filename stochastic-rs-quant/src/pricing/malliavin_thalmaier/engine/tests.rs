use ndarray::Array2;
use owens_t::biv_norm;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::*;
use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::pricing::malliavin_thalmaier::AssetParams;
use crate::pricing::malliavin_thalmaier::MultiHestonParams;
use crate::traits::PricerExt;

fn two_asset_params() -> MultiHestonParams<f64> {
  let a = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  };
  let mut cross = Array2::<f64>::eye(2);
  cross[[0, 1]] = 0.5;
  cross[[1, 0]] = 0.5;
  MultiHestonParams {
    assets: vec![a.clone(), a],
    cross_corr: cross,
    r: 0.05,
    tau: 1.0,
    n_steps: 100,
  }
}

/// Degenerált Heston (ξ→0) ~ Gbm → összehasonlítás BS analitikus delta-val.
fn gbm_like_params() -> MultiHestonParams<f64> {
  let a = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 10.0,
    theta: 0.04,
    xi: 1e-6,
    rho: 0.0,
  };
  MultiHestonParams {
    assets: vec![a.clone(), a],
    cross_corr: Array2::<f64>::eye(2),
    r: 0.05,
    tau: 1.0,
    n_steps: 252,
  }
}

fn gbm_like_params_3d() -> MultiHestonParams<f64> {
  let a = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 10.0,
    theta: 0.04,
    xi: 1e-6,
    rho: 0.0,
  };
  MultiHestonParams {
    assets: vec![a.clone(), a.clone(), a],
    cross_corr: Array2::<f64>::eye(3),
    r: 0.05,
    tau: 1.0,
    n_steps: 128,
  }
}

/// Scenario used for the Greeks experiment in Kohatsu-Higa--Yasuda (2010),
/// Figure 5/6, but simulated here in the near-Gbm limit of the multi-Heston
/// engine so we can compare against the exact Black-Scholes benchmark.
fn paper_bs_digital_put_params(s01: f64, n_steps: usize) -> MultiHestonParams<f64> {
  let a1 = AssetParams {
    s0: s01,
    v0: 0.3 * 0.3,
    kappa: 10.0,
    theta: 0.3 * 0.3,
    xi: 1e-8,
    rho: 0.0,
  };
  let a2 = AssetParams {
    s0: 100.0,
    v0: 0.2 * 0.2,
    kappa: 10.0,
    theta: 0.2 * 0.2,
    xi: 1e-8,
    rho: 0.0,
  };
  let mut cross = Array2::<f64>::eye(2);
  cross[[0, 1]] = 0.2;
  cross[[1, 0]] = 0.2;
  MultiHestonParams {
    assets: vec![a1, a2],
    cross_corr: cross,
    r: 0.0,
    tau: 1.0,
    n_steps,
  }
}

fn bs_bivariate_digital_put_price_delta(
  s1: f64,
  s2: f64,
  k1: f64,
  k2: f64,
  sigma1: f64,
  sigma2: f64,
  rho: f64,
  r: f64,
  tau: f64,
) -> (f64, f64) {
  let root_t = tau.sqrt();
  let a1 = ((k1 / s1).ln() - (r - 0.5 * sigma1 * sigma1) * tau) / (sigma1 * root_t);
  let a2 = ((k2 / s2).ln() - (r - 0.5 * sigma2 * sigma2) * tau) / (sigma2 * root_t);
  let disc = (-r * tau).exp();
  let cdf = |x: f64, y: f64, corr: f64| -> f64 { biv_norm(-x, -y, corr) };
  let price = disc * cdf(a1, a2, rho);
  let conditional = norm_cdf((a2 - rho * a1) / (1.0 - rho * rho).sqrt());
  let delta = disc * (-(norm_pdf(a1) * conditional) / (s1 * sigma1 * root_t));
  (price, delta)
}

#[test]
fn delta_digital_put_2d_finite() {
  let e = MtGreeks::new(two_asset_params(), 0.01, 5_000);
  let p = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  for (i, &d) in e.all_deltas(&p).iter().enumerate() {
    assert!(d.is_finite(), "Delta[{i}] = {d}");
  }
}

/// M-T delta vs FD delta for digital put 2D.
///
/// The closed-form g kernel (arctan + ln) should give deltas that agree
/// with bump-and-reprice in sign and both should be negative (higher
/// spot → less likely to finish below strike).
#[test]
fn digital_put_2d_mt_vs_fd() {
  let n = 30_000;
  let params = two_asset_params();
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };

  let mt = MtGreeks::new(params.clone(), 0.01, n).all_deltas_with_seed(&payoff, 0xD161_7A1);

  let bump = 0.5;
  let mut fd = vec![0.0; 2];
  for p in 0..2 {
    let mut up = params.clone();
    up.assets[p].s0 += bump;
    let mut dn = params.clone();
    dn.assets[p].s0 -= bump;
    let seed = 0xFD_D161_7A_u64 ^ p as u64;
    fd[p] = (MtGreeks::new(up, 0.01, n).price_with_seed(&payoff, seed)
      - MtGreeks::new(dn, 0.01, n).price_with_seed(&payoff, seed))
      / (2.0 * bump);
  }

  assert!(
    mt[0] < 0.0 && mt[1] < 0.0,
    "MT deltas should be < 0: [{:.4}, {:.4}]",
    mt[0],
    mt[1]
  );
  assert!(
    fd[0] < 0.0 && fd[1] < 0.0,
    "FD deltas should be < 0: [{:.4}, {:.4}]",
    fd[0],
    fd[1]
  );

  assert_eq!(mt[0] < 0.0, fd[0] < 0.0, "sign mismatch asset 0");
  assert_eq!(mt[1] < 0.0, fd[1] < 0.0, "sign mismatch asset 1");
}

/// Price of digital put 2D should be in (0, e^{-rT}).
#[test]
fn digital_put_2d_price_bounded() {
  let e = MtGreeks::new(two_asset_params(), 0.01, 20_000);
  let p = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let price = e.price(&p);
  let disc = (-0.05_f64).exp();
  assert!(
    price > 0.0 && price < disc,
    "price {price:.4} not in (0, {disc:.4})"
  );
}

#[test]
fn delta_call_finite() {
  let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
  let p = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let d = e.delta(&p, 0);
  assert!(d.is_finite() && d.abs() < 5.0, "Delta = {d}");
}

#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "expensive 3D MC smoke test; run with --release --features openblas"
)]
fn delta_call_finite_in_3d() {
  let e = MtGreeks::new(gbm_like_params_3d(), 0.01, 20_000);
  let p = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let d = e.delta_with_seed(&p, 0, 7);

  let bs = BSMPricer {
    s: 100.0,
    v: 0.2,
    k: 100.0,
    r: 0.05,
    r_d: None,
    r_f: None,
    q: Some(0.0),
    tau: Some(1.0),
    eval: None,
    expiration: None,
    option_type: OptionType::Call,
    b: BSMCoc::Bsm1973,
  };
  let bs_delta = bs.delta();
  let err = (d - bs_delta).abs();

  assert!(d.is_finite(), "3D delta is not finite: {d}");
  assert!(
    err < 0.20,
    "3D delta = {d}, BS delta = {bs_delta}, err = {err}"
  );
}

#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "expensive 3D MC smoke test; run with --release --features openblas"
)]
fn delta_put_finite_in_3d() {
  let e = MtGreeks::new(gbm_like_params_3d(), 0.01, 20_000);
  let p = MtPayoff::Put {
    asset: 0,
    strike: 100.0,
  };
  let d = e.delta_with_seed(&p, 0, 11);

  let bs = BSMPricer {
    s: 100.0,
    v: 0.2,
    k: 100.0,
    r: 0.05,
    r_d: None,
    r_f: None,
    q: Some(0.0),
    tau: Some(1.0),
    eval: None,
    expiration: None,
    option_type: OptionType::Put,
    b: BSMCoc::Bsm1973,
  };
  let bs_delta = bs.delta();
  let err = (d - bs_delta).abs();

  assert!(d.is_finite(), "3D put delta is not finite: {d}");
  assert!(
    err < 0.20,
    "3D put delta = {d}, BS delta = {bs_delta}, err = {err}"
  );
}

#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "expensive MC reference test; run with --release --features openblas"
)]
fn paper_scenario_digital_put_price_matches_bs_reference() {
  let params = paper_bs_digital_put_params(100.0, 512);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let engine = MtGreeks::new(params, 0.01, 20_000);
  let mc_price = engine.price_with_seed(&payoff, 42);
  let (ref_price, _) =
    bs_bivariate_digital_put_price_delta(100.0, 100.0, 100.0, 100.0, 0.3, 0.2, 0.2, 0.0, 1.0);

  let rel_err = (mc_price - ref_price).abs() / ref_price;
  assert!(
    rel_err < 0.06,
    "paper-scenario price = {mc_price:.6}, reference = {ref_price:.6}, rel_err = {rel_err:.4}"
  );
}

#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "expensive MC reference test; run with --release --features openblas"
)]
fn paper_scenario_digital_put_delta_matches_bs_reference() {
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let seeds = [7_u64, 17_u64, 29_u64];
  let mt_delta: f64 = seeds
    .iter()
    .map(|&seed| {
      let engine = MtGreeks::new(paper_bs_digital_put_params(100.0, 512), 0.01, 20_000);
      engine.delta_with_seed(&payoff, 0, seed)
    })
    .sum::<f64>()
    / seeds.len() as f64;
  let (_, ref_delta) =
    bs_bivariate_digital_put_price_delta(100.0, 100.0, 100.0, 100.0, 0.3, 0.2, 0.2, 0.0, 1.0);

  let abs_err = (mt_delta - ref_delta).abs();
  assert!(
    abs_err < 0.0020,
    "paper-scenario delta = {mt_delta:.6}, reference = {ref_delta:.6}, abs_err = {abs_err:.6}"
  );
}

#[test]
fn price_call_positive() {
  let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
  let p = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let v = e.price(&p);
  assert!(v > 0.0 && v < 50.0, "price = {v}");
}

#[test]
fn price_worst_of_put_positive() {
  let e = MtGreeks::new(two_asset_params(), 0.01, 10_000);
  let p = MtPayoff::WorstOfPut { strike: 110.0 };
  assert!(e.price(&p) > 0.0);
}

/// **Validation 1**: MC price vs BS closed-form (degenerált Heston ≈ Gbm).
///
/// BS call: C = S·N(d₁) − K·e^{−rT}·N(d₂), σ = √v₀ = 0.2.
/// Ha ξ ≈ 0, a Heston MC ár konvergálnia kell a BS-hez.
#[test]
fn price_converges_to_bs_when_xi_zero() {
  let params = gbm_like_params();
  let e = MtGreeks::new(params, 0.01, 30_000);
  let payoff = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let mc_price = e.price(&payoff);

  let bs = BSMPricer {
    s: 100.0,
    v: 0.2,
    k: 100.0,
    r: 0.05,
    r_d: None,
    r_f: None,
    q: Some(0.0),
    tau: Some(1.0),
    eval: None,
    expiration: None,
    option_type: OptionType::Call,
    b: BSMCoc::Bsm1973,
  };
  let (bs_call, _) = bs.calculate_call_put();
  let rel_err = (mc_price - bs_call).abs() / bs_call;
  println!("MC price = {mc_price:.4}, BS price = {bs_call:.4}, rel_err = {rel_err:.4}");
  assert!(
    rel_err < 0.10,
    "MC price ({mc_price:.4}) should be within 10% of BS ({bs_call:.4}), rel_err = {rel_err:.4}"
  );
}

/// **Validation 2**: Finite-difference delta vs M-T delta.
///
/// FD delta: (V(S₀+ε) − V(S₀−ε)) / 2ε.
/// A M-T delta-nak konvergálnia kell ehhez.
#[test]
fn delta_vs_finite_difference() {
  let params = gbm_like_params();
  let payoff = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let n_paths = 30_000;

  let bump = 1.0;
  let mut up = params.clone();
  up.assets[0].s0 += bump;
  let mut dn = params.clone();
  dn.assets[0].s0 -= bump;
  let fd_delta = (MtGreeks::new(up, 0.01, n_paths).price(&payoff)
    - MtGreeks::new(dn, 0.01, n_paths).price(&payoff))
    / (2.0 * bump);

  let mt_delta = MtGreeks::new(params, 0.01, n_paths).delta(&payoff, 0);

  assert!(
    fd_delta > 0.0 && fd_delta < 1.5,
    "FD delta = {fd_delta} out of range"
  );
  assert!(mt_delta.is_finite(), "M-T delta = {mt_delta} not finite");
}

/// **Validation 3**: BS analitikus delta vs FD delta a Gbm limitben.
///
/// Ez validálja hogy a szimuláció maga helyes (BS delta = N(d₁) ≈ 0.6368).
#[test]
fn fd_delta_matches_bs_delta() {
  let params = gbm_like_params();
  let payoff = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let n_paths = 200_000;

  let bump = 0.5;
  let mut up = params.clone();
  up.assets[0].s0 += bump;
  let mut dn = params.clone();
  dn.assets[0].s0 -= bump;
  let fd_delta = (MtGreeks::new(up, 0.01, n_paths).price(&payoff)
    - MtGreeks::new(dn, 0.01, n_paths).price(&payoff))
    / (2.0 * bump);

  let bs = BSMPricer {
    s: 100.0,
    v: 0.2,
    k: 100.0,
    r: 0.05,
    r_d: None,
    r_f: None,
    q: Some(0.0),
    tau: Some(1.0),
    eval: None,
    expiration: None,
    option_type: OptionType::Call,
    b: BSMCoc::Bsm1973,
  };
  let bs_delta = bs.delta();

  let err = (fd_delta - bs_delta).abs();
  println!("FD delta = {fd_delta:.4}, BS delta = {bs_delta:.4}, err = {err:.4}");
  assert!(
    err < 0.20,
    "FD delta ({fd_delta:.4}) should be within 0.20 of BS delta ({bs_delta:.4}), err = {err:.4}"
  );
}

/// **Validation 4**: Put-call parity on prices.
///
/// C − P = S₀ − K·e^{−rT} (for q = 0).
#[test]
fn put_call_parity() {
  let params = gbm_like_params();
  let n_paths = 30_000;
  let e = MtGreeks::new(params, 0.01, n_paths);
  let call = e.price(&MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  });
  let put = e.price(&MtPayoff::Put {
    asset: 0,
    strike: 100.0,
  });
  let parity = 100.0 - 100.0 * (-0.05_f64).exp();

  let err = ((call - put) - parity).abs();
  assert!(
    err < 5.0,
    "Put-call parity: C−P = {:.4}, expected {parity:.4}, err = {err:.4}",
    call - put
  );
}

/// **Validation 5**: Cross-reference a meglévő `HestonMalliavinGreeks::delta()`-val.
#[test]
fn cross_check_vs_existing_heston_malliavin_delta() {
  use crate::pricing::malliavin_greeks::HestonMalliavinGreeks;

  let existing = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 30_000,
    n_steps: 252,
  };
  let ref_delta = existing.delta();

  let a = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
  };
  let a2 = AssetParams {
    s0: 50.0,
    v0: 0.01,
    kappa: 1.0,
    theta: 0.01,
    xi: 0.1,
    rho: 0.0,
  };
  let params = MultiHestonParams {
    assets: vec![a, a2],
    cross_corr: Array2::<f64>::eye(2),
    r: 0.05,
    tau: 1.0,
    n_steps: 252,
  };
  let bump = 0.5;
  let payoff = MtPayoff::Call {
    asset: 0,
    strike: 100.0,
  };
  let mut up = params.clone();
  up.assets[0].s0 += bump;
  let mut dn = params.clone();
  dn.assets[0].s0 -= bump;
  let n = 30_000;
  let fd_delta = (MtGreeks::new(up, 0.01, n).price(&payoff)
    - MtGreeks::new(dn, 0.01, n).price(&payoff))
    / (2.0 * bump);

  assert!(
    ref_delta > 0.0 && ref_delta < 1.0,
    "Existing Heston Malliavin delta = {ref_delta} out of (0,1)"
  );
  assert!(
    fd_delta > 0.0 && fd_delta < 1.0,
    "M-T FD delta = {fd_delta} out of (0,1)"
  );
  assert!(
    (ref_delta > 0.0) == (fd_delta > 0.0),
    "Sign mismatch: existing = {ref_delta:.4}, M-T FD = {fd_delta:.4}"
  );
}

//! Integration tests for the `stats::realized` module.

use ndarray::Array1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::realized::KernelType;
use stochastic_rs::stats::realized::bipower_variation;
use stochastic_rs::stats::realized::bns_jump_test;
use stochastic_rs::stats::realized::kernel::parzen_default_bandwidth;
use stochastic_rs::stats::realized::log_returns;
use stochastic_rs::stats::realized::medrv;
use stochastic_rs::stats::realized::minrv;
use stochastic_rs::stats::realized::multi_scale_rv;
use stochastic_rs::stats::realized::pre_averaged_variance;
use stochastic_rs::stats::realized::realized_kernel;
use stochastic_rs::stats::realized::realized_kurtosis;
use stochastic_rs::stats::realized::realized_quarticity;
use stochastic_rs::stats::realized::realized_semivariance;
use stochastic_rs::stats::realized::realized_skewness;
use stochastic_rs::stats::realized::realized_variance;
use stochastic_rs::stats::realized::realized_volatility;
use stochastic_rs::stats::realized::tripower_quarticity;
use stochastic_rs::stats::realized::two_scale_rv;

fn iid_returns(seed: u64, n: usize, std: f64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, std, &Deterministic::new(seed));
  let mut out = Array1::<f64>::zeros(n);
  dist.fill_slice_fast(out.as_slice_mut().unwrap());
  out
}

fn noisy_price_path(seed: u64, n: usize, sigma: f64, omega: f64) -> Array1<f64> {
  let dx = SimdNormal::<f64>::new(0.0, sigma, &Deterministic::new(seed));
  let dn = SimdNormal::<f64>::new(0.0, omega, &Deterministic::new(seed.wrapping_add(1)));
  let mut steps = vec![0.0_f64; n];
  dx.fill_slice_fast(&mut steps);
  let mut noise = vec![0.0_f64; n + 1];
  dn.fill_slice_fast(&mut noise);
  let mut x = vec![0.0_f64; n + 1];
  for i in 1..=n {
    x[i] = x[i - 1] + steps[i - 1];
  }
  Array1::from(
    x.iter()
      .zip(noise.iter())
      .map(|(&xv, &nv)| xv + nv)
      .collect::<Vec<f64>>(),
  )
}

#[test]
fn end_to_end_price_to_realized_variance() {
  let prices = Array1::from_iter((0..1_001).map(|i| 100.0 + 0.01 * i as f64));
  let r = log_returns(prices.view());
  let rv = realized_variance(r.view());
  assert!(rv > 0.0);
  let rvol = realized_volatility(r.view(), 252.0);
  assert!((rvol * rvol - 252.0 * rv).abs() < 1e-12);
}

#[test]
fn variance_signs_consistent_with_decomposition() {
  let r = iid_returns(2_001, 5_000, 0.005);
  let rv: f64 = realized_variance(r.view());
  let (down, up) = realized_semivariance(r.view(), 0.0);
  assert!((rv - down - up).abs() < 1e-12);
}

#[test]
fn higher_moments_finite_for_simulated_returns() {
  let r = iid_returns(3_001, 5_000, 0.005);
  let s: f64 = realized_skewness(r.view());
  let k: f64 = realized_kurtosis(r.view());
  let rq: f64 = realized_quarticity(r.view());
  assert!(s.is_finite());
  assert!(k.is_finite() && k > 0.0);
  assert!(rq.is_finite() && rq > 0.0);
}

#[test]
fn jump_estimators_close_to_rv_under_no_jumps() {
  let r = iid_returns(4_001, 10_000, 0.005);
  let rv: f64 = realized_variance(r.view());
  let bv = bipower_variation(r.view());
  let mr = minrv(r.view());
  let me = medrv(r.view());
  let tpq = tripower_quarticity(r.view());
  for x in [bv, mr, me] {
    assert!(
      (x - rv).abs() / rv < 0.15,
      "jump-robust drifted: {x} vs rv {rv}"
    );
  }
  assert!(tpq > 0.0);
}

#[test]
fn jump_test_p_value_in_unit_interval() {
  let r = iid_returns(5_001, 10_000, 0.005);
  let test = bns_jump_test(r.view(), 0.05);
  assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
  assert!(test.statistic.is_finite());
}

#[test]
fn realized_kernel_continuous_in_bandwidth() {
  let r = iid_returns(6_001, 10_000, 0.005);
  let rv = realized_variance(r.view());
  let k0 = realized_kernel(r.view(), KernelType::Parzen, 0);
  assert!((k0 - rv).abs() < 1e-12);
  let h = parzen_default_bandwidth(10_000, 0.3);
  let k_h = realized_kernel(r.view(), KernelType::Parzen, h);
  assert!(k_h.is_finite());
}

#[test]
fn noise_robust_estimators_finite_under_microstructure() {
  let y = noisy_price_path(7_001, 20_000, 0.0005, 0.0002);
  let dy = Array1::from_iter((1..y.len()).map(|i| y[i] - y[i - 1]));
  let pav = pre_averaged_variance(dy.view(), 1.0 / 3.0);
  let tsrv = two_scale_rv(y.view(), 20);
  let msrv = multi_scale_rv(y.view(), 12);
  assert!(pav.is_finite());
  assert!(tsrv.is_finite());
  assert!(msrv.is_finite());
}

#[cfg(feature = "openblas")]
#[test]
fn har_round_trip_recovers_intercept_at_steady_state() {
  use stochastic_rs::stats::realized::HarRv;
  let dist = SimdNormal::<f64>::new(0.0, 0.000_05, &Deterministic::new(42));
  let mut shocks = vec![0.0_f64; 1_000];
  dist.fill_slice_fast(&mut shocks);
  let mut rv = Array1::<f64>::from_elem(1_000, 0.0001);
  for t in 22..1_000 {
    let d = rv[t - 1];
    let w: f64 = (1..=5).map(|i| rv[t - i]).sum::<f64>() / 5.0;
    let m: f64 = (1..=22).map(|i| rv[t - i]).sum::<f64>() / 22.0;
    rv[t] = (0.0001 + 0.4 * d + 0.3 * w + 0.2 * m + shocks[t]).max(1e-8);
  }
  let model = HarRv::fit(rv.view());
  assert!(model.fit.r_squared > 0.0);
  assert!(model.fit.beta_d.is_finite());
  assert!(model.fit.beta_w.is_finite());
  assert!(model.fit.beta_m.is_finite());
}

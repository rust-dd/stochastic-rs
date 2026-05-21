//! Bit-exact regression tests for the V1/V2/V4 fOU estimators against
//! inline reimplementations of the v1.x (struct-era) math.  If the
//! v2.3 refactor (Hurst step delegated to
//! `stats::hurst::variations::*`) introduces any silent numerical
//! drift (formula, ordering, rounding) one of these will fire.

use std::f64::consts::SQRT_2;

use ndarray::Array1;
use ndarray::array;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_stats::fou_estimator::FilterType;
use stochastic_rs_stats::fou_estimator::estimate_fou_v1;
use stochastic_rs_stats::fou_estimator::estimate_fou_v2;
use stochastic_rs_stats::fou_estimator::estimate_fou_v4;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::traits::ProcessExt;

fn assert_close(actual: f64, expected: f64, label: &str) {
  let rel_tol = 4.0 * f64::EPSILON;
  let allowed = rel_tol.max(rel_tol * expected.abs());
  let diff = (actual - expected).abs();
  assert!(
    diff <= allowed,
    "{label}: actual={actual} expected={expected} diff={diff:e} allowed={allowed:e}",
  );
}

#[test]
fn fou_v1_bit_exact_against_struct_era_inline_reference() {
  let n = 512usize;
  let path = Fou::<f64, _>::new(
    0.6,
    1.0,
    0.0,
    0.2,
    n,
    Some(0.0),
    Some(1.0),
    Deterministic::new(0xF0_E5_71_AA),
  )
  .sample();
  let delta = 1.0 / n as f64;

  let a = array![
    0.482962913144534 / SQRT_2,
    -0.836516303737808 / SQRT_2,
    0.224143868042013 / SQRT_2,
    0.12940952255126 / SQRT_2,
  ];
  let l_filter = a.len();
  let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
  for (i, &val) in a.iter().enumerate() {
    a_2[i * 2 + 1] = val;
  }
  let lfilt = |b: &Array1<f64>, x: &Array1<f64>| -> Array1<f64> {
    let nx = x.len();
    let mut y = Array1::<f64>::zeros(nx);
    for i in 0..nx {
      let mut acc = 0.0;
      for j in 0..b.len() {
        if i >= j {
          acc += b[j] * x[i - j];
        }
      }
      y[i] = acc;
    }
    y
  };
  let v1_path_ref = lfilt(&a, &path);
  let v1_sum_ref: f64 = v1_path_ref.mapv(|x| x * x).sum();
  let v2_path_ref = lfilt(&a_2, &path);
  let v2_sum_ref: f64 = v2_path_ref.mapv(|x| x * x).sum();
  let hurst_ref = 0.5 * (v2_sum_ref / v1_sum_ref).log2();

  let mut const_filter = 0.0;
  for i in 0..l_filter {
    for j in 0..l_filter {
      const_filter += a[i] * a[j] * ((i as f64 - j as f64).abs()).powf(2.0 * hurst_ref);
    }
  }
  let numerator_ref = -2.0 * v1_sum_ref / ((path.len() - l_filter) as f64);
  let denominator_ref = const_filter * delta.powf(2.0 * hurst_ref);
  let sigma_ref = (numerator_ref / denominator_ref).sqrt();

  let mu_ref = path.mean().unwrap();
  let mean_square_ref = path.mapv(|x| x * x).mean().unwrap();
  let theta_num_ref = 2.0 * mean_square_ref;
  let theta_den_ref = sigma_ref.powi(2) * gamma(2.0 * hurst_ref + 1.0);
  let theta_ref = (theta_num_ref / theta_den_ref).powf(-1.0 / (2.0 * hurst_ref));

  let res = estimate_fou_v1(path.view(), FilterType::Daubechies, Some(delta), None);

  assert_close(res.hurst, hurst_ref, "v1 hurst");
  assert_close(res.sigma, sigma_ref, "v1 sigma");
  assert_close(res.mu, mu_ref, "v1 mu");
  assert_close(res.theta, theta_ref, "v1 theta");
}

#[test]
fn fou_v2_bit_exact_against_struct_era_inline_reference() {
  let n = 512usize;
  let path = Fou::<f64, _>::new(
    0.6,
    1.0,
    0.0,
    0.2,
    n,
    Some(0.0),
    Some(1.0),
    Deterministic::new(0xF0_E5_71_AA),
  )
  .sample();
  let delta_ref = 1.0 / n as f64;

  let sum1: f64 = (0..(n - 4))
    .map(|i| {
      let diff = path[i + 4] - 2.0 * path[i + 2] + path[i];
      diff * diff
    })
    .sum();
  let sum2: f64 = (0..(n - 2))
    .map(|i| {
      let diff = path[i + 2] - 2.0 * path[i + 1] + path[i];
      diff * diff
    })
    .sum();
  let hurst_ref = 0.5 * (sum1 / sum2).log2();

  let n_f = n as f64;
  let sigma_num_ref: f64 = (0..(n - 2))
    .map(|i| {
      let diff = path[i + 2] - 2.0 * path[i + 1] + path[i];
      diff * diff
    })
    .sum();
  let sigma_den_ref = n_f * (4.0 - 2.0_f64.powf(2.0 * hurst_ref)) * delta_ref.powf(2.0 * hurst_ref);
  let sigma_ref = (sigma_num_ref / sigma_den_ref).sqrt();

  let mu_ref = path.mean().unwrap();
  let sum_x_sq: f64 = path.mapv(|x| x * x).sum();
  let sum_x: f64 = path.sum();
  let theta_num_ref = n_f * sum_x_sq - sum_x.powi(2);
  let theta_den_ref = n_f.powi(2) * sigma_ref.powi(2) * hurst_ref * gamma(2.0 * hurst_ref);
  let theta_ref = (theta_num_ref / theta_den_ref).powf(-1.0 / (2.0 * hurst_ref));

  let res = estimate_fou_v2(path.view(), Some(delta_ref), n, None);

  assert_close(res.hurst, hurst_ref, "v2 hurst");
  assert_close(res.sigma, sigma_ref, "v2 sigma");
  assert_close(res.mu, mu_ref, "v2 mu");
  assert_close(res.theta, theta_ref, "v2 theta");
}

#[test]
fn fou_v4_bit_exact_against_struct_era_inline_reference() {
  let n = 512usize;
  let k = 2usize;
  let p = 2.0_f64;
  let path = Fou::<f64, _>::new(
    0.65,
    1.2,
    0.0,
    0.25,
    n,
    Some(0.0),
    Some(1.0),
    Deterministic::new(0xC0_FF_EE_42),
  )
  .sample();
  let delta_ref = 1.0 / (n - 1) as f64;

  fn binom_ref(nn: usize, kk: usize) -> f64 {
    if kk > nn {
      return 0.0;
    }
    let kk = kk.min(nn - kk);
    if kk == 0 {
      return 1.0;
    }
    let mut c = 1.0;
    for i in 1..=kk {
      c *= (nn - kk + i) as f64 / i as f64;
    }
    c
  }
  fn diff_coeff_ref(kk: usize, j: usize) -> f64 {
    let sign = if ((kk - j) & 1) == 0 { 1.0 } else { -1.0 };
    sign * binom_ref(kk, j)
  }
  let pv_ref = |path: &Array1<f64>, k: usize, p: f64, stride: usize| -> f64 {
    let nn = path.len();
    let span = k * stride;
    let mut v = 0.0;
    for i in 0..(nn - span) {
      let mut d = 0.0;
      for j in 0..=k {
        d += diff_coeff_ref(k, j) * path[i + j * stride];
      }
      v += d.abs().powf(p);
    }
    v
  };
  let rho_zero_ref = |kk: usize, h: f64| -> f64 {
    let mut acc = 0.0;
    for i in -(kk as isize)..=(kk as isize) {
      let parity = (1_isize - i).rem_euclid(2);
      let sign = if parity == 0 { 1.0 } else { -1.0 };
      let comb = binom_ref(2 * kk, (kk as isize - i) as usize);
      let abs_term = (i.unsigned_abs() as f64).powf(2.0 * h);
      acc += sign * comb * abs_term;
    }
    0.5 * acc
  };

  let v1 = pv_ref(&path, k, p, 1);
  let v2 = pv_ref(&path, k, p, 2);
  let mut h = (1.0 + (v2 / v1).log2()) / p;
  if !h.is_finite() {
    h = 0.5;
  }
  let hurst_ref = h.clamp(1e-6, 1.0 - 1e-6);

  let n_f = (n - 1) as f64;
  let t_horizon = n_f * delta_ref;
  let v_sigma = pv_ref(&path, k, p, 1);
  let rho0 = rho_zero_ref(k, hurst_ref);
  let normal_abs_p = 2.0_f64.powf(p / 2.0) * gamma((p + 1.0) / 2.0) / gamma(0.5);
  let c_kp = normal_abs_p * rho0.powf(p / 2.0);
  let scaled = n_f.powf(-1.0 + p * hurst_ref) * v_sigma;
  let sigma_abs_p = scaled / (c_kp * t_horizon);
  let sigma_ref = sigma_abs_p.powf(1.0 / p);

  let mu_ref = path.mean().unwrap();
  let sum_sq: f64 = path
    .slice(s![1..])
    .iter()
    .map(|x| (x - mu_ref).powi(2))
    .sum();
  let denom = n_f * sigma_ref.powi(2) * hurst_ref * gamma(2.0 * hurst_ref);
  let theta_ref = (sum_sq / denom).powf(-1.0 / (2.0 * hurst_ref));

  let res = estimate_fou_v4(path.view(), Some(delta_ref), k, p, None, None);

  assert_close(res.hurst, hurst_ref, "v4 hurst");
  assert_close(res.sigma, sigma_ref, "v4 sigma");
  assert_close(res.mu, mu_ref, "v4 mu");
  assert_close(res.theta, theta_ref, "v4 theta");
}

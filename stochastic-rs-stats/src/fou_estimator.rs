//! # fOU Estimator
//!
//! $$
//! (\hat\kappa,\hat\theta,\hat\sigma)=\arg\min \sum_i\left(X_{t_{i+1}}-X_{t_i}-\kappa(\theta-X_{t_i})\Delta t\right)^2
//! $$
//!
//! ## Estimators
//!
//! Free-function estimators that take zero-copy
//! [`ndarray::ArrayView1<f64>`] paths and return a [`FouEstimateResult`].
//! The Hurst step in each estimator is delegated to
//! [`crate::hurst::variations`] (so swap-in via
//! [`crate::hurst::HurstEstimator`] is possible); this module owns the
//! `sigma`, `mu`, `theta` recovery formulas that are estimator-specific
//! to the fOU SDE.
//!
//! - [`estimate_fou_v1`] — Daubechies-filter-based fOU estimator
//!   (Coeurjolly variant).  Hurst from
//!   [`crate::hurst::variations::daubechies_h_inner`].
//! - [`estimate_fou_v2`] — moments-based fOU estimator (no linear
//!   filters).  Hurst from
//!   [`crate::hurst::variations::central_diff_h_inner`].
//! - [`estimate_fou_v4`] — high-frequency `k`-th-order `p`-power-variation
//!   estimator (arXiv:1703.09372).  Hurst from
//!   [`crate::hurst::variations::power_variation_h_inner`].
//!
//! In addition, [`FOUParameterEstimationV3`] is a Monte-Carlo
//! verification helper that simulates an fGN and then re-estimates;
//! useful for measuring estimator bias / variance against ground
//! truth, **not** for fitting real time series.
//!
//! For a paper-grade Whittle estimator on real data, see
//! [`crate::hurst::Whittle`] (Paxson + L-BFGS-B).

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_stochastic::noise::fgn::Fgn;

use crate::hurst::variations::a2_coefficients;
use crate::hurst::variations::daubechies_coeffs;
use crate::hurst::variations::lfilter_f64;
use crate::hurst::variations::power_variation;
use crate::traits::ProcessExt;

/// Structured result of an fOU parameter estimator.
///
/// Build with [`From<(f64, f64, f64, f64)>`] — the estimator output tuple
/// is converted positionally as `(hurst, sigma, mu, theta)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FouEstimateResult {
  pub hurst: f64,
  pub sigma: f64,
  pub mu: f64,
  pub theta: f64,
}

impl From<(f64, f64, f64, f64)> for FouEstimateResult {
  fn from((hurst, sigma, mu, theta): (f64, f64, f64, f64)) -> Self {
    Self {
      hurst,
      sigma,
      mu,
      theta,
    }
  }
}

impl From<FouEstimateResult> for (f64, f64, f64, f64) {
  fn from(r: FouEstimateResult) -> Self {
    (r.hurst, r.sigma, r.mu, r.theta)
  }
}

/// Linear filter selection for [`estimate_fou_v1`].
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum FilterType {
  Daubechies,
}

/// V1 fOU estimator — Daubechies-filter-based variations approach.
///
/// Hurst from `H = ½ log₂(V₂ / V₁)` where `V₁ = Σ (a · X)²` and `V₂ = Σ
/// (a₂ · X)²` with `a` the Daubechies-4 low-pass filter and `a₂` the
/// zero-interlaced (dilated) kernel; `sigma`, `mu`, `theta` follow
/// from method-of-moments expressions.
pub fn estimate_fou_v1(
  path: ArrayView1<f64>,
  filter_type: FilterType,
  delta: Option<f64>,
  hurst_override: Option<f64>,
) -> FouEstimateResult {
  let series_length = path.len();
  let (a, l_filter) = match filter_type {
    FilterType::Daubechies => {
      let a = daubechies_coeffs();
      let l = a.len();
      (a, l)
    }
  };
  let a_2 = a2_coefficients(&a);

  let v1_path = lfilter_f64::<f64>(&a, &Array1::from_vec(vec![1.0]), path);
  let v1_sum: f64 = v1_path.mapv(|x| x * x).sum();
  let v2_path = lfilter_f64::<f64>(&a_2, &Array1::from_vec(vec![1.0]), path);
  let v2_sum: f64 = v2_path.mapv(|x| x * x).sum();

  let hurst = hurst_override.unwrap_or_else(|| 0.5 * (v2_sum / v1_sum).log2());
  let dt = delta.unwrap_or(1.0 / series_length as f64);

  let mut const_filter = 0.0;
  for i in 0..l_filter {
    for j in 0..l_filter {
      const_filter += a[i] * a[j] * ((i as f64 - j as f64).abs()).powf(2.0 * hurst);
    }
  }
  let numerator = -2.0 * v1_sum / ((series_length - l_filter) as f64);
  let denominator = const_filter * dt.powf(2.0 * hurst);
  let sigma = (numerator / denominator).sqrt();

  let mu = path.mean().unwrap();

  let mean_square = path.mapv(|x| x * x).mean().unwrap();
  let theta_num = 2.0 * mean_square;
  let theta_den = sigma.powi(2) * gamma(2.0 * hurst + 1.0);
  let theta = (theta_num / theta_den).powf(-1.0 / (2.0 * hurst));

  FouEstimateResult {
    hurst,
    sigma,
    mu,
    theta,
  }
}

/// V2 fOU estimator — moments-based (no linear filters).
///
/// `H = ½ log₂(Σ(Δ⁴X)² / Σ(Δ²X)²)` with `Δⁿ` the n-th-order central
/// finite difference; `sigma` from `(4 - 2^{2H})` scaling; `mu` is the
/// path mean; `theta` from an ergodic-moment estimator.
pub fn estimate_fou_v2(
  path: ArrayView1<f64>,
  delta: Option<f64>,
  series_length: usize,
  hurst_override: Option<f64>,
) -> FouEstimateResult {
  assert_eq!(
    path.len(),
    series_length,
    "estimate_fou_v2: path.len() ({}) must equal series_length ({})",
    path.len(),
    series_length
  );
  assert!(
    series_length >= 5,
    "estimate_fou_v2: series_length must be >= 5 for fourth-difference variation"
  );

  let n = series_length;
  let n_f = n as f64;

  let hurst = hurst_override.unwrap_or_else(|| {
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
    0.5 * (sum1 / sum2).log2()
  });

  let dt = delta.unwrap_or(1.0 / n_f);

  let sigma_num: f64 = (0..(n - 2))
    .map(|i| {
      let diff = path[i + 2] - 2.0 * path[i + 1] + path[i];
      diff * diff
    })
    .sum();
  let sigma_den = n_f * (4.0 - 2.0_f64.powf(2.0 * hurst)) * dt.powf(2.0 * hurst);
  let sigma = (sigma_num / sigma_den).sqrt();

  let mu = path.mean().unwrap();

  let sum_x_sq: f64 = path.mapv(|x| x * x).sum();
  let sum_x: f64 = path.sum();
  let theta_num = n_f * sum_x_sq - sum_x.powi(2);
  let theta_den = n_f.powi(2) * sigma.powi(2) * hurst * gamma(2.0 * hurst);
  let theta = (theta_num / theta_den).powf(-1.0 / (2.0 * hurst));

  FouEstimateResult {
    hurst,
    sigma,
    mu,
    theta,
  }
}

/// **Simulation+estimation round-trip helper (V3) — different shape from
/// V1/V2/V4 by design.**
///
/// Constructor takes the *true* parameters (`hurst`, `sigma`, `alpha`,
/// `mu`), `get_path` internally generates an fGN sample via
/// `Fgn::new(...)` using those parameters, and the estimator stage
/// recovers them. Useful for measuring estimator bias / variance against
/// a known ground truth, NOT for fitting a real observed time series.
pub struct FOUParameterEstimationV3 {
  alpha: f64,
  mu: f64,
  sigma: f64,
  initial_value: f64,
  T: f64,
  delta: f64,
  series_length: usize,
  hurst: f64,
  path: Option<Array1<f64>>,
  estimated_hurst: Option<f64>,
  estimated_sigma: Option<f64>,
  estimated_mu: Option<f64>,
  estimated_alpha: Option<f64>,
}

impl FOUParameterEstimationV3 {
  pub fn new(
    series_length: usize,
    hurst: f64,
    sigma: f64,
    alpha: f64,
    mu: f64,
    initial_value: f64,
    T: f64,
    delta: f64,
  ) -> Self {
    Self {
      alpha,
      mu,
      sigma,
      initial_value,
      T,
      delta,
      series_length,
      hurst,
      path: None,
      estimated_hurst: None,
      estimated_sigma: None,
      estimated_mu: None,
      estimated_alpha: None,
    }
  }

  pub fn estimate_parameters(&mut self) -> FouEstimateResult {
    self.get_path();
    self.hurst_estimator();
    self.sigma_estimator();
    self.mu_estimator();
    self.alpha_estimator();
    FouEstimateResult::from((
      self.estimated_hurst.unwrap(),
      self.estimated_sigma.unwrap(),
      self.estimated_mu.unwrap(),
      self.estimated_alpha.unwrap(),
    ))
  }

  fn get_path(&mut self) {
    let m_sub = 8;
    let gamma = self.delta / m_sub as f64;
    let fgn_length = self.series_length * m_sub;
    let fgn = Fgn::new(self.hurst, fgn_length - 1, Some(self.T), Unseeded);
    let fgn_sample = fgn.sample();

    let mut full_fou = Array1::zeros(fgn_length);
    full_fou[0] = self.initial_value;
    for i in 1..fgn_length {
      full_fou[i] = full_fou[i - 1]
        + self.alpha * (self.mu - full_fou[i - 1]) * gamma
        + self.sigma * fgn_sample[i - 1];
    }

    let mut fou = Array1::zeros(self.series_length);
    fou[0] = self.initial_value;
    for i in 1..self.series_length {
      let start = (i - 1) * m_sub;
      let end = i * m_sub;
      let sum_sub_series: f64 = full_fou.slice(s![start..end]).sum() * gamma / m_sub as f64;
      fou[i] = full_fou[end - 1] + self.alpha * sum_sub_series;
    }
    self.path = Some(fou);
  }

  fn hurst_estimator(&mut self) {
    let X = self.path.as_ref().unwrap();
    let N = self.series_length;
    let sum1: f64 = (0..(N - 4))
      .map(|i| {
        let diff = X[i + 4] - 2.0 * X[i + 2] + X[i];
        diff * diff
      })
      .sum();
    let sum2: f64 = (0..(N - 2))
      .map(|i| {
        let diff = X[i + 2] - 2.0 * X[i + 1] + X[i];
        diff * diff
      })
      .sum();
    self.estimated_hurst = Some(0.5 * (sum1 / sum2).log2());
  }

  fn sigma_estimator(&mut self) {
    let H = self.estimated_hurst.unwrap();
    let X = self.path.as_ref().unwrap();
    let N = self.series_length as f64;
    let delta = self.delta;
    let numerator: f64 = (0..(self.series_length - 2))
      .map(|i| {
        let diff = X[i + 2] - 2.0 * X[i + 1] + X[i];
        diff * diff
      })
      .sum();
    let denominator = N * (4.0 - 2.0_f64.powf(2.0 * H)) * delta.powf(2.0 * H);
    self.estimated_sigma = Some((numerator / denominator).sqrt());
  }

  fn mu_estimator(&mut self) {
    let X = self.path.as_ref().unwrap();
    self.estimated_mu = Some(X.mean().unwrap());
  }

  fn alpha_estimator(&mut self) {
    let X = self.path.as_ref().unwrap();
    let H = self.estimated_hurst.unwrap();
    let N = self.series_length as f64;
    let sigma = self.estimated_sigma.unwrap();
    let sum_X_squared = X.mapv(|x| x * x).sum();
    let sum_X = X.sum();
    let numerator = N * sum_X_squared - sum_X.powi(2);
    let denominator = N.powi(2) * sigma.powi(2) * H * gamma(2.0 * H);
    self.estimated_alpha = Some((numerator / denominator).powf(-1.0 / (2.0 * H)));
  }
}

/// V4 fOU estimator — high-frequency `k`-th-order `p`-power-variation
/// estimator (arXiv:1703.09372).
///
/// `H` follows from the scale-ratio of `V_{k,p}` at strides 1 and 2:
/// `V_{k,p}(2)/V_{k,p}(1) ≈ 2^{pH-1}`. `sigma` follows from
/// `n^{-1+pH} V_{k,p} / (c_{k,p} T) = |sigma|^p`. `theta` is recovered
/// from the discrete ergodic estimator
/// `theta = ((1/(n σ² H Γ(2H))) Σ (X − μ)²)^{-1/(2H)}`.
pub fn estimate_fou_v4(
  path: ArrayView1<f64>,
  delta: Option<f64>,
  k: usize,
  p: f64,
  hurst_override: Option<f64>,
  sigma_override: Option<f64>,
) -> FouEstimateResult {
  assert!(
    path.len() >= k + 2,
    "estimate_fou_v4: path length ({}) must be at least k + 2 = {}",
    path.len(),
    k + 2
  );
  assert!(k >= 1, "estimate_fou_v4: k must be at least 1");
  assert!(
    p.is_finite() && p > 0.0,
    "estimate_fou_v4: p must be positive"
  );
  if let Some(dt) = delta {
    assert!(
      dt.is_finite() && dt > 0.0,
      "estimate_fou_v4: delta must be positive"
    );
  }
  if let Some(h) = hurst_override {
    assert!(
      h.is_finite() && (0.0..1.0).contains(&h),
      "estimate_fou_v4: hurst_override must be in (0, 1)"
    );
  }
  if let Some(s) = sigma_override {
    assert!(
      s.is_finite() && s > 0.0,
      "estimate_fou_v4: sigma_override must be positive"
    );
  }

  let dt = delta.unwrap_or(1.0 / (path.len() - 1) as f64);

  let hurst = hurst_override.unwrap_or_else(|| {
    let v1 = power_variation::<f64>(path, k, p, 1);
    let v2 = power_variation::<f64>(path, k, p, 2);
    assert!(
      v1 > 0.0 && v2 > 0.0,
      "estimate_fou_v4: power variations must be positive"
    );
    let mut h = (1.0 + (v2 / v1).log2()) / p;
    if !h.is_finite() {
      h = 0.5;
    }
    h.clamp(1e-6, 1.0 - 1e-6)
  });

  let sigma = sigma_override.unwrap_or_else(|| {
    let n = (path.len() - 1) as f64;
    let t_horizon = n * dt;
    let v = power_variation::<f64>(path, k, p, 1);
    let c_kp = c_k_p(k, p, hurst);
    let scaled = n.powf(-1.0 + p * hurst) * v;
    let sigma_abs_p = scaled / (c_kp * t_horizon);
    assert!(
      sigma_abs_p.is_finite() && sigma_abs_p > 0.0,
      "estimate_fou_v4: estimated sigma^p must be positive"
    );
    sigma_abs_p.powf(1.0 / p)
  });

  let mu = path.mean().unwrap();
  let n = (path.len() - 1) as f64;
  let sum_sq: f64 = path.slice(s![1..]).iter().map(|x| (x - mu).powi(2)).sum();
  let denom = n * sigma.powi(2) * hurst * gamma(2.0 * hurst);
  let base = sum_sq / denom;
  assert!(
    base.is_finite() && base > 0.0,
    "estimate_fou_v4: invalid theta base value"
  );
  let theta = base.powf(-1.0 / (2.0 * hurst));
  assert!(
    theta.is_finite() && theta > 0.0,
    "estimate_fou_v4: estimated theta must be positive"
  );

  FouEstimateResult {
    hurst,
    sigma,
    mu,
    theta,
  }
}

fn c_k_p(k: usize, p: f64, h: f64) -> f64 {
  let rho0 = rho_k_h_zero(k, h);
  assert!(
    rho0.is_finite() && rho0 > 0.0,
    "estimate_fou_v4: rho_(k,H)(0) must be positive"
  );
  let normal_abs_p = 2.0_f64.powf(p / 2.0) * gamma((p + 1.0) / 2.0) / gamma(0.5);
  normal_abs_p * rho0.powf(p / 2.0)
}

fn rho_k_h_zero(k: usize, h: f64) -> f64 {
  let mut acc = 0.0;
  for i in -(k as isize)..=(k as isize) {
    let parity = (1_isize - i).rem_euclid(2);
    let sign = if parity == 0 { 1.0 } else { -1.0 };
    let comb = crate::hurst::variations::binomial(2 * k, (k as isize - i) as usize);
    let abs_term = (i.unsigned_abs() as f64).powf(2.0 * h);
    acc += sign * comb * abs_term;
  }
  0.5 * acc
}

#[cfg(test)]
mod tests {
  use stochastic_rs_stochastic::diffusion::fou::Fou;

  use super::*;

  #[test]
  fn fou_v4_returns_finite_params_with_fixed_hurst() {
    let n = 1024usize;
    let h_true = 0.7;
    let path = Fou::<f64>::new(h_true, 1.5, 0.0, 0.35, n, Some(0.0), Some(1.0), Unseeded).sample();
    let res = estimate_fou_v4(
      path.view(),
      Some(1.0 / (n - 1) as f64),
      2,
      2.0,
      Some(h_true),
      None,
    );
    assert!(res.hurst.is_finite() && (0.0..1.0).contains(&res.hurst));
    assert!(res.sigma.is_finite() && res.sigma > 0.0);
    assert!(res.theta.is_finite() && res.theta > 0.0);
  }

  #[test]
  fn fou_v4_estimated_hurst_in_range_when_not_fixed() {
    let n = 768usize;
    let path = Fou::<f64>::new(0.65, 1.2, 0.0, 0.25, n, Some(0.0), Some(1.0), Unseeded).sample();
    let res = estimate_fou_v4(path.view(), Some(1.0 / (n - 1) as f64), 2, 2.0, None, None);
    assert!((0.0..1.0).contains(&res.hurst));
    assert!(res.sigma > 0.0 && res.theta > 0.0);
  }

  #[test]
  fn fou_v1_runs_on_fbm_path() {
    let n = 1024usize;
    let path = Fou::<f64>::new(0.7, 1.5, 0.0, 0.35, n, Some(0.0), Some(1.0), Unseeded).sample();
    let res = estimate_fou_v1(path.view(), FilterType::Daubechies, None, None);
    assert!(res.hurst.is_finite() && res.sigma > 0.0);
  }

  #[test]
  fn fou_v2_runs_on_fbm_path() {
    let n = 1024usize;
    let path = Fou::<f64>::new(0.7, 1.5, 0.0, 0.35, n, Some(0.0), Some(1.0), Unseeded).sample();
    let res = estimate_fou_v2(path.view(), None, n, None);
    assert!(res.hurst.is_finite() && res.sigma > 0.0);
  }

  #[test]
  fn fou_v1_with_hurst_override_skips_estimation() {
    let n = 512usize;
    let path = Fou::<f64>::new(0.6, 1.0, 0.0, 0.2, n, Some(0.0), Some(1.0), Unseeded).sample();
    let h_pinned = 0.42;
    let res = estimate_fou_v1(path.view(), FilterType::Daubechies, None, Some(h_pinned));
    assert_eq!(res.hurst, h_pinned);
  }

  #[test]
  fn fou_v3_round_trip_recovers_in_range_params() {
    let mut v3 = FOUParameterEstimationV3::new(512, 0.65, 0.3, 1.5, 0.1, 0.0, 1.0, 1.0 / 511.0);
    let res = v3.estimate_parameters();
    assert!(res.hurst.is_finite() && (0.0..1.0).contains(&res.hurst));
    assert!(res.sigma > 0.0 && res.theta > 0.0);
  }

  #[test]
  #[should_panic(expected = "must be >= 5")]
  fn fou_v2_rejects_short_path() {
    let path = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let _ = estimate_fou_v2(path.view(), None, path.len(), None);
  }

  #[test]
  fn fou_estimate_result_round_trips_tuple() {
    let r: FouEstimateResult = (0.7, 0.3, 0.05, 1.2).into();
    assert_eq!(r.hurst, 0.7);
    assert_eq!(r.sigma, 0.3);
    let t: (f64, f64, f64, f64) = r.into();
    assert_eq!(t, (0.7, 0.3, 0.05, 1.2));
  }
}

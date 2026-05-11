//! # fOU Estimator
//!
//! $$
//! (\hat\kappa,\hat\theta,\hat\sigma)=\arg\min \sum_i\left(X_{t_{i+1}}-X_{t_i}-\kappa(\theta-X_{t_i})\Delta t\right)^2
//! $$
//!
//! ## Estimators
//!
//! Three free-function estimators that take zero-copy
//! [`ndarray::ArrayView1<f64>`] paths and return a
//! [`FouEstimateResult`]:
//!
//! - [`estimate_fou_v1`] — Daubechies-filter-based fOU estimator
//!   (Coeurjolly variant).
//! - [`estimate_fou_v2`] — moments-based fOU estimator
//!   (closed-form moment matching, no linear filters).
//! - [`estimate_fou_v4`] — high-frequency `k`-th-order `p`-power-variation
//!   estimator. Reference: arXiv:1703.09372.
//!
//! In addition, [`FOUParameterEstimationV3`] is a struct (different shape
//! by design): a Monte-Carlo verification helper that internally generates
//! an fGN sample using the parameters supplied to its constructor and then
//! re-estimates them. Useful for measuring estimator bias / variance
//! against known ground truth, **not** for fitting a real time series.
//!
//! For a paper-grade Whittle estimator on real data, see
//! [`crate::fukasawa_hurst::estimate`] (Paxson + L-BFGS-B). The free fns in
//! this module remain available for educational comparison and as
//! lightweight alternatives.
use std::f64::consts::SQRT_2;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::array;
use ndarray::s;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_stochastic::noise::fgn::Fgn;

use crate::traits::ProcessExt;

/// Structured result of an fOU parameter estimator.
///
/// Build with [`From<(f64, f64, f64, f64)>`] — the estimator output tuple
/// is converted positionally as `(hurst, sigma, mu, theta)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FouEstimateResult {
  /// Hurst exponent $H \in (0, 1)$.
  pub hurst: f64,
  /// Diffusion / noise scale $\sigma$.
  pub sigma: f64,
  /// Long-run mean $\mu$.
  pub mu: f64,
  /// Mean-reversion speed $\theta$.
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

#[inline]
fn daubechies_coeffs() -> Array1<f64> {
  array![
    0.482962913144534 / SQRT_2,
    -0.836516303737808 / SQRT_2,
    0.224143868042013 / SQRT_2,
    0.12940952255126 / SQRT_2,
  ]
}

#[inline]
fn filter_coefficients(filter_type: FilterType) -> (Array1<f64>, usize) {
  let a: Array1<f64> = match filter_type {
    FilterType::Daubechies => daubechies_coeffs(),
  };
  let l = a.len();
  (a, l)
}

#[inline]
fn a2_coefficients(a: &Array1<f64>) -> Array1<f64> {
  let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
  for (i, &val) in a.iter().enumerate() {
    a_2[i * 2 + 1] = val;
  }
  a_2
}

/// Direct-form-II IIR filter `lfilter` (subset of scipy.signal.lfilter):
/// `y[i] = Σ_j b[j] x[i-j] − Σ_{j≥1} a[j] y[i-j]` with zero initial conditions.
fn lfilter(b: &Array1<f64>, a: &Array1<f64>, x: ArrayView1<f64>) -> Array1<f64> {
  let n = x.len();
  let mut y = Array1::<f64>::zeros(n);
  for i in 0..n {
    let mut acc = 0.0;
    for j in 0..b.len() {
      if i >= j {
        acc += b[j] * x[i - j];
      }
    }
    for j in 1..a.len() {
      if i >= j {
        acc -= a[j] * y[i - j];
      }
    }
    y[i] = acc;
  }
  y
}

/// V1 fOU estimator — Daubechies-filter-based variations approach.
///
/// Computes `V1 = Σ (a * X)²` and `V2 = Σ (a₂ * X)²` where `a` is the
/// chosen low-pass filter (Daubechies-4) and `a₂` is the dilated kernel
/// with zeros inserted between coefficients. The Hurst exponent follows
/// from `H = ½ log₂(V₂/V₁)`. `sigma`, `mu`, `theta` are then recovered
/// from method-of-moments expressions.
///
/// Pass `Some(h)` for `hurst_override` to skip the H estimation step
/// (useful when calibrating with a Hurst recovered from a different,
/// more accurate estimator like
/// [`crate::fukasawa_hurst::estimate`]).
///
/// # Arguments
/// * `path` — observed equidistant trajectory.
/// * `filter_type` — currently only [`FilterType::Daubechies`].
/// * `delta` — sampling interval. If `None`, defaults to `1 / N`.
/// * `hurst_override` — externally-supplied Hurst exponent; if `None`,
///   estimated from the variations ratio.
pub fn estimate_fou_v1(
  path: ArrayView1<f64>,
  filter_type: FilterType,
  delta: Option<f64>,
  hurst_override: Option<f64>,
) -> FouEstimateResult {
  let series_length = path.len();
  let (a, l_filter) = filter_coefficients(filter_type);
  let a_2 = a2_coefficients(&a);

  let v1_path = lfilter(&a, &array![1.0], path);
  let v1_sum: f64 = v1_path.mapv(|x| x * x).sum();
  let v2_path = lfilter(&a_2, &array![1.0], path);
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
/// Estimates Hurst from the second-difference / fourth-difference power
/// ratio `H = ½ log₂(Σ(Δ⁴X)² / Σ(Δ²X)²)` where `Δⁿ` is the n-th order
/// finite difference of the path. `sigma` follows from a closed-form
/// scaling expression involving `(4 − 2^{2H})`; `mu` is the path mean;
/// `theta` follows from an ergodic-moment estimator.
///
/// # Arguments
/// * `path` — observed equidistant trajectory of length `series_length`.
/// * `delta` — sampling interval. If `None`, defaults to `1 / series_length`.
/// * `series_length` — length of the path (passed in to mirror the v1.x
///   API; must equal `path.len()`).
/// * `hurst_override` — externally-supplied Hurst exponent; if `None`,
///   estimated from the second/fourth-difference ratio.
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
/// Unlike the V1/V2/V4 free-function estimators which take a user-supplied
/// path and estimate the fOU parameters from it, V3 is a Monte-Carlo
/// verification helper: the constructor takes the *true* parameters
/// (`hurst`, `sigma`, `alpha`, `mu`), `get_path` internally generates an
/// fGN sample via `Fgn::new(hurst, …).sample()` using those parameters,
/// and the estimator stage then recovers them. Useful for measuring
/// estimator bias / variance against a known ground truth, NOT for
/// fitting a real observed time series. For real estimation on user data,
/// reach for [`estimate_fou_v1`], [`estimate_fou_v2`], [`estimate_fou_v4`],
/// or [`crate::fukasawa_hurst::estimate`].
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
  // Estimated parameters
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
    FOUParameterEstimationV3 {
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
    let M = 8;
    let gamma = self.delta / M as f64;

    let fgn_length = self.series_length * M;

    let fgn = Fgn::new(self.hurst, fgn_length - 1, Some(self.T));
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
      let start = (i - 1) * M;
      let end = i * M;
      let sum_sub_series: f64 = full_fou.slice(s![start..end]).sum() * gamma / M as f64;
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
    let estimated_hurst = 0.5 * (sum1 / sum2).log2();
    self.estimated_hurst = Some(estimated_hurst);
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
    let mean = X.mean().unwrap();
    self.estimated_mu = Some(mean);
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
    let estimated_alpha = (numerator / denominator).powf(-1.0 / (2.0 * H));
    self.estimated_alpha = Some(estimated_alpha);
  }
}

/// V4 fOU estimator — high-frequency `k`-th-order `p`-power-variation
/// estimator (arXiv:1703.09372).
///
/// `H` follows from the scale-ratio of `V_{k,p}` at strides 1 and 2:
/// `V_{k,p}(2)/V_{k,p}(1) ≈ 2^{pH−1}`. `sigma` follows from
/// `n^{−1+pH} V_{k,p} / (c_{k,p} T) = |sigma|^p`. `theta` is recovered
/// from the discrete ergodic estimator
/// `theta = ((1/(n σ² H Γ(2H))) Σ (X − μ)²)^{−1/(2H)}`.
///
/// Pass `hurst_override` / `sigma_override` to skip the corresponding
/// estimation step (useful for calibration pipelines that want to fix
/// some parameters externally).
///
/// # Arguments
/// * `path` — observed equidistant trajectory.
/// * `delta` — sampling interval. If `None`, defaults to `1 / (N − 1)`.
/// * `k` — finite-difference order (`k ≥ 1`).
/// * `p` — power for the variation `V_{k,p}` (`p > 0`).
/// * `hurst_override` — externally-supplied Hurst; if `None`, estimated.
/// * `sigma_override` — externally-supplied sigma; if `None`, estimated.
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
    let v1 = power_variation(path, k, p, 1);
    let v2 = power_variation(path, k, p, 2);
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
    let v = power_variation(path, k, p, 1);
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

fn power_variation(path: ArrayView1<f64>, k: usize, p: f64, stride: usize) -> f64 {
  let n = path.len();
  let span = k * stride;
  assert!(
    n > span,
    "estimate_fou_v4: path is too short for requested (k, stride)"
  );
  let mut v = 0.0;
  for i in 0..(n - span) {
    let mut d = 0.0;
    for j in 0..=k {
      let coeff = diff_coeff(k, j);
      d += coeff * path[i + j * stride];
    }
    v += d.abs().powf(p);
  }
  v
}

#[inline]
fn diff_coeff(k: usize, j: usize) -> f64 {
  let sign = if ((k - j) & 1) == 0 { 1.0 } else { -1.0 };
  sign * binomial(k, j)
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
    let comb = binomial(2 * k, (k as isize - i) as usize);
    let abs_term = (i.unsigned_abs() as f64).powf(2.0 * h);
    acc += sign * comb * abs_term;
  }
  0.5 * acc
}

#[inline]
fn binomial(n: usize, k: usize) -> f64 {
  if k > n {
    return 0.0;
  }
  let k = k.min(n - k);
  if k == 0 {
    return 1.0;
  }
  let mut c = 1.0;
  for i in 1..=k {
    c *= (n - k + i) as f64 / i as f64;
  }
  c
}

#[cfg(test)]
mod tests {
  use stochastic_rs_stochastic::diffusion::fou::Fou;

  use super::*;
  use crate::traits::ProcessExt;

  /// Few-ULP tolerance for the v1/v2/v4 reference-regression tests. The
  /// production estimator and the inline reference math are textually
  /// identical but compile at two separate sites — under release +
  /// `-C instrument-coverage`, codegen decisions (FMA contraction,
  /// inlining) can diverge enough to produce a 1-ULP gap that propagates
  /// through `(num/den).sqrt()`. Real algorithmic drift would dwarf this
  /// floor by many orders of magnitude.
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
  fn fou_v4_returns_finite_params_with_fixed_hurst() {
    let n = 1024usize;
    let h_true = 0.7;
    let sigma_true = 0.35;
    let path = Fou::<f64>::new(h_true, 1.5, 0.0, sigma_true, n, Some(0.0), Some(1.0)).sample();

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
    assert!(res.mu.is_finite());
    assert!(res.theta.is_finite() && res.theta > 0.0);
  }

  #[test]
  fn fou_v4_estimated_hurst_in_range_when_not_fixed() {
    let n = 768usize;
    let path = Fou::<f64>::new(0.65, 1.2, 0.0, 0.25, n, Some(0.0), Some(1.0)).sample();

    let res = estimate_fou_v4(path.view(), Some(1.0 / (n - 1) as f64), 2, 2.0, None, None);

    assert!((0.0..1.0).contains(&res.hurst));
    assert!(res.sigma > 0.0 && res.theta > 0.0);
  }

  #[test]
  fn fou_v1_runs_on_fbm_path() {
    let n = 1024usize;
    let path = Fou::<f64>::new(0.7, 1.5, 0.0, 0.35, n, Some(0.0), Some(1.0)).sample();
    let res = estimate_fou_v1(path.view(), FilterType::Daubechies, None, None);
    assert!(res.hurst.is_finite());
    assert!(res.sigma.is_finite() && res.sigma > 0.0);
    assert!(res.mu.is_finite());
    assert!(res.theta.is_finite());
  }

  #[test]
  fn fou_v2_runs_on_fbm_path() {
    let n = 1024usize;
    let path = Fou::<f64>::new(0.7, 1.5, 0.0, 0.35, n, Some(0.0), Some(1.0)).sample();
    let res = estimate_fou_v2(path.view(), None, n, None);
    assert!(res.hurst.is_finite());
    assert!(res.sigma.is_finite() && res.sigma > 0.0);
    assert!(res.mu.is_finite());
    assert!(res.theta.is_finite());
  }

  #[test]
  fn fou_v1_with_hurst_override_skips_estimation() {
    let n = 512usize;
    let path = Fou::<f64>::new(0.6, 1.0, 0.0, 0.2, n, Some(0.0), Some(1.0)).sample();
    let h_pinned = 0.42;
    let res = estimate_fou_v1(path.view(), FilterType::Daubechies, None, Some(h_pinned));
    assert_eq!(res.hurst, h_pinned);
  }

  #[test]
  fn fou_v3_round_trip_recovers_in_range_params() {
    let mut v3 = FOUParameterEstimationV3::new(512, 0.65, 0.3, 1.5, 0.1, 0.0, 1.0, 1.0 / 511.0);
    let res = v3.estimate_parameters();
    assert!(res.hurst.is_finite() && (0.0..1.0).contains(&res.hurst));
    assert!(res.sigma.is_finite() && res.sigma > 0.0);
    assert!(res.mu.is_finite());
    assert!(res.theta.is_finite());
  }

  #[test]
  #[should_panic(expected = "must be >= 5")]
  fn fou_v2_rejects_short_path() {
    let path = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let _ = estimate_fou_v2(path.view(), None, path.len(), None);
  }

  /// Bit-exact regression — reimplements the v1 (struct-era) V1 math
  /// inline against a deterministic fOU path and asserts that
  /// [`estimate_fou_v1`] produces *identical* f64 output. If the refactor
  /// introduces any silent numerical drift (formula, ordering, rounding)
  /// this fires.
  #[test]
  fn fou_v1_bit_exact_against_struct_era_inline_reference() {
    use ndarray::array;
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    // Deterministic reference path (same params as the audit test fixture).
    let n = 512usize;
    let path =
      Fou::<f64, _>::seeded(0.6, 1.0, 0.0, 0.2, n, Some(0.0), Some(1.0), 0xF0_E5_71_AA).sample();
    let delta = 1.0 / n as f64;

    // Daubechies coeffs
    let a = array![
      0.482962913144534 / SQRT_2,
      -0.836516303737808 / SQRT_2,
      0.224143868042013 / SQRT_2,
      0.12940952255126 / SQRT_2,
    ];
    let l_filter = a.len();
    // a_2 = zeros-interlaced a
    let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
    for (i, &val) in a.iter().enumerate() {
      a_2[i * 2 + 1] = val;
    }
    // lfilter on path with b=a, a=[1.0]
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

  /// Bit-exact regression for v2 — same idea as
  /// `fou_v1_bit_exact_against_struct_era_inline_reference` but for the
  /// moments-based estimator.
  #[test]
  fn fou_v2_bit_exact_against_struct_era_inline_reference() {
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    let n = 512usize;
    let path =
      Fou::<f64, _>::seeded(0.6, 1.0, 0.0, 0.2, n, Some(0.0), Some(1.0), 0xF0_E5_71_AA).sample();
    let delta_ref = 1.0 / n as f64;

    // Inline v1.x (struct-era) math
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
    let sigma_den_ref =
      n_f * (4.0 - 2.0_f64.powf(2.0 * hurst_ref)) * delta_ref.powf(2.0 * hurst_ref);
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

  /// Bit-exact regression for v4 — high-frequency power-variation estimator.
  #[test]
  fn fou_v4_bit_exact_against_struct_era_inline_reference() {
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    let n = 512usize;
    let k = 2usize;
    let p = 2.0_f64;
    let path =
      Fou::<f64, _>::seeded(0.65, 1.2, 0.0, 0.25, n, Some(0.0), Some(1.0), 0xC0_FF_EE_42).sample();
    let delta_ref = 1.0 / (n - 1) as f64;

    // Inline reference math: power_variation, c_k_p, rho_k_h_zero, binomial.
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

  #[test]
  fn fou_estimate_result_round_trips_tuple() {
    let r: FouEstimateResult = (0.7, 0.3, 0.05, 1.2).into();
    assert_eq!(r.hurst, 0.7);
    assert_eq!(r.sigma, 0.3);
    assert_eq!(r.mu, 0.05);
    assert_eq!(r.theta, 1.2);
    let t: (f64, f64, f64, f64) = r.into();
    assert_eq!(t, (0.7, 0.3, 0.05, 1.2));
  }
}

//! # fOU Estimator
//!
//! $$
//! (\hat\kappa,\hat\theta,\hat\sigma)=\arg\min \sum_i\left(X_{t_{i+1}}-X_{t_i}-\kappa(\theta-X_{t_i})\Delta t\right)^2
//! $$
//!
use std::f64::consts::SQRT_2;

use ndarray::Array1;
use ndarray::array;
use ndarray::s;
use statrs::function::gamma::gamma;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::ProcessExt;

// Version 1: FOUParameterEstimationV1 with linear filter methods
pub struct FOUParameterEstimationV1 {
  /// Observed path/sample trajectory.
  pub path: Array1<f64>,
  /// Filter/kernel selection used by estimator.
  pub filter_type: FilterType,
  /// Delta sensitivity / sampling interval depending on context.
  pub delta: Option<f64>,
  // Estimated parameters
  hurst: Option<f64>,
  sigma: Option<f64>,
  mu: Option<f64>,
  theta: Option<f64>,
  // Filter coefficients
  a: Array1<f64>,
  L: usize,
  V1: f64,
  V2: f64,
}

impl FOUParameterEstimationV1 {
  pub fn new(path: Array1<f64>, filter_type: FilterType, delta: Option<f64>) -> Self {
    Self {
      path,
      filter_type,
      delta,
      hurst: None,
      sigma: None,
      mu: None,
      theta: None,
      a: Array1::zeros(0),
      L: 0,
      V1: 0.0,
      V2: 0.0,
    }
  }
}

#[derive(PartialEq)]
pub enum FilterType {
  Daubechies,
  Classical,
}

impl FOUParameterEstimationV1 {
  pub fn estimate_parameters(&mut self) -> (f64, f64, f64, f64) {
    self.linear_filter();

    if self.hurst.is_none() {
      self.hurst_estimator();
    }

    self.sigma_estimator();
    self.mu_estimator();
    self.theta_estimator();

    (
      self.hurst.unwrap(),
      self.sigma.unwrap(),
      self.mu.unwrap(),
      self.theta.unwrap(),
    )
  }

  pub fn set_hurst(&mut self, hurst: f64) {
    self.hurst = Some(hurst);
  }

  fn hurst_estimator(&mut self) {
    let hurst = 0.5 * ((self.V2 / self.V1).log2());
    self.hurst = Some(hurst);
  }

  fn sigma_estimator(&mut self) {
    let hurst = self.hurst.unwrap();
    let V1 = self.V1;
    let a = &self.a;
    let L = self.L;

    let series_length = self.path.len();
    let delta: f64 = self.delta.unwrap_or(1.0 / series_length as f64);

    let mut const_filter = 0.0;

    for i in 0..L {
      for j in 0..L {
        const_filter += a[i] * a[j] * ((i as f64 - j as f64).abs()).powf(2.0 * hurst);
      }
    }

    let numerator = -2.0 * V1 / ((series_length - L) as f64);
    let denominator = const_filter * delta.powf(2.0 * hurst);

    let sigma_squared = numerator / denominator;
    let sigma = sigma_squared.sqrt();
    self.sigma = Some(sigma);
  }

  fn mu_estimator(&mut self) {
    let mean = self.path.mean().unwrap();
    self.mu = Some(mean);
  }

  fn theta_estimator(&mut self) {
    let mean_square = self.path.mapv(|x| x.powi(2)).mean().unwrap();
    let sigma = self.sigma.unwrap();
    let hurst = self.hurst.unwrap();

    let numerator = 2.0 * mean_square;
    let denominator = sigma.powi(2) * gamma(2.0 * hurst + 1.0);
    let theta = (numerator / denominator).powf(-1.0 / (2.0 * hurst));

    self.theta = Some(theta);
  }

  fn linear_filter(&mut self) {
    let (a, L) = self.get_filter_coefficients();
    self.a = a.clone();
    self.L = L;

    let a_2 = self.get_a2_coefficients(&a);

    let V1_path = self.lfilter(&self.a, &array![1.0], &self.path);
    self.V1 = V1_path.mapv(|x| x.powi(2)).sum();

    let V2_path = self.lfilter(&a_2, &array![1.0], &self.path);
    self.V2 = V2_path.mapv(|x| x.powi(2)).sum();
  }

  fn get_filter_coefficients(&self) -> (Array1<f64>, usize) {
    let a: Array1<f64>;
    let L: usize;
    if self.filter_type == FilterType::Daubechies {
      a = array![
        0.482962913144534 / SQRT_2,
        -0.836516303737808 / SQRT_2,
        0.224143868042013 / SQRT_2,
        0.12940952255126 / SQRT_2
      ];
      L = a.len();
    } else if self.filter_type == FilterType::Classical {
      unimplemented!("Classical filter not implemented yet.");
    } else {
      a = array![
        0.482962913144534 / SQRT_2,
        -0.836516303737808 / SQRT_2,
        0.224143868042013 / SQRT_2,
        0.12940952255126 / SQRT_2
      ];
      L = a.len();
    }
    (a, L)
  }

  fn get_a2_coefficients(&self, a: &Array1<f64>) -> Array1<f64> {
    // Inserting zeros between the coefficients
    let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
    for (i, &val) in a.iter().enumerate() {
      a_2[i * 2 + 1] = val;
    }
    a_2
  }

  fn lfilter(&self, b: &Array1<f64>, a: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
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
}

// Version 2: FOUParameterEstimationV2 without linear filters
pub struct FOUParameterEstimationV2 {
  /// Observed path/sample trajectory.
  pub path: Array1<f64>,
  /// Delta sensitivity / sampling interval depending on context.
  pub delta: Option<f64>,
  /// Number of samples in the input series.
  pub series_length: usize,
  // Estimated parameters
  hurst: Option<f64>,
  sigma: Option<f64>,
  mu: Option<f64>,
  theta: Option<f64>,
}

impl FOUParameterEstimationV2 {
  pub fn new(path: Array1<f64>, delta: Option<f64>, series_length: usize) -> Self {
    Self {
      path,
      delta,
      series_length,
      hurst: None,
      sigma: None,
      mu: None,
      theta: None,
    }
  }

  pub fn estimate_parameters(&mut self) -> (f64, f64, f64, f64) {
    if self.hurst.is_none() {
      self.hurst_estimator();
    }
    self.sigma_estimator();
    self.mu_estimator();
    self.theta_estimator();

    (
      self.hurst.unwrap(),
      self.sigma.unwrap(),
      self.mu.unwrap(),
      self.theta.unwrap(),
    )
  }

  pub fn set_hurst(&mut self, hurst: f64) {
    self.hurst = Some(hurst);
  }

  fn hurst_estimator(&mut self) {
    let X = &self.path;
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
    self.hurst = Some(estimated_hurst);
  }

  fn sigma_estimator(&mut self) {
    let H = self.hurst.unwrap();
    let X = &self.path;
    let N = self.series_length as f64;
    let delta = self.delta.unwrap_or(1.0 / N);

    let numerator: f64 = (0..(self.series_length - 2))
      .map(|i| {
        let diff = X[i + 2] - 2.0 * X[i + 1] + X[i];
        diff * diff
      })
      .sum();

    let denominator = N * (4.0 - 2.0_f64.powf(2.0 * H)) * delta.powf(2.0 * H);
    let estimated_sigma = (numerator / denominator).sqrt();
    self.sigma = Some(estimated_sigma);
  }

  fn mu_estimator(&mut self) {
    let mean = self.path.mean().unwrap();
    self.mu = Some(mean);
  }

  fn theta_estimator(&mut self) {
    let X = &self.path;
    let H = self.hurst.unwrap();
    let N = self.series_length as f64;
    let sigma = self.sigma.unwrap();

    let sum_X_squared = X.mapv(|x| x * x).sum();
    let sum_X = X.sum();
    let numerator = N * sum_X_squared - sum_X.powi(2);
    let denominator = N.powi(2) * sigma.powi(2) * H * gamma(2.0 * H);

    let estimated_theta = (numerator / denominator).powf(-1.0 / (2.0 * H));
    self.theta = Some(estimated_theta);
  }
}

// Version 3: FOUParameterEstimationV3 with get_path method
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

  pub fn estimate_parameters(&mut self) -> (f64, f64, f64, f64) {
    self.get_path();
    self.hurst_estimator();
    self.sigma_estimator();
    self.mu_estimator();
    self.alpha_estimator();

    (
      self.estimated_hurst.unwrap(),
      self.estimated_sigma.unwrap(),
      self.estimated_mu.unwrap(),
      self.estimated_alpha.unwrap(),
    )
  }

  fn get_path(&mut self) {
    let M = 8;
    let gamma = self.delta / M as f64;

    let fgn_length = self.series_length * M;

    // Generate fGN sample of length fgn_length
    let fgn = FGN::new(self.hurst, fgn_length - 1, Some(self.T));
    let fgn_sample = fgn.sample();

    // Initialize full_fou array
    let mut full_fou = Array1::zeros(fgn_length);
    full_fou[0] = self.initial_value;

    for i in 1..fgn_length {
      full_fou[i] = full_fou[i - 1]
        + self.alpha * (self.mu - full_fou[i - 1]) * gamma
        + self.sigma * fgn_sample[i - 1];
    }

    // Initialize fou array
    let mut fou = Array1::zeros(self.series_length);
    fou[0] = self.initial_value;

    for i in 1..self.series_length {
      let start = (i - 1) * M;
      let end = i * M;

      let sum_sub_series: f64 = full_fou.slice(s![start..end]).sum() * gamma / M as f64;
      fou[i] = full_fou[end - 1] + self.alpha * sum_sub_series;
    }

    // Store the path
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
    let estimated_sigma = (numerator / denominator).sqrt();
    self.estimated_sigma = Some(estimated_sigma);
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

/// Version 4: fOU estimator aligned with high-frequency power-variation formulas.
///
/// Source:
/// - https://arxiv.org/abs/1703.09372
pub struct FOUParameterEstimationV4 {
  /// Observed path on an equidistant grid.
  pub path: Array1<f64>,
  /// Sampling interval. If omitted, defaults to `1/(N-1)`.
  pub delta: Option<f64>,
  /// Difference order `k` for power variation.
  pub k: usize,
  /// Power `p` for `V_{k,p}`.
  pub p: f64,
  // Estimated parameters
  hurst: Option<f64>,
  sigma: Option<f64>,
  mu: Option<f64>,
  theta: Option<f64>,
}

impl FOUParameterEstimationV4 {
  #[must_use]
  pub fn new(path: Array1<f64>, delta: Option<f64>, k: usize, p: f64) -> Self {
    assert!(path.len() >= k + 2, "path length must be at least k + 2");
    assert!(k >= 1, "k must be at least 1");
    assert!(p.is_finite() && p > 0.0, "p must be positive");
    if let Some(dt) = delta {
      assert!(dt.is_finite() && dt > 0.0, "delta must be positive");
    }

    Self {
      path,
      delta,
      k,
      p,
      hurst: None,
      sigma: None,
      mu: None,
      theta: None,
    }
  }

  pub fn estimate_parameters(&mut self) -> (f64, f64, f64, f64) {
    if self.hurst.is_none() {
      self.hurst_estimator();
    }
    if self.sigma.is_none() {
      self.sigma_estimator();
    }
    self.mu_estimator();
    self.theta_estimator();

    (
      self.hurst.unwrap(),
      self.sigma.unwrap(),
      self.mu.unwrap(),
      self.theta.unwrap(),
    )
  }

  pub fn set_hurst(&mut self, hurst: f64) {
    assert!(
      hurst.is_finite() && (0.0..1.0).contains(&hurst),
      "hurst must be in (0, 1)"
    );
    self.hurst = Some(hurst);
  }

  pub fn set_sigma(&mut self, sigma: f64) {
    assert!(sigma.is_finite() && sigma > 0.0, "sigma must be positive");
    self.sigma = Some(sigma);
  }

  fn effective_delta(&self) -> f64 {
    self.delta.unwrap_or(1.0 / (self.path.len() - 1) as f64)
  }

  fn hurst_estimator(&mut self) {
    // Scale-ratio estimate based on k-th order p-variations at stride 1 and 2.
    // V(stride=2)/V(stride=1) ≈ 2^(pH-1), hence H ≈ (1 + log2 ratio)/p.
    let v1 = self.power_variation(self.k, self.p, 1);
    let v2 = self.power_variation(self.k, self.p, 2);
    assert!(v1 > 0.0 && v2 > 0.0, "power variations must be positive");

    let mut h = (1.0 + (v2 / v1).log2()) / self.p;
    if !h.is_finite() {
      h = 0.5;
    }
    h = h.clamp(1e-6, 1.0 - 1e-6);
    self.hurst = Some(h);
  }

  fn sigma_estimator(&mut self) {
    // |sigma_hat|^p = n^{-1+pH} V_{k,p}^n(X)_T / (c_{k,p} T)
    let h = self.hurst.unwrap();
    let n = (self.path.len() - 1) as f64;
    let t_horizon = n * self.effective_delta();
    let v = self.power_variation(self.k, self.p, 1);
    let c_k_p = self.c_k_p(self.k, self.p, h);

    let scaled = n.powf(-1.0 + self.p * h) * v;
    let sigma_abs_p = scaled / (c_k_p * t_horizon);
    assert!(
      sigma_abs_p.is_finite() && sigma_abs_p > 0.0,
      "estimated sigma^p must be positive"
    );

    let sigma = sigma_abs_p.powf(1.0 / self.p);
    self.sigma = Some(sigma);
  }

  fn mu_estimator(&mut self) {
    self.mu = Some(self.path.mean().unwrap());
  }

  fn theta_estimator(&mut self) {
    // Discrete ergodic estimator:
    // theta_bar_n = ( (1/(n sigma^2 H Gamma(2H))) sum_{k=1}^n X_{kh}^2 )^{-1/(2H)}.
    let h = self.hurst.unwrap();
    let sigma = self.sigma.unwrap();
    let mu = self.mu.unwrap_or_else(|| self.path.mean().unwrap());
    let n = (self.path.len() - 1) as f64;

    let sum_sq: f64 = self
      .path
      .slice(s![1..])
      .iter()
      .map(|x| (x - mu).powi(2))
      .sum();
    let denom = n * sigma.powi(2) * h * gamma(2.0 * h);
    let base = sum_sq / denom;
    assert!(base.is_finite() && base > 0.0, "invalid theta base value");

    let theta = base.powf(-1.0 / (2.0 * h));
    assert!(
      theta.is_finite() && theta > 0.0,
      "estimated theta must be positive"
    );
    self.theta = Some(theta);
  }

  fn power_variation(&self, k: usize, p: f64, stride: usize) -> f64 {
    let n = self.path.len();
    let span = k * stride;
    assert!(n > span, "path is too short for requested (k, stride)");

    let mut v = 0.0;
    for i in 0..(n - span) {
      let mut d = 0.0;
      for j in 0..=k {
        let coeff = self.diff_coeff(k, j);
        d += coeff * self.path[i + j * stride];
      }
      v += d.abs().powf(p);
    }
    v
  }

  fn diff_coeff(&self, k: usize, j: usize) -> f64 {
    let sign = if ((k - j) & 1) == 0 { 1.0 } else { -1.0 };
    sign * Self::binomial(k, j)
  }

  fn c_k_p(&self, k: usize, p: f64, h: f64) -> f64 {
    let rho0 = self.rho_k_h_zero(k, h);
    assert!(
      rho0.is_finite() && rho0 > 0.0,
      "rho_(k,H)(0) must be positive"
    );

    let normal_abs_p = 2.0_f64.powf(p / 2.0) * gamma((p + 1.0) / 2.0) / gamma(0.5);
    normal_abs_p * rho0.powf(p / 2.0)
  }

  fn rho_k_h_zero(&self, k: usize, h: f64) -> f64 {
    let mut acc = 0.0;
    for i in -(k as isize)..=(k as isize) {
      let parity = (1_isize - i).rem_euclid(2);
      let sign = if parity == 0 { 1.0 } else { -1.0 };
      let comb = Self::binomial(2 * k, (k as isize - i) as usize);
      let abs_term = (i.unsigned_abs() as f64).powf(2.0 * h);
      acc += sign * comb * abs_term;
    }
    0.5 * acc
  }

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
}

#[cfg(test)]
mod fou_v4_tests {
  use super::FOUParameterEstimationV4;
  use crate::stochastic::diffusion::fou::FOU;
  use crate::traits::ProcessExt;

  #[test]
  fn fou_v4_returns_finite_params() {
    let n = 1024usize;
    let h_true = 0.7;
    let sigma_true = 0.35;
    let path = FOU::<f64>::new(h_true, 1.5, 0.0, sigma_true, n, Some(0.0), Some(1.0)).sample();

    let mut est = FOUParameterEstimationV4::new(path, Some(1.0 / (n - 1) as f64), 2, 2.0);
    est.set_hurst(h_true);
    let (h, sigma, mu, theta) = est.estimate_parameters();
    println!(
      "FOUV4 fixed-H test => estimated: H={:.6}, sigma={:.6}, mu={:.6}, theta={:.6} | true: H={:.6}, sigma={:.6}",
      h, sigma, mu, theta, h_true, sigma_true
    );

    assert!(h.is_finite() && (0.0..1.0).contains(&h));
    assert!(sigma.is_finite() && sigma > 0.0);
    assert!(mu.is_finite());
    assert!(theta.is_finite() && theta > 0.0);
  }

  #[test]
  fn fou_v4_estimated_hurst_in_range_when_not_fixed() {
    let n = 768usize;
    let path = FOU::<f64>::new(0.65, 1.2, 0.0, 0.25, n, Some(0.0), Some(1.0)).sample();
    let mut est = FOUParameterEstimationV4::new(path, Some(1.0 / (n - 1) as f64), 2, 2.0);

    let (h, sigma, _mu, theta) = est.estimate_parameters();
    println!(
      "FOUV4 free-H test  => estimated: H={:.6}, sigma={:.6}, theta={:.6}",
      h, sigma, theta
    );
    assert!((0.0..1.0).contains(&h));
    assert!(sigma > 0.0 && theta > 0.0);
  }
}

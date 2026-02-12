use std::f64::consts::SQRT_2;

use impl_new_derive::ImplNew;
use ndarray::array;
use ndarray::s;
use ndarray::Array1;
use statrs::function::gamma::gamma;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Process;

// Version 1: FOUParameterEstimationV1 with linear filter methods
#[derive(ImplNew)]
pub struct FOUParameterEstimationV1 {
  pub path: Array1<f64>,
  pub filter_type: FilterType,
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
#[derive(ImplNew)]
pub struct FOUParameterEstimationV2 {
  pub path: Array1<f64>,
  pub delta: Option<f64>,
  pub series_length: usize,
  // Estimated parameters
  hurst: Option<f64>,
  sigma: Option<f64>,
  mu: Option<f64>,
  theta: Option<f64>,
}

impl FOUParameterEstimationV2 {
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

//! # GBM
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dW_t,\quad S_0=s_0
//! $$
//!
use ndarray::Array1;
use ndarray::s;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::LogNormal;
use statrs::statistics::Distribution as StatDistribution;
use statrs::statistics::Median;
use statrs::statistics::Mode;

use crate::traits::DistributionExt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct GBM<T: FloatExt> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  distribution: Option<LogNormal>,
}

impl<T: FloatExt> GBM<T> {
  pub fn new(mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    let x0_f64 = x0.unwrap_or(T::one()).to_f64().unwrap();
    let mu_f64 = mu.to_f64().unwrap();
    let sigma_f64 = sigma.to_f64().unwrap();
    let t_f64 = t.unwrap_or(T::one()).to_f64().unwrap();

    let mu_ln = x0_f64.ln() + (mu_f64 - 0.5 * sigma_f64 * sigma_f64) * t_f64;
    let sigma_ln = sigma_f64 * t_f64.sqrt();

    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      distribution: LogNormal::new(mu_ln, sigma_ln).ok(),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for GBM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut gbm = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return gbm;
    }

    gbm[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return gbm;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let drift_scale = self.mu * dt;
    let diff_scale = self.sigma * dt.sqrt();
    let mut prev = gbm[0];
    let mut tail_view = gbm.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("GBM output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    for z in tail.iter_mut() {
      let next = prev + drift_scale * prev + diff_scale * prev * *z;
      *z = next;
      prev = next;
    }

    gbm
  }
}

impl<T: FloatExt> GBM<T> {
  /// Malliavin derivative of the GBM process
  ///
  /// The Malliavin derivative of the GBM process is given by
  /// `D_r S_t = \sigma S_t \mathbf{1}_{0 \le r \le t}`.
  ///
  /// The Malliavin derivate of the GBM shows the sensitivity of the stock price with respect to the Wiener process.
  pub fn malliavin(&self) -> [Array1<T>; 2] {
    let gbm = self.sample();
    let mut m = Array1::zeros(self.n);

    // reverse due the option pricing
    let s_t = *gbm.last().unwrap();
    for i in 0..self.n {
      m[i] = self.sigma * s_t;
    }

    [gbm, m]
  }
}

impl<T: FloatExt> DistributionExt for GBM<T> {
  fn pdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().map_or(0.0, |d| d.pdf(x))
  }

  fn cdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().map_or(0.0, |d| d.cdf(x))
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    self.distribution.as_ref().map_or(0.0, |d| d.inverse_cdf(p))
  }

  fn mean(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .and_then(|d| d.mean())
      .unwrap_or(0.0)
  }

  fn mode(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .and_then(|d| d.mode())
      .unwrap_or(0.0)
  }

  fn median(&self) -> f64 {
    self.distribution.as_ref().map_or(0.0, |d| d.median())
  }

  fn variance(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .and_then(|d| d.variance())
      .unwrap_or(0.0)
  }

  fn skewness(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .and_then(|d| d.skewness())
      .unwrap_or(0.0)
  }

  fn entropy(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .and_then(|d| d.entropy())
      .unwrap_or(0.0)
  }
}

py_process_1d!(PyGBM, GBM,
  sig: (mu, sigma, n, x0=None, t=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

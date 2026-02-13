use ndarray::Array1;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::LogNormal;
use statrs::statistics::Distribution as StatDistribution;
use statrs::statistics::Median;
use statrs::statistics::Mode;

use crate::stochastic::noise::gn::Gn;
use crate::traits::DistributionExt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct GBM<T: FloatExt> {
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
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
      gn: Gn::new(n - 1, t),
      distribution: LogNormal::new(mu_ln, sigma_ln).ok(),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for GBM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut gbm = Array1::<T>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
  }
}

impl<T: FloatExt> GBM<T> {
  /// Malliavin derivative of the GBM process
  ///
  /// The Malliavin derivative of the GBM process is given by
  /// D_r S_t = \sigma S_t * 1_[0, r](r)
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

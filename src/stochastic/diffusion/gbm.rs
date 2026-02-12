use ndarray::Array1;
use num_complex::Complex64;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::LogNormal;
use statrs::statistics::Distribution as StatDistribution;
use statrs::statistics::Median;
use statrs::statistics::Mode;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::DistributionExt;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

pub struct GBM<T: Float> {
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
  distribution: Option<LogNormal>,
}

impl<T: Float> GBM<T> {
  pub fn new(mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
      distribution: None,
    }
  }
}

impl<T: Float> ProcessExt<T> for GBM<T> {
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

impl<T: Float> GBM<T> {
  /// Distribution of the GBM process
  // fn distribution(&mut self) {
  //   let mu = self.x0.unwrap() * (self.mu * self.t.unwrap()).exp();
  //   let sigma = (self.x0.unwrap().powi(2)
  //     * (2.0 * self.mu * self.t.unwrap()).exp()
  //     * ((self.sigma.powi(2) * self.t.unwrap()).exp() - T::one()))
  //   .sqrt();

  //   self.distribution = Some(LogNormal::new(mu, sigma).unwrap());
  // }

  /// Mallaivin derivative of the GBM process
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

// TODO: needs rework
impl<T: Float> DistributionExt for GBM<T> {
  /// Characteristic function of the distribution
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    unimplemented!()
  }

  /// Probability density function of the distribution
  fn pdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().pdf(x)
  }

  /// Cumulative distribution function of the distribution
  fn cdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().cdf(x)
  }

  /// Inverse cumulative distribution function of the distribution
  fn inv_cdf(&self, p: f64) -> f64 {
    self.distribution.as_ref().unwrap().inverse_cdf(p)
  }

  /// Mean of the distribution
  fn mean(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mean()
      .expect("Mean not found")
  }

  /// Mode of the distribution
  fn mode(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mode()
      .expect("Mode not found")
  }

  /// Median of the distribution
  fn median(&self) -> f64 {
    self.distribution.as_ref().unwrap().median()
  }

  /// Variance of the distribution
  fn variance(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .variance()
      .expect("Variance not found")
  }

  /// Skewness of the distribution
  fn skewness(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .skewness()
      .expect("Skewness not found")
  }

  /// Kurtosis of the distribution
  fn kurtosis(&self) -> f64 {
    unimplemented!()
  }

  /// Entropy of the distribution
  fn entropy(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .entropy()
      .expect("Entropy not found")
  }

  /// Moment generating function of the distribution
  fn moment_generating_function(&self, _t: f64) -> f64 {
    unimplemented!()
  }
}

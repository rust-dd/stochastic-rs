use impl_new_derive::ImplNew;
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
use crate::stochastic::Process;

#[derive(ImplNew)]
pub struct GBM<T> {
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  distribution: Option<LogNormal>,
}

impl<T: Float> Process<T> for GBM<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(&self, noise_fn: impl FnOnce(&Self::Noise) -> Self::Output) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = noise_fn(&gn);

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
  fn distribution(&mut self) {
    let mu = self.x0.unwrap() * (self.mu * self.t.unwrap()).exp();
    let sigma = (self.x0.unwrap().powi(2)
      * (2.0 * self.mu * self.t.unwrap()).exp()
      * ((self.sigma.powi(2) * self.t.unwrap()).exp() - T::one()))
    .sqrt();

    self.distribution = Some(LogNormal::new(mu.into(), sigma.into()).unwrap());
  }

  /// Mallaivin derivative of the GBM process
  ///
  /// The Malliavin derivative of the GBM process is given by
  /// D_r S_t = \sigma S_t * 1_[0, r](r)
  ///
  /// The Malliavin derivate of the GBM shows the sensitivity of the stock price with respect to the Wiener process.
  fn malliavin(&self) -> [Array1<T>; 2] {
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::plot_2d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn gbm_length_equals_n() {
    let gbm = GBM::new(0.25, 0.5, N, Some(X0), Some(1.0));
    assert_eq!(gbm.sample().len(), N);
  }

  #[test]
  fn gbm_starts_with_x0() {
    let gbm = GBM::new(0.25, 0.5, N, Some(X0), Some(1.0));
    assert_eq!(gbm.sample()[0], X0);
  }

  #[test]
  fn gbm_plot() {
    let gbm = GBM::new(0.25, 0.5, N * 10, Some(X0), Some(1.0));

    plot_1d!(gbm.sample(), "Geometric Brownian Motion (GBM) process");
  }

  #[test]
  fn gbm_benchmark() {
    let gbm = GBM::new(0.25, 0.5, N * 10, Some(X0), Some(1.0));

    let iters = N * 10;

    let start = std::time::Instant::now();
    for _ in 0..iters {
      gbm.sample();
    }
    let basic_ms_per = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    #[cfg(feature = "simd")]
    {
      let start = std::time::Instant::now();
      for _ in 0..iters {
        gbm.sample_simd();
      }
      let simd_ms_per = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

      println!("Basic: {:.6} ms, SIMD: {:.6} ms", basic_ms_per, simd_ms_per);
    }
  }

  #[test]
  fn gbm_malliavin() {
    let gbm = GBM::new(0.25, 0.5, N, Some(X0), Some(1.0));
    let process = gbm.sample();
    let malliavin = gbm.malliavin();
    plot_2d!(
      process,
      "Geometric Brownian Motion (GBM) process",
      malliavin.unwrap(),
      "Malliavin derivative of the GBM process"
    );
  }
}

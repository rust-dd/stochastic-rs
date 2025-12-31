#[cfg(feature = "malliavin")]
use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use num_complex::Complex64;
use rand_distr::Normal;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::LogNormal;
use statrs::statistics::Distribution as StatDistribution;
use statrs::statistics::Median;
use statrs::statistics::Mode;

use crate::stochastic::DistributionExt;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct GBM<T> {
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub distribution: Option<LogNormal>,
  #[cfg(feature = "malliavin")]
  pub calculate_malliavin: Option<bool>,
  #[cfg(feature = "malliavin")]
  malliavin: Mutex<Option<Array1<T>>>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for GBM<f64> {
  /// Sample the GBM process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut gbm = Array1::<f64>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    #[cfg(feature = "malliavin")]
    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut malliavin = Array1::zeros(self.n);

      // reverse due the option pricing
      for i in 0..self.n {
        malliavin[i] = self.sigma * gbm.last().unwrap();
      }

      // This equivalent to the following:
      // self.malliavin.lock().unwrap().replace(Some(malliavin));
      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }

    gbm
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use ndarray::Array1;
    use wide::f64x8;

    use crate::stats::distr::normal_f64::SimdNormal;

    let n = self.n;
    assert!(n >= 1, "n must be >= 1");

    let t = self.t.unwrap_or(1.0);
    let s0 = self.x0.unwrap_or(1.0);

    if n == 1 {
      return Array1::from(vec![s0]);
    }

    let dt = t / (n - 1) as f64;

    let drift_dt = (self.mu - 0.5 * self.sigma * self.sigma) * dt;
    let vol_sdt = self.sigma * dt.sqrt();

    let driftv = f64x8::splat(drift_dt);
    let volv = f64x8::splat(self.sigma);

    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));
    let mut gbm = Array1::zeros(n);
    let mut s = s0;
    gbm[0] = s;

    let mut i = 1usize;

    while i + 8 <= n - 1 {
      let z = f64x8::from([
        gn[i - 1],
        gn[i],
        gn[i + 1],
        gn[i + 2],
        gn[i + 3],
        gn[i + 4],
        gn[i + 5],
        gn[i + 6],
      ]);

      let g = (driftv + volv * z).exp().to_array();

      for gj in g {
        s *= gj;
        gbm[i] = s;
      }

      i += 8;
    }

    while i < n - 1 {
      let z = gn[i];
      s *= (drift_dt + vol_sdt * z).exp();
      gbm[i] = s;
      i += 1;
    }

    gbm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Distribution of the GBM process
  fn distribution(&mut self) {
    let mu = self.x0.unwrap() * (self.mu * self.t.unwrap()).exp();
    let sigma = (self.x0.unwrap().powi(2)
      * (2.0 * self.mu * self.t.unwrap()).exp()
      * ((self.sigma.powi(2) * self.t.unwrap()).exp() - 1.0))
      .sqrt();

    self.distribution = Some(LogNormal::new(mu, sigma).unwrap());
  }

  /// Mallaivin derivative of the GBM process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t * 1_[0, r](r)
  ///
  /// The Malliavin derivate of the GBM shows the sensitivity of the stock price with respect to the Wiener process.
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().as_ref().unwrap().clone()
  }
}

#[cfg(feature = "f64")]
impl DistributionExt for GBM<f64> {
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for GBM<f32> {
  /// Sample the GBM process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut gbm = Array1::<f32>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut gbm = Array1::<f32>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  #[cfg(feature = "malliavin")]
  use crate::plot_2d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn gbm_length_equals_n() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );
    assert_eq!(gbm.sample().len(), N);
  }

  #[test]
  fn gbm_starts_with_x0() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );
    assert_eq!(gbm.sample()[0], X0);
  }

  #[test]
  fn gbm_plot() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N * 10,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );

    plot_1d!(gbm.sample(), "Geometric Brownian Motion (GBM) process");
  }

  #[test]
  fn gbm_benchmark() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N * 10,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );

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
  #[cfg(feature = "malliavin")]
  fn gbm_malliavin() {
    let gbm = GBM::new(0.25, 0.5, N, Some(X0), Some(1.0), None, None, Some(true));
    let process = gbm.sample();
    let malliavin = gbm.malliavin();
    plot_2d!(
      process,
      "Geometric Brownian Motion (GBM) process",
      malliavin,
      "Malliavin derivative of the GBM process"
    );
  }
}

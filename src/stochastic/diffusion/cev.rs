use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CEV<T: Float> {
  pub mu: T,
  pub sigma: T,
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: Float> CEV<T> {
  pub fn new(mu: T, sigma: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      gamma,
      n,
      x0,
      t,
    }
  }
}

impl<T: Float> Process<T> for CEV<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  /// Sample the CEV process
  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = noise_fn(&gn);

    let mut cev = Array1::<T>::zeros(self.n);
    cev[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      cev[i] = cev[i - 1]
        + self.mu * cev[i - 1] * dt
        + self.sigma * cev[i - 1].powf(self.gamma) * gn[i - 1]
    }

    cev
  }
}

impl<T: Float> CEV<T> {
  /// Calculate the Malliavin derivative of the CEV process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t^{\gamma} * 1_{[0, r]}(r) exp(\int_0^r (\mu - \frac{\gamma^2 \sigma^2 S_u^{2\gamma - 2}}{2}) du + \int_0^r \gamma \sigma S_u^{\gamma - 1} dW_u)
  ///
  /// The Malliavin derivative of the CEV process shows the sensitivity of the stock price with respect to the Wiener process.
  fn malliavin(&self) -> [Array1<T>; 2] {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let cev = self.euler_maruyama(|gn| gn.sample());

    let mut det_term = Array1::zeros(self.n);
    let mut stochastic_term = Array1::zeros(self.n);
    let mut m = Array1::zeros(self.n);

    for i in 0..self.n {
      det_term[i] = (self.mu
        - (self.gamma.powi(2)
          * self.sigma.powi(2)
          * cev[i].powf(2.0 * self.gamma - T::from_usize(2))
          / T::from_usize(2)))
        * dt;
      if i > 0 {
        stochastic_term[i] =
          self.sigma * self.gamma * cev[i].powf(self.gamma - T::one()) * gn[i - 1];
      }
      m[i] = self.sigma * cev[i].powf(self.gamma) * (det_term[i] + stochastic_term[i]).exp();
    }

    [cev, m]
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
  fn cev_length_equals_n() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(cev.sample().len(), N);
  }

  #[test]
  fn cev_starts_with_x0() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(cev.sample()[0], X0);
  }

  #[test]
  fn cev_plot() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0));
    plot_1d!(
      cev.sample(),
      "Constant Elasticity of Variance (CEV) process"
    );
  }

  #[test]
  fn cev_malliavin() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0));
    let malliavin = cev.malliavin();
    plot_2d!(
      malliavin[0],
      "Constant Elasticity of Variance (CEV) process",
      malliavin[1],
      "Malliavin derivative of the CEV process"
    );
  }
}

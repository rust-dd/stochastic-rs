use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CEV<T: FloatExt> {
  pub mu: T,
  pub sigma: T,
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> CEV<T> {
  fn new(mu: T, sigma: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      gamma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CEV<T> {
  type Output = Array1<T>;

  /// Sample the CEV process
  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

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

impl<T: FloatExt> CEV<T> {
  /// Calculate the Malliavin derivative of the CEV process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t^{\gamma} * 1_{[0, r]}(r) exp(\int_0^r (\mu - \frac{\gamma^2 \sigma^2 S_u^{2\gamma - 2}}{2}) du + \int_0^r \gamma \sigma S_u^{\gamma - 1} dW_u)
  ///
  /// The Malliavin derivative of the CEV process shows the sensitivity of the stock price with respect to the Wiener process.
  fn malliavin(&self) -> [Array1<T>; 2] {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = gn.sample();
    let cev = self.sample();

    let mut det_term = Array1::zeros(self.n);
    let mut stochastic_term = Array1::zeros(self.n);
    let mut m = Array1::zeros(self.n);

    for i in 0..self.n {
      det_term[i] = (self.mu
        - (self.gamma.powi(2)
          * self.sigma.powi(2)
          * cev[i].powf(T::from_usize_(2) * self.gamma - T::from_usize_(2))
          / T::from_usize_(2)))
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

py_process_1d!(PyCEV, CEV,
  sig: (mu, sigma, gamma, n, x0=None, t=None, dtype=None),
  params: (mu: f64, sigma: f64, gamma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

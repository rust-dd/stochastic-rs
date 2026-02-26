//! # Cev
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t^{\gamma}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CEV<T: FloatExt> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
}

impl<T: FloatExt> CEV<T> {
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

impl<T: FloatExt> ProcessExt<T> for CEV<T> {
  type Output = Array1<T>;

  /// Sample the CEV process
  fn sample(&self) -> Self::Output {
    let mut cev = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return cev;
    }

    cev[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return cev;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let diff_scale = self.sigma * dt.sqrt();
    let mut prev = cev[0];
    let mut tail_view = cev.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("CEV output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    for z in tail.iter_mut() {
      let next = prev + self.mu * prev * dt + diff_scale * prev.powf(self.gamma) * *z;
      *z = next;
      prev = next;
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
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let mut gn = Array1::<T>::zeros(self.n.saturating_sub(1));
    if let Some(gn_slice) = gn.as_slice_mut() {
      T::fill_standard_normal_slice(gn_slice);
      let sqrt_dt = dt.sqrt();
      for z in gn_slice.iter_mut() {
        *z = *z * sqrt_dt;
      }
    }
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

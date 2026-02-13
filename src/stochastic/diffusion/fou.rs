use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FOU<T: FloatExt> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> FOU<T> {
  #[must_use]
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FOU<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample();

    let mut fou = Array1::<T>::zeros(self.n);
    fou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fou[i] = fou[i - 1] + self.theta * (self.mu - fou[i - 1]) * dt + self.sigma * fgn[i - 1];
    }

    fou
  }
}

py_process_1d!(PyFOU, FOU,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

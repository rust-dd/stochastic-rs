use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

pub struct FGBM<T: FloatExt> {
  pub hurst: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> FGBM<T> {
  #[must_use]
  pub fn new(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FGBM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut fgbm = Array1::<T>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }
}

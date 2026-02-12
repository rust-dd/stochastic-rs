use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct SABR<T: Float> {
  pub alpha: T,
  pub beta: T,
  pub rho: T,
  pub n: usize,
  pub f0: Option<T>,
  pub v0: Option<T>,
  pub t: Option<T>,
  cgns: CGNS<T>,
}

impl<T: Float> SABR<T> {
  pub fn new(
    alpha: T,
    beta: T,
    rho: T,
    n: usize,
    f0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      alpha,
      beta,
      rho,
      n,
      f0,
      v0,
      t,
      cgns: CGNS::new(rho, n, t),
    }
  }
}

impl<T: Float> Process<T> for SABR<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    f_[0] = self.f0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

    for i in 1..self.n {
      f_[i] = f_[i - 1] + v[i - 1] * f_[i - 1].powf(self.beta) * cgn1[i - 1];
      v[i] = v[i - 1] + self.alpha * v[i - 1] * cgn2[i - 1];
    }

    [f_, v]
  }
}

impl<T: Float> SABR<T> {
  /// Calculate the Malliavin derivative of the SABR model
  ///
  /// The Malliavin derivative of the volaility process in the SABR model is given by:
  /// D_r \sigma_t = \alpha \sigma_t 1_{[0, T]}(r)
  fn malliavin_of_vol(&self) -> [Array1<T>; 3] {
    let [f, v] = self.sample();

    let mut malliavin = Array1::<T>::zeros(self.n);

    for i in 0..self.n {
      malliavin[i] = self.alpha * *v.last().unwrap();
    }

    [f, v, malliavin]
  }
}

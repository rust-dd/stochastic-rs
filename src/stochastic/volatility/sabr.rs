use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct SABR<T: FloatExt> {
  pub alpha: T,
  pub beta: T,
  pub rho: T,
  pub n: usize,
  pub f0: Option<T>,
  pub v0: Option<T>,
  pub t: Option<T>,
  cgns: CGNS<T>,
}

impl<T: FloatExt> SABR<T> {
  pub fn new(
    alpha: T,
    beta: T,
    rho: T,
    n: usize,
    f0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(
      beta >= T::zero() && beta <= T::one(),
      "beta must be in [0, 1] for SABR"
    );
    assert!(alpha >= T::zero(), "alpha must be non-negative");
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }

    Self {
      alpha,
      beta,
      rho,
      n,
      f0,
      v0,
      t,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for SABR<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    f_[0] = self.f0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    for i in 1..self.n {
      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = f_[i - 1] + v_prev * f_prev.powf(self.beta) * cgn1[i - 1];
      // Exact step for dV = alpha * V * dW preserves non-negativity.
      v[i] =
        v_prev * (self.alpha * cgn2[i - 1] - T::from_f64_fast(0.5) * self.alpha.powi(2) * dt).exp();
    }

    [f_, v]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn volatility_stays_non_negative() {
    let p = SABR::new(0.4_f64, 0.5, -0.3, 256, Some(1.0), Some(0.2), Some(1.0));
    let [_f, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }
}

impl<T: FloatExt> SABR<T> {
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

py_process_2x1d!(PySABR, SABR,
  sig: (alpha, beta, rho, n, f0=None, v0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, rho: f64, n: usize, f0: Option<f64>, v0: Option<f64>, t: Option<f64>)
);

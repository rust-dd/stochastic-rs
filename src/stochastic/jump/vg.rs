use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct VG<T: FloatExt> {
  pub mu: T,
  pub sigma: T,
  pub nu: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: FloatExt> VG<T> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
    }
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt> ProcessExt<T> for VG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut vg = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return vg;
    }
    vg[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return vg;
    }

    let dt = self.dt();
    let gamma = SimdGamma::new(dt / self.nu, self.nu);
    let gammas = Array1::random(self.n - 1, &gamma);
    let mut z = Array1::<T>::zeros(self.n - 1);
    let z_slice = z.as_slice_mut().expect("VG normals must be contiguous");
    T::fill_standard_normal_slice(z_slice);

    for i in 1..self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * z[i - 1];
    }

    vg
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = VG::new(0.1_f64, 0.2, 0.3, 1, Some(2.5), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 2.5);
  }
}

py_process_1d!(PyVG, VG,
  sig: (mu, sigma, nu, n, x0=None, t=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

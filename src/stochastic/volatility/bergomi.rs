use ndarray::s;
use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

pub struct Bergomi<T: FloatExt> {
  pub nu: T,
  pub v0: Option<T>,
  pub s0: Option<T>,
  pub r: T,
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  cgns: CGNS<T>,
}

impl<T: FloatExt> Bergomi<T> {
  pub fn new(
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
    m: Option<usize>,
  ) -> Self {
    Self {
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      m,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Bergomi<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    s[0] = self.s0.unwrap_or(T::from_usize_(100));
    v2[0] = self.v0.unwrap_or(T::one()).powi(2);

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = cgn2.slice(s![..i]).sum();
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0.unwrap_or(T::one()).powi(2)
        * (self.nu * t * sum_z - T::from_f64_fast(0.5) * self.nu.powi(2) * t.powi(2)).exp()
    }

    [s, v2]
  }
}

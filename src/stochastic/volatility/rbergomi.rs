use ndarray::s;
use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct RoughBergomi<T: Float> {
  pub hurst: T,
  pub nu: T,
  pub v0: Option<T>,
  pub s0: Option<T>,
  pub r: T,
  pub rho: T,
  pub n: usize,
  pub t: Option<T>,
  cgns: CGNS<T>,
}

impl<T: Float> RoughBergomi<T> {
  pub fn new(
    hurst: T,
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
  ) -> Self {
    RoughBergomi {
      hurst,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for RoughBergomi<T> {
  type Output = [Array1<T>; 2];
  type Noise = CGNS<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, z] = noise_fn(&self.cgns);

    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    s[0] = self.s0.unwrap_or(T::from_usize(100));
    v2[0] = self.v0.unwrap_or(T::one()).powi(2);

    for i in 1..=self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = z.slice(s![..i]).sum();
      let t = i as f64 * dt;
      v2[i] = self.v0.unwrap_or(T::one()).powi(2)
        * (self.nu * (2.0 * self.hurst).sqrt() * t.powf(self.hurst - 0.5.into()) * sum_z
          - 0.5 * self.nu.powi(2) * t.powf(2.0 * self.hurst))
        .exp();
    }

    [s, v2]
  }
}

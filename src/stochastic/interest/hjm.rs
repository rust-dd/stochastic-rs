use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct HJM<T: Float> {
  pub a: fn(T) -> T,
  pub b: fn(T) -> T,
  pub p: fn(T, T) -> T,
  pub q: fn(T, T) -> T,
  pub v: fn(T, T) -> T,
  pub alpha: fn(T, T) -> T,
  pub sigma: fn(T, T) -> T,
  pub n: usize,
  pub r0: Option<T>,
  pub p0: Option<T>,
  pub f0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> HJM<T> {
  pub fn new(
    a: fn(T) -> T,
    b: fn(T) -> T,
    p: fn(T, T) -> T,
    q: fn(T, T) -> T,
    v: fn(T, T) -> T,
    alpha: fn(T, T) -> T,
    sigma: fn(T, T) -> T,
    n: usize,
    r0: Option<T>,
    p0: Option<T>,
    f0: Option<T>,
    t: Option<T>,
    gn: Gn<T>,
  ) -> Self {
    Self {
      a,
      b,
      p,
      q,
      v,
      alpha,
      sigma,
      n,
      r0,
      p0,
      f0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for HJM<T> {
  type Output = [Array1<T>; 3];
  type Noise = Gn<T>;

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
    let dt = self.gn.dt();

    let mut r = Array1::<T>::zeros(self.n);
    let mut p = Array1::<T>::zeros(self.n);
    let mut f = Array1::<T>::zeros(self.n);

    let gn1 = noise_fn(&self.gn);
    let gn2 = noise_fn(&self.gn);
    let gn3 = noise_fn(&self.gn);

    let t_max = self.t.unwrap_or(T::one());

    for i in 1..self.n {
      let t = i as f64 * dt;

      r[i] = r[i - 1] + (self.a)(t) * dt + (self.b)(t) * gn1[i - 1];
      p[i] =
        p[i - 1] + (self.p)(t, t_max) * ((self.q)(t, t_max) * dt + (self.v)(t, t_max) * gn2[i - 1]);
      f[i] = f[i - 1] + (self.alpha)(t, t_max) * dt + (self.sigma)(t, t_max) * gn3[i - 1];
    }

    [r, p, f]
  }
}

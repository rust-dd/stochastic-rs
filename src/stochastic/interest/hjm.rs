use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling3DExt;

#[derive(ImplNew)]
pub struct HJM<T> {
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
  pub m: Option<usize>,
}

impl Sampling3DExt<f64> for HJM<f64> {
  fn sample(&self) -> [Array1<f64>; 3] {
    let t_max = self.t.unwrap_or(1.0);
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut r = Array1::<f64>::zeros(self.n);
    let mut p = Array1::<f64>::zeros(self.n);
    let mut f = Array1::<f64>::zeros(self.n);

    let gn1 = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let gn2 = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
    let gn3 = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..self.n {
      let t = i as f64 * dt;

      r[i] = r[i - 1] + (self.a)(t) * dt + (self.b)(t) * gn1[i - 1];
      p[i] =
        p[i - 1] + (self.p)(t, t_max) * ((self.q)(t, t_max) * dt + (self.v)(t, t_max) * gn2[i - 1]);
      f[i] = f[i - 1] + (self.alpha)(t, t_max) * dt + (self.sigma)(t, t_max) * gn3[i - 1];
    }

    [r, p, f]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl Sampling3DExt<f32> for HJM<f32> {
  fn sample(&self) -> [Array1<f32>; 3] {
    let t_max = self.t.unwrap_or(1.0);
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut r = Array1::<f32>::zeros(self.n);
    let mut p = Array1::<f32>::zeros(self.n);
    let mut f = Array1::<f32>::zeros(self.n);

    let gn1 = Array1::random(self.n, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);
    let gn2 = Array1::random(self.n, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);
    let gn3 = Array1::random(self.n, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);

    for i in 1..self.n {
      let t = i as f32 * dt;

      r[i] = r[i - 1] + (self.a)(t) * dt + (self.b)(t) * gn1[i - 1];
      p[i] =
        p[i - 1] + (self.p)(t, t_max) * ((self.q)(t, t_max) * dt + (self.v)(t, t_max) * gn2[i - 1]);
      f[i] = f[i - 1] + (self.alpha)(t, t_max) * dt + (self.sigma)(t, t_max) * gn3[i - 1];
    }

    [r, p, f]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

use impl_new_derive::ImplNew;
use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Sampling2DExt;
use crate::stochastic::Sampling3DExt;

#[derive(ImplNew)]
pub struct Bates1996<D, T>
where
  D: Distribution<T> + Send + Sync,
{
  pub mu: Option<T>,
  pub b: Option<T>,
  pub r: Option<T>,
  pub r_f: Option<T>,
  pub lambda: T,
  pub k: T,
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub rho: T,
  pub n: usize,
  pub s0: Option<T>,
  pub v0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
  pub cgns: CGNS<T>,
  pub cpoisson: CompoundPoisson<D, T>,
}

impl<D> Sampling2DExt<f64> for Bates1996<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut s = Array1::<f64>::zeros(self.n);
    let mut v = Array1::<f64>::zeros(self.n);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    let drift = match (self.mu, self.b, self.r, self.r_f) {
      (Some(r), Some(r_f), ..) => r - r_f,
      (Some(b), ..) => b,
      _ => self.mu.unwrap(),
    };

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1]
        + jumps.sum();

      let dv = (self.alpha - self.beta * v[i - 1]) * dt + self.sigma * v[i - 1] * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }

    [s, v]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl<D> Sampling2DExt<f32> for Bates1996<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn sample(&self) -> [Array1<f32>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;

    let mut s = Array1::<f32>::zeros(self.n);
    let mut v = Array1::<f32>::zeros(self.n);

    s[0] = self.s0.unwrap_or(0.0);
    v[0] = self.v0.unwrap_or(0.0);

    let drift = match (self.mu, self.b, self.r, self.r_f) {
      (Some(r), Some(r_f), ..) => r - r_f,
      (Some(b), ..) => b,
      _ => self.mu.unwrap(),
    };

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      s[i] = s[i - 1]
        + (drift - self.lambda * self.k) * s[i - 1] * dt
        + s[i - 1] * v[i - 1].sqrt() * cgn1[i - 1]
        + jumps.sum();

      let dv = (self.alpha - self.beta * v[i - 1]) * dt + self.sigma * v[i - 1] * cgn2[i - 1];

      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v[i - 1] + dv).abs(),
        false => (v[i - 1] + dv).max(0.0),
      }
    }

    [s, v]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Normal;

  use super::*;
  use crate::plot_2d;
  use crate::stochastic::process::poisson::Poisson;
  use crate::stochastic::N;
  use crate::stochastic::S0;
  use crate::stochastic::X0;

  #[test]
  fn bates1996__length_equals_n() {
    let bates1996 = Bates1996::new(
      Some(3.0),
      None,
      None,
      None,
      1.0,
      2.25,
      2.5,
      0.9,
      0.1,
      0.1,
      N,
      Some(S0),
      Some(X0),
      None,
      None,
      None,
      CGNS::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(bates1996.sample()[0].len(), N);
    assert_eq!(bates1996.sample()[1].len(), N);
  }

  #[test]
  fn bates1996__starts_with_x0() {
    let bates1996 = Bates1996::new(
      Some(3.0),
      None,
      None,
      None,
      1.0,
      2.25,
      2.5,
      0.9,
      0.1,
      0.1,
      N,
      Some(S0),
      Some(X0),
      None,
      None,
      None,
      CGNS::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(bates1996.sample()[0][0], S0);
    assert_eq!(bates1996.sample()[1][0], X0);
  }

  #[test]
  fn bates1996__plot() {
    let bates1996 = Bates1996::new(
      Some(3.0),
      None,
      None,
      None,
      1.0,
      2.25,
      2.5,
      0.9,
      0.1,
      0.1,
      N,
      Some(S0),
      Some(X0),
      None,
      None,
      None,
      CGNS::new(0.7, N, None, None),
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    let [s, v] = bates1996.sample();
    plot_2d!(s, "Bates1996 process (s)", v, "Bates1996 process (v)");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn bates1996__malliavin() {
    unimplemented!()
  }
}

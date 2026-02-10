use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct Bates1996<T, D>
where
  T: Float,
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
  cgns: CGNS<T>,
  pub cpoisson: CompoundPoisson<T, D>,
}

impl<T, D> Bates1996<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    mu: Option<T>,
    b: Option<T>,
    r: Option<T>,
    r_f: Option<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    n: usize,
    s0: Option<T>,
    v0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      mu,
      b,
      r,
      r_f,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      use_sym,
      cgns: CGNS::new(rho, n - 1, t),
      cpoisson,
    }
  }
}

impl<T, D> Process<T> for Bates1996<T, D>
where
  T: Float,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];
  type Noise = CGNS<T>;

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
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = noise_fn(&self.cgns);

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero());

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
        false => (v[i - 1] + dv).max(T::zero()),
      }
    }

    [s, v]
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
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
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
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
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
      CompoundPoisson::new(
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64)),
      ),
    );

    let [s, v] = bates1996.sample();
    plot_2d!(s, "Bates1996 process (s)", v, "Bates1996 process (v)");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn bates1996__malliavin() {
    unimplemented!()
  }
}

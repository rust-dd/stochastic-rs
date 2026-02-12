use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

/// Kou process
///
/// https://www.columbia.edu/~sk75/MagSci02.pdf
///
pub struct KOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,
  pub sigma: T,
  pub lambda: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  gn: Gn<T>,
}

impl<T, D> KOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Create a new Kou process
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for KOU<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut merton = Array1::<T>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha
          - self.sigma.powf(T::from_usize_(2)) / T::from_usize_(2)
          - self.lambda * self.theta)
          * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
  }
}

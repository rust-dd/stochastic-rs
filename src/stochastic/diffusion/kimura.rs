use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
pub struct Kimura<T: Float> {
  pub a: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub gn: Gn<T>,
}

impl<T: Float> Kimura<T> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Kimura {
      a,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> ProcessExt<T> for Kimura<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      // enforce [0,1] domain when computing coefficients
      let xi = x[i - 1].clamp(T::zero(), T::one());
      let sqrt_term = (xi * (T::one() - xi)).sqrt();
      let drift = self.a * xi * (T::one() - xi) * dt;
      let diff = self.sigma * sqrt_term * gn[i - 1];
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      x[i] = next;
    }

    x
  }
}

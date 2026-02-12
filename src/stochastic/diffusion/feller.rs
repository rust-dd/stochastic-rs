use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
pub struct FellerLogistic<T: FloatExt> {
  pub kappa: T,
  pub theta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
  gn: Gn<T>,
}

impl<T: FloatExt> FellerLogistic<T> {
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FellerLogistic<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1].max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(T::zero()),
      };
    }

    x
  }
}

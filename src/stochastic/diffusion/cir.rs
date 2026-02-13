use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Cox-Ingersoll-Ross (CIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW(t)
/// where X(t) is the CIR process.
pub struct CIR<T: FloatExt> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  gn: Gn<T>,
}

impl<T: FloatExt> CIR<T> {
  /// Create a new CIR process.
  pub fn new(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(
      T::from_usize_(2) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CIR<T> {
  type Output = Array1<T>;

  /// Sample the Cox-Ingersoll-Ross (CIR) process
  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut cir = Array1::<T>::zeros(self.n);
    cir[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let dcir = self.theta * (self.mu - cir[i - 1]) * dt
        + self.sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1];

      cir[i] = match self.use_sym.unwrap_or(false) {
        true => (cir[i - 1] + dcir).abs(),
        false => (cir[i - 1] + dcir).max(T::zero()),
      };
    }

    cir
  }
}

py_process_1d!(PyCIR, CIR,
  sig: (theta, mu, sigma, n, x0=None, t=None, use_sym=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

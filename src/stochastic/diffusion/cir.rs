//! # CIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Cox-Ingersoll-Ross (CIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW(t)
/// where X(t) is the CIR process.
pub struct CIR<T: FloatExt> {
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
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
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CIR<T> {
  type Output = Array1<T>;

  /// Sample the Cox-Ingersoll-Ross (CIR) process
  fn sample(&self) -> Self::Output {
    let mut cir = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return cir;
    }

    cir[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return cir;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = cir[0];
    let mut tail_view = cir.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("CIR output tail must be contiguous");
    T::fill_standard_normal_scaled_slice(tail, sqrt_dt);

    for z in tail.iter_mut() {
      let dcir = self.theta * (self.mu - prev) * dt + diff_scale * prev.abs().sqrt() * *z;
      let next = match self.use_sym.unwrap_or(false) {
        true => (prev + dcir).abs(),
        false => (prev + dcir).max(T::zero()),
      };
      *z = next;
      prev = next;
    }

    cir
  }
}

py_process_1d!(PyCIR, CIR,
  sig: (theta, mu, sigma, n, x0=None, t=None, use_sym=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

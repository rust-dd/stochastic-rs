//! # fCIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t}\,dB_t^H
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Fractional Cox-Ingersoll-Ross (FCIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the FCIR process.
pub struct FCIR<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
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
  fgn: FGN<T>,
}

impl<T: FloatExt> FCIR<T> {
  #[must_use]
  pub fn new(
    hurst: T,
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
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FCIR<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut fcir = Array1::<T>::zeros(self.n);
    fcir[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
        + self.sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1];

      fcir[i] = match self.use_sym.unwrap_or(false) {
        true => (fcir[i - 1] + dfcir).abs(),
        false => (fcir[i - 1] + dfcir).max(T::zero()),
      };
    }

    fcir
  }
}

py_process_1d!(PyFCIR, FCIR,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, use_sym=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

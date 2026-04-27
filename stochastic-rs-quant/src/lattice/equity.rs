//! # Equity-option lattice models
//!
//! Bridges [`super::tree::BinomialTree`] to [`crate::traits::ModelPricer`]
//! so Cox-Ross-Rubinstein style trees plug into the vol-surface pipeline
//! alongside Fourier and SABR pricers.

use ndarray::Array1;

use super::tree::BinomialTree;
use crate::OptionType;
use crate::traits::FloatExt;
use crate::traits::ModelPricer;

/// Cox-Ross-Rubinstein binomial-tree model for European options.
///
/// $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$,
/// $p = (e^{(r-q)\Delta t} - d)/(u - d)$.
#[derive(Debug, Clone, Copy)]
pub struct CrrModel<T: FloatExt> {
  /// Lognormal volatility $\sigma$.
  pub sigma: T,
  /// Number of tree steps.
  pub steps: usize,
}

impl<T: FloatExt> CrrModel<T> {
  /// Construct a CRR model.
  pub fn new(sigma: T, steps: usize) -> Self {
    assert!(steps >= 1, "steps must be positive");
    Self { sigma, steps }
  }

  fn price_european(&self, s: T, k: T, r: T, q: T, tau: T, option_type: OptionType) -> T {
    let dt = tau / T::from_usize_(self.steps);
    let sqrt_dt = dt.sqrt();
    let up = (self.sigma * sqrt_dt).exp();
    let down = T::one() / up;
    let drift = ((r - q) * dt).exp();
    let p = (drift - down) / (up - down);
    let tree = BinomialTree::from_crr(s, up, down, p, self.steps, dt);
    let discount = (-r * dt).exp();
    let terminal_states = tree.states.last().expect("tree has at least one level");
    let terminal_values = Array1::from_iter(terminal_states.iter().map(|&state| match option_type {
      OptionType::Call => (state - k).max(T::zero()),
      OptionType::Put => (k - state).max(T::zero()),
    }));
    tree.backward_induct(terminal_values, |_, _, _| discount)
  }
}

impl ModelPricer for CrrModel<f64> {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_european(s, k, r, q, tau, OptionType::Call)
  }

  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_european(s, k, r, q, tau, OptionType::Put)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn crr_call_recovers_black_scholes_at_high_steps() {
    let model = CrrModel::new(0.2_f64, 200);
    let bs_call = 10.4506; // Black-Scholes call at S=100, K=100, r=0.05, q=0, T=1, sigma=0.2
    let crr_call = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    assert!((crr_call - bs_call).abs() < 0.05, "got {}", crr_call);
  }

  #[test]
  fn crr_put_call_parity() {
    let model = CrrModel::new(0.25_f64, 100);
    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let q = 0.02;
    let tau = 0.5;
    let c = model.price_call(s, k, r, q, tau);
    let p = model.price_put(s, k, r, q, tau);
    let parity = c - p - s * (-q * tau).exp() + k * (-r * tau).exp();
    assert!(parity.abs() < 0.01, "parity residual {parity}");
  }
}

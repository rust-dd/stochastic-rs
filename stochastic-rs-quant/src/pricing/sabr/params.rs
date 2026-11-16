//! The SABR parameters in Hagan's order: `dF = α F^β dW¹`, `dα = ν α dW²`, `d⟨W¹, W²⟩ = ρ dt`.

/// Hagan, Kumar, Lesniewski & Woodward (2002), "Managing Smile Risk", Wilmott Magazine, 84–108.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SabrParams {
  pub alpha: f64,
  /// CEV exponent; `0` normal, `1` lognormal.
  pub beta: f64,
  pub rho: f64,
  pub nu: f64,
}

impl SabrParams {
  /// Panics unless `α > 0`, `β ∈ [0, 1]`, `ρ ∈ (−1, 1)`, `ν ≥ 0` and every value is finite.
  pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
    assert!(
      alpha.is_finite() && alpha > 0.0,
      "alpha must satisfy `alpha.is_finite() && alpha > 0.0`, got alpha = {alpha:?}"
    );
    assert!(
      (0.0..=1.0).contains(&beta),
      "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = {beta:?}"
    );
    assert!(
      rho > -1.0 && rho < 1.0,
      "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = {rho:?}"
    );
    assert!(
      nu.is_finite() && nu >= 0.0,
      "nu must satisfy `nu.is_finite() && nu >= 0.0`, got nu = {nu:?}"
    );
    Self {
      alpha,
      beta,
      rho,
      nu,
    }
  }
}

//! The six rough Heston parameters, in the convention of the `RlHeston` process.

/// `V = V0 + K∗κ(θ − V) + K∗ν√V dW`, `K(t) = t^{H−½}/Γ(H+½)`, `d⟨W^S, W^v⟩ = ρ dt`.
/// Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, (3), DOI 10.1111/mafi.12173
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoughHestonParams {
  /// Initial variance (not a volatility).
  pub v0: f64,
  /// Mean-reversion speed, El Euch & Rosenbaum's `λ`.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance; El Euch & Rosenbaum write it `λν`.
  pub nu: f64,
  /// Spot–variance correlation.
  pub rho: f64,
  /// Hurst exponent of the variance, `α − ½`.
  pub hurst: f64,
}

impl RoughHestonParams {
  /// Panics unless every value is finite, `v0, θ ≥ 0`, `κ, ν > 0`, `ρ ∈ [−1, 1]` and `0 < H < ½`.
  pub fn new(v0: f64, kappa: f64, theta: f64, nu: f64, rho: f64, hurst: f64) -> Self {
    Self {
      v0,
      kappa,
      theta,
      nu,
      rho,
      hurst,
    }
    .checked()
  }

  /// The first of `new`'s checks these values fail, in its panic's words; `None` if all pass.
  #[doc(hidden)]
  pub fn violation(&self) -> Option<String> {
    let Self {
      v0,
      kappa,
      theta,
      nu,
      rho,
      hurst,
    } = *self;
    if !(v0.is_finite() && v0 >= 0.0) {
      Some(format!(
        "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = {v0:?}"
      ))
    } else if !(kappa.is_finite() && kappa > 0.0) {
      Some(format!(
        "kappa must satisfy `kappa.is_finite() && kappa > 0.0`, got kappa = {kappa:?}"
      ))
    } else if !(theta.is_finite() && theta >= 0.0) {
      Some(format!(
        "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = {theta:?}"
      ))
    } else if !(nu.is_finite() && nu > 0.0) {
      Some(format!(
        "nu must satisfy `nu.is_finite() && nu > 0.0`, got nu = {nu:?}"
      ))
    } else if !(-1.0..=1.0).contains(&rho) {
      Some(format!(
        "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = {rho:?}"
      ))
    } else if !(hurst > 0.0 && hurst < 0.5) {
      Some(format!(
        "hurst must satisfy `hurst > 0.0 && hurst < 0.5`, got hurst = {hurst:?}"
      ))
    } else {
      None
    }
  }

  /// The same values after `new`'s checks, for a constructor handed a literal.
  pub(crate) fn checked(self) -> Self {
    if let Some(reason) = self.violation() {
      panic!("{reason}");
    }
    self
  }

  /// `α = H + ½`, the order of the fractional Riccati equation.
  pub(crate) fn alpha(&self) -> f64 {
    self.hurst + 0.5
  }
}

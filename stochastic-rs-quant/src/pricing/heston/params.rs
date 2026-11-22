//! The five Heston parameters every Heston pricer, engine and Fourier model is built from.

/// `dv = κ(θ − v)dt + σ√v dW`, `d⟨W^S, W^v⟩ = ρ dt`, started at `v0`.
/// Reference: Heston (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonParams {
  /// Initial variance (not a volatility).
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma: f64,
  /// Spot–variance correlation.
  pub rho: f64,
}

impl HestonParams {
  /// Panics unless every value is finite, `v0, θ, κθ ≥ 0`, `σ > 0` and `ρ ∈ [−1, 1]`: `κθ` is the
  /// drift at `v = 0`, which keeps the variance non-negative; Feller's condition is not imposed.
  pub fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
    let params = Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    };
    if let Some(reason) = params.violation() {
      panic!("{reason}");
    }
    params
  }

  /// The first of `new`'s checks these values fail, in its panic's words; `None` if all pass.
  #[doc(hidden)]
  pub fn violation(&self) -> Option<String> {
    let Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    } = *self;
    if !(v0.is_finite() && v0 >= 0.0) {
      Some(format!(
        "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = {v0:?}"
      ))
    } else if !kappa.is_finite() {
      Some(format!(
        "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
      ))
    } else if !(theta.is_finite() && theta >= 0.0) {
      Some(format!(
        "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = {theta:?}"
      ))
    } else if kappa * theta < 0.0 {
      Some(format!(
        "kappa must satisfy `kappa * theta >= 0.0`, got kappa = {kappa:?}"
      ))
    } else if !(sigma.is_finite() && sigma > 0.0) {
      Some(format!(
        "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = {sigma:?}"
      ))
    } else if !(-1.0..=1.0).contains(&rho) {
      Some(format!(
        "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = {rho:?}"
      ))
    } else {
      None
    }
  }

  /// The same values after `new`'s checks, for a constructor handed a literal.
  pub(crate) fn checked(self) -> Self {
    Self::new(self.v0, self.kappa, self.theta, self.sigma, self.rho)
  }
}

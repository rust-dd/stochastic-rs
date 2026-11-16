//! The five Heston parameters every Heston pricer, engine and Fourier model is built from.

/// `dv = κ(θ − v)dt + σ√v dW`, `d⟨W^S, W^v⟩ = ρ dt`, started at `v0`.
/// Heston, S. L. (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327.
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
  /// Panics unless every value is finite, `v0, θ ≥ 0`, `σ > 0` and `ρ ∈ [−1, 1]`; a `κ ≤ 0` is a
  /// non-stationary but well-defined affine model, and Feller's `2κθ ≥ σ²` is not imposed.
  pub fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
    assert!(
      v0.is_finite() && v0 >= 0.0,
      "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = {v0:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite() && theta >= 0.0,
      "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = {theta:?}"
    );
    assert!(
      sigma.is_finite() && sigma > 0.0,
      "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = {sigma:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = {rho:?}"
    );
    Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    }
  }

  /// The same values after `new`'s checks, for a constructor handed a literal.
  pub(crate) fn checked(self) -> Self {
    Self::new(self.v0, self.kappa, self.theta, self.sigma, self.rho)
  }
}

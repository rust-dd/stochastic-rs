//! Hybrid Heston-with-jumps Fourier models: HKDE (Kou jumps) and Bates
//! (lognormal Merton jumps).

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;
use super::KouFourier;
use super::MertonJdFourier;
use super::factor_cd;
use super::heston_cumulants;
use crate::pricing::heston::HestonParams;

/// The first `(name, value, predicate, holds)` that fails, worded as a constructor panic.
fn first_violation(checks: &[(&str, f64, &str, bool)]) -> Option<String> {
  let (name, value, predicate, _) = checks.iter().find(|check| !check.3)?;
  Some(format!(
    "{name} must satisfy `{predicate}`, got {name} = {value:?}"
  ))
}

/// Hkde model parameters — Heston stochastic volatility augmented by a
/// Kou double-exponential jump component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HkdeParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma$.
  pub sigma: f64,
  /// Correlation $\rho$ between the price and variance Brownian motions.
  pub rho: f64,
  /// Poisson jump intensity $\lambda$.
  pub lambda: f64,
  /// Probability $p$ of an upward jump.
  pub p_up: f64,
  /// Upward jump rate $\eta_1>1$.
  pub eta1: f64,
  /// Downward jump rate $\eta_2>0$.
  pub eta2: f64,
}

impl HkdeParams {
  /// The first of [`HkdeFourier::new`]'s checks these values fail, in its panic's words.
  pub(crate) fn violation(&self) -> Option<String> {
    let Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      lambda,
      p_up,
      eta1,
      eta2,
    } = *self;
    first_violation(&[
      (
        "v0",
        v0,
        "v0.is_finite() && v0 >= 0.0",
        v0.is_finite() && v0 >= 0.0,
      ),
      ("kappa", kappa, "kappa.is_finite()", kappa.is_finite()),
      (
        "theta",
        theta,
        "theta.is_finite() && theta >= 0.0",
        theta.is_finite() && theta >= 0.0,
      ),
      (
        "sigma",
        sigma,
        "sigma.is_finite() && sigma > 0.0",
        sigma.is_finite() && sigma > 0.0,
      ),
      (
        "rho",
        rho,
        "(-1.0..=1.0).contains(&rho)",
        (-1.0..=1.0).contains(&rho),
      ),
      (
        "lambda",
        lambda,
        "lambda.is_finite() && lambda >= 0.0",
        lambda.is_finite() && lambda >= 0.0,
      ),
      (
        "p_up",
        p_up,
        "(0.0..=1.0).contains(&p_up)",
        (0.0..=1.0).contains(&p_up),
      ),
      (
        "eta1",
        eta1,
        "eta1.is_finite() && eta1 > 1.0",
        eta1.is_finite() && eta1 > 1.0,
      ),
      (
        "eta2",
        eta2,
        "eta2.is_finite() && eta2 > 0.0",
        eta2.is_finite() && eta2 > 0.0,
      ),
    ])
  }
}

/// Heston + Kou Double-Exponential jump model (Hkde) for Fourier pricing.
///
/// $$
/// dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t
/// $$
/// $$
/// dv_t = \kappa(\theta-v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v
/// $$
///
/// where $J_t$ follows a Kou double-exponential distribution with upward jump rate $\eta_1$,
/// downward jump rate $\eta_2$, and probability of upward jump $p$.
///
/// Source:
/// - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)
#[derive(Debug, Clone)]
pub struct HkdeFourier {
  params: HkdeParams,
}

impl HkdeFourier {
  /// Panics unless every value is finite, `v0, θ, λ ≥ 0`, `σ > 0`, `ρ ∈ [−1, 1]`, `p ∈ [0, 1]`,
  /// `η1 > 1` and `η2 > 0`.
  pub fn new(params: HkdeParams) -> Self {
    if let Some(reason) = params.violation() {
      panic!("{reason}");
    }
    Self { params }
  }

  /// For the fit, whose trial point may sit outside the domain and must price NaN, not panic.
  pub(crate) fn unchecked(params: HkdeParams) -> Self {
    Self { params }
  }

  pub fn params(&self) -> HkdeParams {
    self.params
  }

  fn heston(&self) -> HestonParams {
    let HkdeParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      ..
    } = self.params;
    HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    }
  }

  /// The jumps as a Kou model without diffusion: its compensator is `−λk̄`, `k̄ = E[e^J] − 1`.
  fn jumps(&self) -> KouFourier {
    KouFourier {
      sigma: 0.0,
      lambda: self.params.lambda,
      p_up: self.params.p_up,
      eta1: self.params.eta1,
      eta2: self.params.eta2,
    }
  }
}

impl FourierModelExt for HkdeFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c_heston, d_heston) = factor_cd(&self.heston(), t, xi);
    let jumps = self.jumps();
    let drift = Complex64::i() * xi * jumps.compensator() * t;
    (c_heston + d_heston * self.params.v0 + drift + jumps.jump_exponent(xi) * t).exp()
  }

  /// The Heston factor and the compensated jumps are independent, so their cumulants add.
  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.heston(), t).plus(self.jumps().cumulants(t))
  }
}

/// SVJ / Bates model parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvjParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma$.
  pub sigma: f64,
  /// Correlation $\rho$ between price and variance Brownian motions.
  pub rho: f64,
  /// Jump intensity $\lambda$ (Poisson arrival rate).
  pub lambda: f64,
  /// Mean log-jump size $\mu_J$.
  pub mu_j: f64,
  /// Jump-size volatility $\sigma_J$.
  pub sigma_j: f64,
}

impl SvjParams {
  /// The first of [`BatesFourier::new`]'s checks these values fail, in its panic's words.
  pub(crate) fn violation(&self) -> Option<String> {
    let Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      lambda,
      mu_j,
      sigma_j,
    } = *self;
    first_violation(&[
      (
        "v0",
        v0,
        "v0.is_finite() && v0 >= 0.0",
        v0.is_finite() && v0 >= 0.0,
      ),
      ("kappa", kappa, "kappa.is_finite()", kappa.is_finite()),
      (
        "theta",
        theta,
        "theta.is_finite() && theta >= 0.0",
        theta.is_finite() && theta >= 0.0,
      ),
      (
        "sigma",
        sigma,
        "sigma.is_finite() && sigma > 0.0",
        sigma.is_finite() && sigma > 0.0,
      ),
      (
        "rho",
        rho,
        "(-1.0..=1.0).contains(&rho)",
        (-1.0..=1.0).contains(&rho),
      ),
      (
        "lambda",
        lambda,
        "lambda.is_finite() && lambda >= 0.0",
        lambda.is_finite() && lambda >= 0.0,
      ),
      ("mu_j", mu_j, "mu_j.is_finite()", mu_j.is_finite()),
      (
        "sigma_j",
        sigma_j,
        "sigma_j.is_finite() && sigma_j >= 0.0",
        sigma_j.is_finite() && sigma_j >= 0.0,
      ),
    ])
  }
}

/// Bates / Stochastic Volatility with Jumps (SVJ) model for Fourier pricing.
///
/// Heston + Merton log-normal jumps:
///
/// $$
/// dS = (r-q)S\,dt + \sqrt{v}\,S\,dW^S + (e^J - 1)S\,dN_t,
/// \quad dv = \kappa(\theta - v)\,dt + \sigma\sqrt{v}\,dW^v
/// $$
///
/// Reference: Bates (1996), "Jumps and Stochastic Volatility"
#[derive(Debug, Clone)]
pub struct BatesFourier {
  params: SvjParams,
}

impl BatesFourier {
  /// Panics unless every value is finite, `v0, θ, λ, σ_J ≥ 0`, `σ > 0` and `ρ ∈ [−1, 1]`.
  pub fn new(params: SvjParams) -> Self {
    if let Some(reason) = params.violation() {
      panic!("{reason}");
    }
    Self { params }
  }

  pub fn params(&self) -> SvjParams {
    self.params
  }

  fn heston(&self) -> HestonParams {
    let SvjParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      ..
    } = self.params;
    HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    }
  }

  /// The jumps as a Merton model without diffusion: its compensator is `−λk̄`, `k̄ = E[e^J] − 1`.
  fn jumps(&self) -> MertonJdFourier {
    MertonJdFourier {
      sigma: 0.0,
      lambda: self.params.lambda,
      mu_j: self.params.mu_j,
      sigma_j: self.params.sigma_j,
    }
  }
}

impl FourierModelExt for BatesFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c_heston, d_heston) = factor_cd(&self.heston(), t, xi);
    let jumps = self.jumps();
    let drift = Complex64::i() * xi * jumps.compensator() * t;
    (c_heston + d_heston * self.params.v0 + drift + jumps.jump_exponent(xi) * t).exp()
  }

  /// The Heston factor and the compensated jumps are independent, so their cumulants add.
  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.heston(), t).plus(self.jumps().cumulants(t))
  }
}

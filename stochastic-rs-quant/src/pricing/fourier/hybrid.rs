//! Hybrid Heston-with-jumps Fourier models: HKDE (Kou jumps) and Bates
//! (lognormal Merton jumps).

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;
use super::KouFourier;
use super::MertonJdFourier;
use super::factor_cd;
use super::heston_cumulants;
use crate::calibration::hkde::HkdeParams;
use crate::calibration::svj::SvjParams;
use crate::pricing::heston::HestonParams;

/// Heston + Kou Double-Exponential jump model (Hkde) for Fourier pricing.
///
/// $$
/// dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t
/// $$
/// $$
/// dv_t = \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v
/// $$
///
/// where $J_t$ follows a Kou double-exponential distribution with upward jump rate $\eta_1$,
/// downward jump rate $\eta_2$, and probability of upward jump $p$.
///
/// Source:
/// - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)
#[derive(Debug, Clone)]
pub struct HkdeFourier {
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma_v: f64,
  /// Correlation.
  pub rho: f64,
  /// Jump intensity (Poisson rate).
  pub lam: f64,
  /// Probability of upward jump.
  pub p_up: f64,
  /// Upward jump rate parameter (eta1 > 1 required for finite expectation).
  pub eta1: f64,
  /// Downward jump rate parameter (eta2 > 0).
  pub eta2: f64,
}

impl HkdeFourier {
  /// Panics unless every value is finite, `v0, θ, λ ≥ 0`, `σ_v > 0`, `ρ ∈ [−1, 1]`, `p ∈ [0, 1]`,
  /// `η1 > 1` and `η2 > 0`.
  pub fn new(params: HkdeParams) -> Self {
    if let Some(reason) = params.violation() {
      panic!("{reason}");
    }
    let HkdeParams {
      v0,
      kappa,
      theta,
      sigma_v,
      rho,
      lambda,
      p_up,
      eta1,
      eta2,
    } = params;
    Self {
      v0,
      kappa,
      theta,
      sigma_v,
      rho,
      lam: lambda,
      p_up,
      eta1,
      eta2,
    }
  }

  fn heston(&self) -> HestonParams {
    HestonParams {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma_v,
      rho: self.rho,
    }
  }

  /// The jumps as a Kou model without diffusion: its compensator is `−λk̄`, `k̄ = E[e^J] − 1`.
  fn jumps(&self) -> KouFourier {
    KouFourier {
      sigma: 0.0,
      lambda: self.lam,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
    }
  }
}

impl FourierModelExt for HkdeFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c_heston, d_heston) = factor_cd(&self.heston(), t, xi);
    let jumps = self.jumps();
    let drift = Complex64::i() * xi * jumps.compensator() * t;
    (c_heston + d_heston * self.v0 + drift + jumps.jump_exponent(xi) * t).exp()
  }

  /// The Heston factor and the compensated jumps are independent, so their cumulants add.
  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.heston(), t).plus(self.jumps().cumulants(t))
  }
}

/// Bates / Stochastic Volatility with Jumps (SVJ) model for Fourier pricing.
///
/// Heston + Merton log-normal jumps:
///
/// $$
/// dS = (r-q)S\,dt + \sqrt{v}\,S\,dW^S + (e^J - 1)S\,dN_t,
/// \quad dv = \kappa(\theta - v)\,dt + \sigma_v\sqrt{v}\,dW^v
/// $$
///
/// Reference: Bates (1996), "Jumps and Stochastic Volatility"
#[derive(Debug, Clone)]
pub struct BatesFourier {
  pub v0: f64,
  pub kappa: f64,
  pub theta: f64,
  pub sigma_v: f64,
  pub rho: f64,
  pub lambda: f64,
  pub mu_j: f64,
  pub sigma_j: f64,
}

impl BatesFourier {
  /// Panics unless every value is finite, `v0, θ, λ, σ_J ≥ 0`, `σ_v > 0` and `ρ ∈ [−1, 1]`.
  pub fn new(params: SvjParams) -> Self {
    if let Some(reason) = params.violation() {
      panic!("{reason}");
    }
    let SvjParams {
      v0,
      kappa,
      theta,
      sigma_v,
      rho,
      lambda,
      mu_j,
      sigma_j,
    } = params;
    Self {
      v0,
      kappa,
      theta,
      sigma_v,
      rho,
      lambda,
      mu_j,
      sigma_j,
    }
  }

  fn heston(&self) -> HestonParams {
    HestonParams {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma_v,
      rho: self.rho,
    }
  }

  /// The jumps as a Merton model without diffusion: its compensator is `−λk̄`, `k̄ = E[e^J] − 1`.
  fn jumps(&self) -> MertonJdFourier {
    MertonJdFourier {
      sigma: 0.0,
      lambda: self.lambda,
      mu_j: self.mu_j,
      sigma_j: self.sigma_j,
    }
  }
}

impl FourierModelExt for BatesFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c_heston, d_heston) = factor_cd(&self.heston(), t, xi);
    let jumps = self.jumps();
    let drift = Complex64::i() * xi * jumps.compensator() * t;
    (c_heston + d_heston * self.v0 + drift + jumps.jump_exponent(xi) * t).exp()
  }

  /// The Heston factor and the compensated jumps are independent, so their cumulants add.
  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.heston(), t).plus(self.jumps().cumulants(t))
  }
}

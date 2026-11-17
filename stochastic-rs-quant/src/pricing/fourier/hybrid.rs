//! Hybrid Heston-with-jumps Fourier models: HKDE (Kou jumps) and Bates
//! (lognormal Merton jumps).

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;
use super::heston_cumulants;
use crate::calibration::hkde::HkdeParams;
use crate::calibration::svj::SvjParams;

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

  /// Martingale drift `−λk̄` of the Kou jumps, `k̄ = E[e^J] − 1`.
  fn compensator(&self) -> f64 {
    let k_bar = self.p_up * self.eta1 / (self.eta1 - 1.0)
      + (1.0 - self.p_up) * self.eta2 / (self.eta2 + 1.0)
      - 1.0;
    -self.lam * k_bar
  }
}

impl FourierModelExt for HkdeFourier {
  /// Heston diffusion uses the Albrecher-Mayer-Schoutens-Tistaert (2007)
  /// "Little Heston Trap" form (`g̃ = 1/g_original`, `exp(-d·t)`); the Kou jump
  /// CF is added linearly on top. Required for principal-branch stability at
  /// long τ / high `|ρ|`.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let sigma_v2 = self.sigma_v * self.sigma_v;

    let rsi = self.rho * self.sigma_v * i;
    let d = ((self.kappa - rsi * xi).powi(2) + sigma_v2 * (i * xi + xi * xi)).sqrt();
    let g = (self.kappa - rsi * xi - d) / (self.kappa - rsi * xi + d);
    let exp_dt = (-d * t).exp();

    let c_heston = (self.kappa * self.theta / sigma_v2)
      * ((self.kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_heston = ((self.kappa - rsi * xi - d) / sigma_v2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    let jump_cf = self.lam
      * ((1.0 - self.p_up) * self.eta2 / (Complex64::new(self.eta2, 0.0) + i * xi)
        + self.p_up * self.eta1 / (Complex64::new(self.eta1, 0.0) - i * xi)
        - 1.0);

    let drift_correction = self.compensator();

    (c_heston + d_heston * self.v0 + i * xi * drift_correction * t + jump_cf * t).exp()
  }

  /// The Heston factor's cumulants plus the jumps', `λtE[Jⁿ]` (Cont & Tankov 2004, Prop. 3.13)
  /// and the compensator's drift.
  fn cumulants(&self, t: f64) -> Cumulants {
    let heston = heston_cumulants(self.v0, self.kappa, self.theta, self.sigma_v, self.rho, t);
    let mean_j = self.p_up / self.eta1 - (1.0 - self.p_up) / self.eta2;
    let c1_j = (self.lam * mean_j + self.compensator()) * t;
    let c2_j = 2.0
      * self.lam
      * t
      * (self.p_up / (self.eta1 * self.eta1) + (1.0 - self.p_up) / (self.eta2 * self.eta2));
    let c4_j =
      24.0 * self.lam * t * (self.p_up / self.eta1.powi(4) + (1.0 - self.p_up) / self.eta2.powi(4));

    Cumulants {
      c1: heston.c1 + c1_j,
      c2: heston.c2 + c2_j,
      c4: heston.c4.map(|c4| c4 + c4_j),
    }
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

  /// Martingale drift `−λk̄` of the lognormal jumps, `k̄ = e^{μ_J + σ_J²/2} − 1`.
  fn compensator(&self) -> f64 {
    let k_bar = (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0;
    -self.lambda * k_bar
  }
}

impl FourierModelExt for BatesFourier {
  /// Heston diffusion uses the Albrecher-Mayer-Schoutens-Tistaert (2007)
  /// "Little Heston Trap" form (`g̃ = 1/g_original`, `exp(-d·t)`); the Merton
  /// log-normal jump CF is added linearly on top. Required for principal-branch
  /// stability at long τ / high `|ρ|`.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let sigma_v2 = self.sigma_v * self.sigma_v;

    let rsi = self.rho * self.sigma_v * i;
    let d = ((self.kappa - rsi * xi).powi(2) + sigma_v2 * (i * xi + xi * xi)).sqrt();
    let g = (self.kappa - rsi * xi - d) / (self.kappa - rsi * xi + d);
    let exp_dt = (-d * t).exp();

    let c_heston = (self.kappa * self.theta / sigma_v2)
      * ((self.kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_heston = ((self.kappa - rsi * xi - d) / sigma_v2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    let jump_cf = self.lambda
      * ((i * self.mu_j * xi - 0.5 * self.sigma_j * self.sigma_j * xi * xi).exp() - 1.0);
    let drift_correction = self.compensator();

    (c_heston + d_heston * self.v0 + i * xi * drift_correction * t + jump_cf * t).exp()
  }

  /// The Heston factor's cumulants plus the jumps', `λtE[Jⁿ]` (Cont & Tankov 2004, Prop. 3.13)
  /// and the compensator's drift.
  fn cumulants(&self, t: f64) -> Cumulants {
    let heston = heston_cumulants(self.v0, self.kappa, self.theta, self.sigma_v, self.rho, t);
    let (mu2, s2) = (self.mu_j * self.mu_j, self.sigma_j * self.sigma_j);
    let c1_j = (self.lambda * self.mu_j + self.compensator()) * t;
    let c2_j = self.lambda * (mu2 + s2) * t;
    let c4_j = self.lambda * t * (mu2 * mu2 + 6.0 * mu2 * s2 + 3.0 * s2 * s2);

    Cumulants {
      c1: heston.c1 + c1_j,
      c2: heston.c2 + c2_j,
      c4: heston.c4.map(|c4| c4 + c4_j),
    }
  }
}

//! Hybrid Heston-with-jumps Fourier models: HKDE (Kou jumps) and Bates
//! (lognormal Merton jumps).

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;

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
pub struct HKDEFourier {
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
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Jump intensity (Poisson rate).
  pub lam: f64,
  /// Probability of upward jump.
  pub p_up: f64,
  /// Upward jump rate parameter (eta1 > 1 required for finite expectation).
  pub eta1: f64,
  /// Downward jump rate parameter (eta2 > 0).
  pub eta2: f64,
}

impl FourierModelExt for HKDEFourier {
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

    let k_bar = self.p_up * self.eta1 / (self.eta1 - 1.0)
      + (1.0 - self.p_up) * self.eta2 / (self.eta2 + 1.0)
      - 1.0;

    let jump_cf = self.lam
      * ((1.0 - self.p_up) * self.eta2 / (Complex64::new(self.eta2, 0.0) + i * xi)
        + self.p_up * self.eta1 / (Complex64::new(self.eta1, 0.0) - i * xi)
        - 1.0);

    let drift_correction = -self.lam * k_bar;

    (c_heston
      + d_heston * self.v0
      + i * xi * (self.r - self.q + drift_correction) * t
      + jump_cf * t)
      .exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let ekt = (-self.kappa * t).exp();
    let c1_h = (self.r - self.q) * t + (1.0 - ekt) * (self.theta - self.v0) / (2.0 * self.kappa)
      - 0.5 * self.theta * t;
    let c2_h = self.sigma_v.powi(2) * t * self.theta / (2.0 * self.kappa);

    let c1_j = self.lam * t * (self.p_up / self.eta1 - (1.0 - self.p_up) / self.eta2);
    let c2_j = 2.0
      * self.lam
      * t
      * (self.p_up / (self.eta1 * self.eta1) + (1.0 - self.p_up) / (self.eta2 * self.eta2));
    let c4_j =
      24.0 * self.lam * t * (self.p_up / self.eta1.powi(4) + (1.0 - self.p_up) / self.eta2.powi(4));

    Cumulants {
      c1: c1_h + c1_j,
      c2: c2_h + c2_j,
      c4: c4_j,
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
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for BatesFourier {
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

    let k_bar = (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0;
    let jump_cf = self.lambda
      * ((i * self.mu_j * xi - 0.5 * self.sigma_j * self.sigma_j * xi * xi).exp() - 1.0);
    let drift_correction = -self.lambda * k_bar;

    (c_heston
      + d_heston * self.v0
      + i * xi * (self.r - self.q + drift_correction) * t
      + jump_cf * t)
      .exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let ekt = (-self.kappa * t).exp();
    let c1_h = (self.r - self.q) * t + (1.0 - ekt) * (self.theta - self.v0) / (2.0 * self.kappa)
      - 0.5 * self.theta * t;
    let c2_h = self.sigma_v.powi(2) * t * self.theta / (2.0 * self.kappa);
    let c1_j = self.lambda * self.mu_j * t;
    let c2_j = self.lambda * (self.mu_j.powi(2) + self.sigma_j.powi(2)) * t;

    Cumulants {
      c1: c1_h + c1_j,
      c2: c2_h + c2_j,
      c4: 0.0,
    }
  }
}

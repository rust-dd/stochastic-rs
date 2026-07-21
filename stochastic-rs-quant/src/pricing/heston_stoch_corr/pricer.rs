//! Carr-Madan dampened Fourier inversion for `HestonStochCorrPricer`
//! and the parameter-only [`HscmModel`] used by `ModelPricer` consumers.

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;

use super::model::HestonStochCorrPricer;
use crate::pricing::cf_quadrature::integrate_to_convergence;
use crate::traits::TimeExt;

impl HestonStochCorrPricer {
  /// Price a call option using the Carr-Madan dampened Fourier transform.
  ///
  /// C(K) = exp(−α·ln K) / π · ∫₀^∞ Re\[e^{−iu·ln K} · e^{−rτ} · φ(u−(α+1)i)
  ///        / (α² + α − u² + i(2α+1)u)\] du
  ///
  /// where α = 1.25 is the damping factor.
  pub fn price_call_carr_madan(&self) -> f64 {
    let tau = self.tau_or_from_dates();
    let alpha = 1.25_f64;
    let log_k = self.k.ln();
    let r = self.r;

    let integrand = |u: f64| -> f64 {
      if u.abs() < 1e-14 {
        return 0.0;
      }
      let u_shifted = Complex64::new(u, -(alpha + 1.0));
      let phi = self.char_func_complex(u_shifted);
      let disc_phi = (-r * tau).exp() * phi;
      let denom = Complex64::new(alpha * alpha + alpha - u * u, (2.0 * alpha + 1.0) * u);
      let val = (-Complex64::i() * u * log_k).exp() * disc_phi / denom;
      val.re
    };

    let integral = integrate_to_convergence(integrand, 0.0, 1e-8);
    let call = (-alpha * log_k).exp() * FRAC_1_PI * integral;
    call.max(0.0)
  }

  /// Price a call for a given strike (reuses the model params but different K).
  ///
  /// Useful for calibration where you price many strikes with the same model.
  pub fn price_call_at_strike(&self, k: f64) -> f64 {
    let mut p = self.clone();
    p.k = k;
    p.price_call_carr_madan()
  }
}

/// Heston stochastic-correlation model parameters (model only, no market data).
///
/// Implements [`ModelPricer`](crate::traits::ModelPricer) via the Carr-Madan FFT pricer.
#[derive(Clone, Copy, Debug)]
pub struct HscmModel {
  pub v0: f64,
  pub kappa_v: f64,
  pub theta_v: f64,
  pub sigma_v: f64,
  pub rho0: f64,
  pub kappa_r: f64,
  pub mu_r: f64,
  pub sigma_r: f64,
  pub rho2: f64,
}

impl crate::traits::ModelPricer for HscmModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let mut pricer = HestonStochCorrPricer::new(
      s,
      r,
      k,
      self.v0,
      self.kappa_v,
      self.theta_v,
      self.sigma_v,
      self.rho0,
      self.kappa_r,
      self.mu_r,
      self.sigma_r,
      self.rho2,
      tau,
    );
    pricer.q = Some(q);
    pricer.price_call_carr_madan()
  }
}

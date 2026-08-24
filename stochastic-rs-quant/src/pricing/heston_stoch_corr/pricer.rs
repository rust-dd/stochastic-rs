//! Carr-Madan dampened Fourier inversion for [`HestonStochCorrPricer`].

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;

use super::model::HestonStochCorrPricer;
use crate::pricing::cf_quadrature::integrate_to_convergence;

impl HestonStochCorrPricer {
  /// Price a call option using the Carr-Madan dampened Fourier transform.
  ///
  /// C(K) = exp(−α·ln K) / π · ∫₀^∞ Re\[e^{−iu·ln K} · e^{−rτ} · φ(u−(α+1)i)
  ///        / (α² + α − u² + i(2α+1)u)\] du
  ///
  /// where α = 1.25 is the damping factor.
  pub fn price_call_carr_madan(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let alpha = 1.25_f64;
    let log_k = k.ln();

    let integrand = |u: f64| -> f64 {
      if u.abs() < 1e-14 {
        return 0.0;
      }
      let u_shifted = Complex64::new(u, -(alpha + 1.0));
      let phi = self.char_func_complex(u_shifted, s, r, q, tau);
      let disc_phi = (-r * tau).exp() * phi;
      let denom = Complex64::new(alpha * alpha + alpha - u * u, (2.0 * alpha + 1.0) * u);
      let val = (-Complex64::i() * u * log_k).exp() * disc_phi / denom;
      val.re
    };

    let integral = integrate_to_convergence(integrand, 0.0, 1e-8);
    let call = (-alpha * log_k).exp() * FRAC_1_PI * integral;
    call.max(0.0)
  }
}

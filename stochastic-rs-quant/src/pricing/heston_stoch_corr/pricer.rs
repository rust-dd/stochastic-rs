//! Carr-Madan dampened Fourier inversion for [`HestonStochCorrPricer`].

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;

use super::model::HestonStochCorrPricer;
use crate::OptionType;
use crate::pricing::cf_quadrature::integrate_to_convergence;
use crate::pricing::digital::expired;
use crate::pricing::digital::not_positive;
use crate::pricing::digital::vanilla_payoff;

impl HestonStochCorrPricer {
  /// The call by Carr and Madan's damped transform at `α = 1.25`: the payoff at `tau = 0`; NaN
  /// past expiry and, before it, for a spot or strike that is not positive.
  ///
  /// $$
  /// C(K)=\frac{e^{-\alpha\ln K}}{\pi}\int_0^\infty\operatorname{Re}\!\left[
  /// \frac{e^{-iu\ln K}\,e^{-r\tau}\,\varphi\bigl(u-(\alpha+1)i\bigr)}
  /// {\alpha^2+\alpha-u^2+i(2\alpha+1)u}\right]du
  /// $$
  pub fn price_call_carr_madan(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if let Some(value) = expired(tau, &[r, q], || vanilla_payoff(OptionType::Call, s, k)) {
      return value;
    }
    if not_positive(&[s, k]) {
      return f64::NAN;
    }
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
    floor_price(call)
  }
}

/// Floors inversion round-off below zero at zero and lets a NaN through, which `f64::max`
/// would turn into a plausible `0.0`.
#[inline]
pub(super) fn floor_price(x: f64) -> f64 {
  if x.is_nan() { x } else { x.max(0.0) }
}

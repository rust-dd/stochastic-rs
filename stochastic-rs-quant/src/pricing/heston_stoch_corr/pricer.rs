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
    floor_price(call)
  }
}

/// Floor a price at zero **without** swallowing a `NaN`.
///
/// A floor and a poison-check are different operations, and `f64::max` runs
/// them together into one wrong answer: it returns the non-`NaN` operand, so
/// `f64::NAN.max(0.0)` is `0.0`. The deep-wing inversion really can come back
/// a few ulp below zero and that value should floor; a `NaN` has no price to
/// floor and travels on, as [the failure
/// convention](crate::traits::ModelPricer#how-pricing-fails) requires.
///
/// Every non-finite market input poisons this model's characteristic
/// function — `tau` through the Rk4 step size, `s` through `ln S`, `r` and
/// `q` through the drift — and `tau` arrives as `NaN` legitimately from
/// [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt). Same shape as
/// `pricing/fourier/pricer.rs`'s floor of the same name and
/// `VarianceSwapPricer::fair_strike_replication`'s.
#[inline]
pub(super) fn floor_price(x: f64) -> f64 {
  if x.is_nan() { x } else { x.max(0.0) }
}

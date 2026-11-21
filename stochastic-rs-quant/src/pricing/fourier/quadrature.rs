//! The Lewis and Gil–Pelaez quadrature pricers, each less a Gaussian control whose share is closed.

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;
use stochastic_rs_distributions::special::norm_cdf;

use super::FourierModelExt;
use super::pricer::floor_price;
use crate::OptionType;
use crate::pricing::cf_quadrature::CONTROL_VARIANCE_CAP;
use crate::pricing::cf_quadrature::Inversion;
use crate::pricing::cf_quadrature::Rule;
use crate::pricing::cf_quadrature::Walk;
use crate::pricing::cf_quadrature::gaussian_share;
use crate::pricing::cf_quadrature::head_floor;
use crate::pricing::cf_quadrature::reach_and_scale;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;

/// `w = −2c₁` within `[0, CONTROL_VARIANCE_CAP]`, `0` for a non-finite `c₁`: the split is exact for
/// any `w`, which only decides how small the rest is.
fn control_variance(model: &impl FourierModelExt, t: f64) -> f64 {
  let c1 = model.cumulants(t).c1;
  if c1.is_finite() {
    (-2.0 * c1).clamp(0.0, CONTROL_VARIANCE_CAP)
  } else {
    0.0
  }
}

/// Lewis's single-strike pricer less a Gaussian of variance `w = −2c₁`, whose integral is Black's
/// price; the rest is walked to its phase reach, the tail by Ooura and Mori's rule.
/// Reference: Lewis (2001), SSRN 282110, DOI 10.2139/ssrn.282110
///
/// $$
/// C=e^{-r\tau}\Bigl[\mathrm{Black}(F,K,w)-\frac{\sqrt{FK}}{\pi}\int_0^\infty
/// \operatorname{Re}\frac{e^{-iu\kappa}\bigl(\phi(u-\tfrac i2)-e^{-w(u^2+1/4)/2}\bigr)}{u^2+\tfrac14}du\Bigr],
/// \quad\kappa=\ln\frac KF
/// $$
pub struct LewisPricer;

impl LewisPricer {
  /// The European call, floored at zero: the payoff at `t = 0`; NaN past expiry and, before it,
  /// for a non-finite `s`, `k`, `r` or `q`.
  pub fn price_call(model: &impl FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    Self::price_call_walk(model, s, k, r, q, t, &mut Walk::Adaptive)
  }

  /// [`Self::price_call`] with its integral run as `walk` says.
  pub(crate) fn price_call_walk(
    model: &impl FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    walk: &mut Walk<'_, Rule>,
  ) -> f64 {
    if let Some(value) = expired(t, &[r, q], || vanilla_payoff(OptionType::Call, s, k)) {
      return value;
    }
    let forward = s * ((r - q) * t).exp();
    let kappa = (k / forward).ln();
    let w = control_variance(model, t);
    let chf = |u: f64| model.chf(t, Complex64::new(u, -0.5));
    let (reach, scale) = reach_and_scale(|u| chf(u).norm(), kappa);
    let kernel = |u: f64| Complex64::new(0.0, -u * kappa).exp() / (u * u + 0.25);
    let inversion = Inversion {
      chf_part: |u: f64| kernel(u) * chf(u),
      control_part: |u: f64| kernel(u) * (-0.5 * w * (u * u + 0.25)).exp(),
      control_frequency: -kappa,
      head_floor: None,
      reach,
      scale,
    };
    let integral = inversion.integrate(walk);
    let black = if w > 0.0 {
      let d1 = (-kappa + 0.5 * w) / w.sqrt();
      forward * norm_cdf(d1) - k * norm_cdf(d1 - w.sqrt())
    } else {
      (forward - k).max(0.0)
    };
    let call = (-r * t).exp() * (black - (forward * k).sqrt() * FRAC_1_PI * integral);
    floor_price(call)
  }
}

/// Gil-Pelaez's pricer: `P₁` and `P₂` each less a Gaussian of variance `w = −2c₁` and its own mean,
/// integrated from `u = 0`, the head in `ln u` and the tail by Ooura and Mori's rule.
/// Reference: Gil-Pelaez (1951), Biometrika 38(3–4), 481–482, DOI 10.1093/biomet/38.3-4.481
///
/// $$
/// P_j=N\Bigl(\frac{u_jw-\kappa}{\sqrt w}\Bigr)+\frac1\pi\int_0^\infty
/// \operatorname{Re}\frac{e^{-iu\kappa}\bigl(\phi_j(u)-e^{iuu_jw-u^2w/2}\bigr)}{iu}du,
/// \quad u_1=\tfrac12,\ u_2=-\tfrac12
/// $$
pub struct GilPelaezPricer;

impl GilPelaezPricer {
  /// The call `Se^{−qτ}P₁ − Ke^{−rτ}P₂`, floored at zero: the payoff at `t = 0`; NaN past expiry,
  /// for a non-finite `s`, `k`, `r` or `q` before it, and where `φ(−i) = E[e^X]` is not positive.
  pub fn price_call(model: &impl FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    if let Some(value) = expired(t, &[r, q], || vanilla_payoff(OptionType::Call, s, k)) {
      return value;
    }
    let kappa = (k / s).ln() - (r - q) * t;
    let w = control_variance(model, t);
    let share_norm = model.chf(t, Complex64::new(0.0, -1.0));
    if !(share_norm.is_finite() && share_norm.norm() > 0.0) {
      return f64::NAN;
    }
    let (reach, scale) = reach_and_scale(|u| model.chf(t, Complex64::from(u)).norm(), kappa);
    let share = |u_j: f64, shift: f64, norm: Complex64| {
      let mean = u_j * w;
      let chf = |u: f64| model.chf(t, Complex64::new(u, shift)) / norm;
      let kernel = |u: f64| Complex64::new(0.0, -u * kappa).exp() / Complex64::new(0.0, u);
      let inversion = Inversion {
        chf_part: |u: f64| kernel(u) * chf(u),
        control_part: |u: f64| kernel(u) * Complex64::new(-0.5 * u * u * w, u * mean).exp(),
        control_frequency: mean - kappa,
        head_floor: Some(head_floor(|u| (chf(u) - 1.0).norm())),
        reach,
        scale,
      };
      gaussian_share(kappa, mean, w) + FRAC_1_PI * inversion.integrate(&mut Walk::Adaptive)
    };
    let p1 = share(0.5, -1.0, share_norm);
    let p2 = share(-0.5, 0.0, Complex64::ONE);
    floor_price(s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2)
  }
}

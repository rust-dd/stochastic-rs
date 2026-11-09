//! A flat quadratic kernel about the midpoint: the mean and a middle quantile as offsets from it,
//! the odd part by `sinh` of the slope, which summed kernel values lose below ε of the width.

use super::Kernel;
use super::Truncation;
use super::panels::GL64_W;
use super::panels::GL64_X;

impl<K: Kernel> Truncation<K> {
  /// `(ψW, κW²)`, the slope and curvature at the midpoint in units of the width `W`, for a flat quadratic kernel.
  fn flat(&self) -> Option<(f64, f64)> {
    (K::QUADRATIC && !self.narrow && self.about_midpoint()).then(|| {
      let c = self.centre.x;
      let w = self.width;
      (self.kernel.d1(c, 0.0) * w, self.kernel.d2(c, 0.0) * w * w)
    })
  }

  /// The mean's offset from the midpoint in units of the width:
  /// `∫ z e^{κz²/2} sinh(ψz) / ∫ e^{κz²/2} cosh(ψz)` on `[0, ½]`.
  pub(super) fn flat_mean(&self) -> Option<f64> {
    let (slope, curvature) = self.flat()?;
    let even = |z: f64| (0.5 * curvature * z * z).exp();
    let odd = rule(0.5, |z| z * even(z) * (slope * z).sinh());
    Some(odd / rule(0.5, |z| even(z) * (slope * z).cosh()))
  }

  /// The level-`p` point's offset `z` from the midpoint in widths: `∫_0^z k = (p − ½)M + A`, `A`
  /// half the mass above the midpoint less that below, by Newton from `z = (p − ½)M + A`.
  pub(super) fn flat_quantile(&self, p: f64) -> Option<f64> {
    let (slope, curvature) = self.flat()?;
    let ln_k = |z: f64| z * (slope + 0.5 * curvature * z);
    let even = |z: f64| (0.5 * curvature * z * z).exp();
    let mass = 2.0 * rule(0.5, |z| even(z) * (slope * z).cosh());
    let want = (p - 0.5) * mass + rule(0.5, |z| even(z) * (slope * z).sinh());
    let mut z = want;
    for _ in 0..50 {
      let step = (rule(z, |u| ln_k(u).exp()) - want) / ln_k(z).exp();
      z -= step;
      if step.abs() <= 4.0 * f64::EPSILON * z.abs() {
        break;
      }
    }
    Some(z.clamp(-0.5, 0.5))
  }
}

/// `∫_0^h f` by one 64-node rule, `h` of either sign: the integrands are entire and vary by less than `e` over it.
fn rule(h: f64, f: impl Fn(f64) -> f64) -> f64 {
  let half = 0.5 * h;
  let sum: f64 = GL64_X
    .iter()
    .zip(&GL64_W)
    .map(|(x, w)| w * (f(half * (1.0 - x)) + f(half * (1.0 + x))))
    .sum();
  half * sum
}

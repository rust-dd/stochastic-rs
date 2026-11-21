//! The double-exponential rule for Fourier-type integrals, rescaled to a unit frequency.
//! Reference: Ooura (2005), Publ. RIMS 41, (2.2)–(2.5) and (3.5), DOI 10.2977/prims/1145474603

use std::f64::consts::PI;
use std::sync::OnceLock;

use num_complex::Complex64;

/// The mesh in `t`; Ooura and Mori's `β`.
const H: f64 = 0.15;
const BETA: f64 = 0.25;
/// Nodes `nh` for `n` in `−N..=N`; past `|t| = 6` every weight is below `1e-17`.
const N: i32 = 40;

/// `(x_n, φ'(nh)·(1 − e^{−iπφ̂(nh)/h}))` at `ω = 1`, `x_n = πφ(nh)/h`, `φ̂ = φ − t`.
fn nodes() -> &'static [(f64, Complex64)] {
  static NODES: OnceLock<Vec<(f64, Complex64)>> = OnceLock::new();
  NODES.get_or_init(|| {
    let alpha = BETA / (1.0 + (1.0 + PI / H).ln() / (4.0 * H)).sqrt();
    let k1 = 2.0 + alpha + BETA;
    (-N..=N)
      .filter_map(|n| {
        let t = f64::from(n) * H;
        let e = -2.0 * t - alpha * (1.0 - (-t).exp()) - BETA * t.exp_m1();
        let den = -e.exp_m1();
        let (phi, dphi) = if n == 0 {
          (1.0 / k1, (alpha - BETA + k1 * k1) / (2.0 * k1 * k1))
        } else {
          let de = -2.0 - alpha * (-t).exp() - BETA * t.exp();
          (t / den, (den + t * e.exp() * de) / (den * den))
        };
        let weight = dphi * (1.0 - Complex64::new(0.0, -PI * (phi - t) / H).exp());
        (weight.norm() > 1e-17 && phi.is_finite()).then_some((PI * phi / H, weight))
      })
      .collect()
  })
}

/// `∫_start^∞ g` for a `g` that oscillates at a frequency near `omega > 0`, and the evaluations.
pub(super) fn tail(g: impl Fn(f64) -> Complex64, start: f64, omega: f64) -> (Complex64, usize) {
  let nodes = nodes();
  let sum = nodes.iter().fold(Complex64::ZERO, |sum, &(x, weight)| {
    let value = g(start + x / omega);
    if value.is_finite() {
      sum + value * weight
    } else {
      sum
    }
  });
  (sum * (PI / omega), nodes.len())
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `∫_1^∞ e^{iu}/u du = −Ci(1) + i(π/2 − Si(1))`, mpmath at 30 digits.
  #[test]
  fn the_rule_integrates_a_slowly_decaying_oscillation() {
    let (got, _) = tail(|u| Complex64::new(0.0, u).exp() / u, 1.0, 1.0);
    let want = Complex64::new(-0.3374039229009681, 0.6247132564277136);
    assert!((got - want).norm() < 1e-12, "{got} vs {want}");
  }

  /// A frequency off its nominal value by a third still integrates: `∫_0^∞ e^{4iu/3}/(u + 1) du`.
  #[test]
  fn the_rule_holds_off_its_nominal_frequency() {
    let w = 4.0 / 3.0;
    let (got, _) = tail(|u| Complex64::new(0.0, w * u).exp() / (u + 1.0), 0.0, 1.0);
    let want = Complex64::new(0.2458152736988919, 0.5246528196133966);
    assert!((got - want).norm() < 1e-10, "{got} vs {want}");
  }
}

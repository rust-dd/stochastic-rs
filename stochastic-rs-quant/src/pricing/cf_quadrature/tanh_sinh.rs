//! The tanh-sinh rule on one panel, its error read from the last level difference.
//! Takahasi & Mori (1974), Publ. RIMS Kyoto Univ. 9, 721–741; levels as in `quadrature` 0.1.2.

use std::f64::consts::PI;

use super::nodes::LEVELS;

/// `(∫_lo^hi f, |last level difference|)`, stopping at the first level from the third whose
/// difference is under a quarter of `target`; a non-finite sample counts as zero.
pub(super) fn integrate<F>(f: &F, lo: f64, hi: f64, target: f64) -> (f64, f64)
where
  F: Fn(f64) -> f64,
{
  let half = 0.5 * (hi - lo);
  let centre = 0.5 * (lo + hi);
  let sample = |x: f64| {
    let value = f(half * x + centre);
    if value.is_finite() { value } else { 0.0 }
  };
  let level_target = 0.25 * target / half;

  let mut integral = PI * sample(0.0);
  let mut difference = f64::MAX;
  for (level, nodes) in LEVELS.iter().enumerate() {
    let added = nodes.iter().fold(0.0, |sum, &(weight, x)| {
      sum + weight * (sample(x) + sample(-x))
    });
    difference = (0.5 * integral - added).abs();
    integral = 0.5 * integral + added;
    // Two differences `δ → δ²` do not show convergence: an unresolved oscillation gives them too.
    if level >= 2 && difference <= level_target {
      break;
    }
  }
  (half * integral, half * difference)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_smooth_panel_meets_its_target() {
    let (got, difference) = integrate(&|x: f64| (-x).exp(), 0.0, 8.0, 1e-10);
    assert!((got - (1.0 - (-8.0_f64).exp())).abs() < 1e-12, "{got}");
    assert!(difference <= 0.25e-10, "{difference:e}");
  }

  /// A decaying oscillation whose differences fall as `δ → δ²` two levels before the rule resolves
  /// it: stopping on the square would leave the integral 1.5e-5 off.
  #[test]
  fn a_squared_level_difference_does_not_end_the_levels() {
    let (w, a, phase) = (0.454_f64, 0.2_f64, 1.0_f64);
    let primitive = |x: f64| {
      (-a * x).exp() * (w * (w * x + phase).sin() - a * (w * x + phase).cos()) / (a * a + w * w)
    };
    let f = |x: f64| (w * x + phase).cos() * (-a * x).exp();
    let (got, difference) = integrate(&f, 24.0, 56.0, 1e-10);
    let want = primitive(56.0) - primitive(24.0);
    assert!((got - want).abs() < 1e-13, "{got} vs {want}");
    assert!(difference <= 0.25e-10, "{difference:e}");
  }
}

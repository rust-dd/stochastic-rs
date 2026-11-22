//! The tanh-sinh rule on one panel, its error read from the last level difference.
//! Reference: Takahasi & Mori (1974), Publ. RIMS 9, 721–741, (3.3), DOI 10.2977/prims/1195192451
//! Ported from `quadrature` 0.1.2 and J. D. Cook's C++, both BSD: notices in `LICENSE-THIRD-PARTY`.

use std::f64::consts::PI;

use super::nodes::LEVELS;

/// `(∫_lo^hi f, |last level difference|, levels, ∫|f|)`, stopping at the first level from the
/// third whose difference is under a quarter of `target`; a non-finite sample counts as zero.
pub(super) fn integrate<F>(f: &F, lo: f64, hi: f64, target: f64) -> (f64, f64, usize, f64)
where
  F: Fn(f64) -> f64,
{
  let level_target = 0.25 * target / (0.5 * (hi - lo));
  // Two differences `δ → δ²` do not show convergence: an unresolved oscillation gives them too.
  sum(f, lo, hi, |level, difference| {
    level >= 2 && difference <= level_target
  })
}

/// `∫_lo^hi f` on the first `levels` levels, the rule [`integrate`] stopped at.
pub(super) fn integrate_levels<F>(f: &F, lo: f64, hi: f64, levels: usize) -> f64
where
  F: Fn(f64) -> f64,
{
  sum(f, lo, hi, |level, _| level + 1 >= levels).0
}

fn sum<F>(f: &F, lo: f64, hi: f64, done: impl Fn(usize, f64) -> bool) -> (f64, f64, usize, f64)
where
  F: Fn(f64) -> f64,
{
  let half = 0.5 * (hi - lo);
  let centre = 0.5 * (lo + hi);
  let sample = |x: f64| {
    let value = f(half * x + centre);
    if value.is_finite() { value } else { 0.0 }
  };

  let mut integral = PI * sample(0.0);
  let mut size = integral.abs();
  let mut difference = f64::MAX;
  let mut levels = 0;
  for (level, nodes) in LEVELS.iter().enumerate() {
    let (added, added_size) = nodes.iter().fold((0.0, 0.0), |(sum, size), &(weight, x)| {
      let (right, left) = (sample(x), sample(-x));
      (
        sum + weight * (right + left),
        size + weight * (right.abs() + left.abs()),
      )
    });
    difference = (0.5 * integral - added).abs();
    integral = 0.5 * integral + added;
    size = 0.5 * size + added_size;
    levels = level + 1;
    if done(level, difference) {
      break;
    }
  }
  (half * integral, half * difference, levels, half * size)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_smooth_panel_meets_its_target() {
    let (got, difference, _, size) = integrate(&|x: f64| (-x).exp(), 0.0, 8.0, 1e-10);
    assert!((size - got).abs() < 1e-12, "{size}");
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
    let (got, difference, levels, _) = integrate(&f, 24.0, 56.0, 1e-10);
    assert_eq!(integrate_levels(&f, 24.0, 56.0, levels), got);
    let want = primitive(56.0) - primitive(24.0);
    assert!((got - want).abs() < 1e-13, "{got} vs {want}");
    assert!(difference <= 0.25e-10, "{difference:e}");
  }
}

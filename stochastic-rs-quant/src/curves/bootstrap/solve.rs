//! The root of a quote's equation in the node it reads on the curve that holds it.

use stochastic_rs_distributions::brent;

use crate::traits::RealExt;

/// Root of `residual` (value, and slope where known) from `start` by Newton, else secant, steps
/// inside the bracket the signs show; Brent's `zero` once a step leaves `(0, ∞)` or 20 do not stop.
pub(in crate::curves) fn trial_node<T: RealExt>(
  start: T,
  mut residual: impl FnMut(T) -> (T, Option<T>),
) -> Option<T> {
  let tolerance = T::from_f64_fast(4.0) * T::machine_epsilon();
  let (mut x, (mut f, mut slope)) = (start, residual(start));
  let mut previous: Option<(T, T)> = None;
  let mut bracket: Option<[(T, T); 2]> = None;
  for _ in 0..20 {
    if f == T::zero() {
      return Some(x);
    }
    if !f.is_finite() {
      break;
    }
    bracket = match bracket {
      Some([lo, hi]) if (lo.1 > T::zero()) == (f > T::zero()) => Some([(x, f), hi]),
      Some([lo, _]) => Some([lo, (x, f)]),
      None => previous
        .filter(|p| (p.1 > T::zero()) != (f > T::zero()))
        .map(|p| if p.0 < x { [p, (x, f)] } else { [(x, f), p] }),
    };
    let (next, probe) = match (slope, previous) {
      (Some(d), _) => (x - f / d, false),
      (None, Some((xp, fp))) => (x - f * (x - xp) / (f - fp), false),
      (None, None) => (x * (T::one() + T::machine_epsilon().sqrt()), true),
    };
    let converged = |n: T| !probe && n.is_finite() && (n - x).abs() <= tolerance * n.abs();
    if converged(next) && next > T::zero() {
      return Some(next);
    }
    let next = match bracket {
      Some([lo, hi]) if !(next > lo.0.min(hi.0) && next < lo.0.max(hi.0)) => {
        T::from_f64_fast(0.5) * (lo.0 + hi.0)
      }
      _ => next,
    };
    if !(next.is_finite() && next > T::zero()) {
      break;
    }
    if converged(next) {
      return Some(next);
    }
    previous = Some((x, f));
    x = next;
    (f, slope) = residual(x);
  }
  let mut real = |node: f64| {
    let value = residual(T::from_f64_fast(node)).0;
    value.to_f64().unwrap_or(f64::NAN)
  };
  let found = match bracket {
    Some([lo, hi]) => {
      let (a, b) = (lo.0.to_f64(), hi.0.to_f64());
      a.zip(b)
        .and_then(|(a, b)| brent::zero(a, b, f64::MIN_POSITIVE, &mut real))
    }
    None => {
      let centre = start.to_f64().unwrap_or(f64::NAN);
      (1..=60).find_map(|k| {
        let width = 2.0_f64.powi(k);
        brent::zero(centre / width, centre * width, f64::MIN_POSITIVE, &mut real)
      })
    }
  };
  found.map(T::from_f64_fast)
}

#[cfg(test)]
mod tests {
  use std::cell::Cell;

  use super::*;

  /// `ln x + x = 3/2` at `1.2649597201255005`, by mpmath.
  const ROOT: f64 = 1.2649597201255005;

  fn counted(slope: bool) -> (Option<f64>, usize) {
    let calls = Cell::new(0);
    let root = trial_node(0.5, |x: f64| {
      calls.set(calls.get() + 1);
      (x.ln() + x - 1.5, slope.then(|| 1.0 / x + 1.0))
    });
    (root, calls.get())
  }

  #[test]
  fn newton_and_secant_steps_reach_the_root_from_afar() {
    for slope in [true, false] {
      let (root, calls) = counted(slope);
      let root = root.unwrap();
      assert!(
        (root - ROOT).abs() <= 4.0 * f64::EPSILON * ROOT,
        "{slope}: {root}"
      );
      assert!(calls <= if slope { 6 } else { 9 }, "{slope}: {calls} calls");
    }
  }

  /// `x^1.7 = 0.35` by secant steps: `f32` stops at its own precision in as few calls as `f64`.
  #[test]
  fn an_f32_root_stops_at_its_own_epsilon() {
    let root = 0.35_f64.powf(1.0 / 1.7);
    let calls = Cell::new(0);
    let got = trial_node(0.3_f32, |x: f32| {
      calls.set(calls.get() + 1);
      (x.powf(1.7) - 0.35, None)
    })
    .unwrap();
    let error = (f64::from(got) - root).abs() / root;
    assert!(error <= 4.0 * f64::from(f32::EPSILON), "{got}: {error:e}");
    assert!(calls.get() <= 9, "{} calls", calls.get());
  }

  /// Newton from 3 on `ln x` steps below 0; Brent's bracket around the start finds 1.
  #[test]
  fn a_step_out_of_the_positive_axis_falls_back_to_brent() {
    let root = trial_node(3.0, |x: f64| (x.ln(), Some(1.0 / x))).unwrap();
    assert!((root - 1.0).abs() <= 4.0 * f64::EPSILON, "{root}");
  }

  #[test]
  fn no_root_on_the_positive_axis_is_none() {
    assert_eq!(trial_node(1.0, |x: f64| (x * x + 1.0, Some(2.0 * x))), None);
    assert_eq!(trial_node(1.0, |x: f64| (f64::NAN * x, None)), None);
  }
}

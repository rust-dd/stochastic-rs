//! Quadrature for characteristic-function inversion over `[a, ∞)`: tanh-sinh panels, walked until
//! the integrand's envelope can no longer move the answer, since a fixed upper limit truncates it.

use std::cell::Cell;

mod nodes;
mod tanh_sinh;

/// Starting panel width. Wide enough that a smooth integrand clears it in one
/// tanh-sinh pass, narrow enough that the rule's fixed ~350-point budget still
/// resolves the phase `e^{-iu·ln(K/S)}` at ordinary moneyness.
const INITIAL_WIDTH: f64 = 8.0;
/// Floor on the panel width, so a pathological integrand cannot drive the walk
/// to a standstill.
const MIN_WIDTH: f64 = 1.0 / 64.0;
/// Ceiling on the panel width: the tanh-sinh point budget is fixed, so a wider panel is not
/// integrated at all and the `(b−a)/2` rescaling amplifies its noise.
const MAX_WIDTH: f64 = 64.0;
/// Backstop on the number of panels. The envelope test below terminates long
/// before this in every in-tree caller.
const MAX_PANELS: usize = 128;
/// Backstop on bisection depth, i.e. at most 64 sub-panels per panel.
const MAX_DEPTH: u32 = 6;

/// The pieces `(lo, hi, levels)` one walk integrated: a fixed rule, so integrands that share it
/// differ smoothly where a walk of each could stop at different panels.
#[derive(Clone, Debug, Default)]
pub(crate) struct Panels(Vec<(f64, f64, usize)>);

/// How an integral runs: adaptively, recording each walk's rule, or on recorded rules in order.
pub(crate) enum Walk<'a, P> {
  Adaptive,
  Record(&'a mut Vec<P>),
  Replay(std::slice::Iter<'a, P>),
}

impl<P> Walk<'_, P> {
  /// `adaptive`, given a rule to fill when recording, or `replay` on the next recorded rule.
  pub(crate) fn run<R>(
    &mut self,
    adaptive: impl FnOnce(Option<&mut P>) -> R,
    replay: impl FnOnce(&P) -> R,
  ) -> R
  where
    P: Default,
  {
    match self {
      Walk::Adaptive => adaptive(None),
      Walk::Record(rules) => {
        let mut rule = P::default();
        let value = adaptive(Some(&mut rule));
        rules.push(rule);
        value
      }
      Walk::Replay(rules) => {
        let rule = rules.next();
        // A record shorter than its replay is a caller's slip; a release build walks instead.
        debug_assert!(rule.is_some(), "a replay outran its recorded rules");
        rule.map_or_else(|| adaptive(None), replay)
      }
    }
  }
}

/// One panel, bisected until the rule's last level difference meets `target`, which halves with
/// each bisection so the sub-panels' budget still sums to it; also the bisection depth used.
fn refine<F>(
  f: &F,
  lo: f64,
  hi: f64,
  target: f64,
  depth: u32,
  leaves: &mut Option<&mut Panels>,
) -> (f64, u32)
where
  F: Fn(f64) -> f64,
{
  let (integral, difference, levels) = tanh_sinh::integrate(f, lo, hi, target);
  if difference <= target || depth == 0 {
    if let Some(leaves) = leaves {
      leaves.0.push((lo, hi, levels));
    }
    return (integral, 0);
  }
  let mid = 0.5 * (lo + hi);
  let (left, dl) = refine(f, lo, mid, 0.5 * target, depth - 1, leaves);
  let (right, dr) = refine(f, mid, hi, 0.5 * target, depth - 1, leaves);
  (left + right, 1 + dl.max(dr))
}

/// `∫_a^∞ f` to relative `tol`, until the decay of the evaluated `max |f|` bounds the tail; a NaN
/// sample makes it NaN, a `±∞` one counts as zero (Lévy losses overflow on unprojected iterates).
pub(crate) fn integrate_to_convergence<F>(f: F, a: f64, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  walk(f, a, tol, None)
}

/// [`integrate_to_convergence`] run as `mode` says: recorded, or on a rule recorded before.
pub(crate) fn integrate_walk<F>(f: F, a: f64, tol: f64, mode: &mut Walk<'_, Panels>) -> f64
where
  F: Fn(f64) -> f64,
{
  mode.run(
    |leaves| walk(&f, a, tol, leaves),
    |panels| integrate_on(&f, panels),
  )
}

fn walk<F>(f: F, a: f64, tol: f64, mut leaves: Option<&mut Panels>) -> f64
where
  F: Fn(f64) -> f64,
{
  let poisoned = Cell::new(false);
  let peak = Cell::new(0.0_f64);
  let watched = |u: f64| -> f64 {
    let v = f(u);
    if v.is_nan() {
      poisoned.set(true);
    }
    let magnitude = v.abs();
    if magnitude.is_finite() && magnitude > peak.get() {
      peak.set(magnitude);
    }
    v
  };

  let mut lo = a;
  let mut width = INITIAL_WIDTH;
  let mut total = 0.0_f64;
  let mut scale = 0.0_f64;
  let mut prev_env = f64::INFINITY;

  for panel_index in 0..MAX_PANELS {
    peak.set(0.0);
    let (panel, depth_used) = refine(&watched, lo, lo + width, tol, MAX_DEPTH, &mut leaves);
    if poisoned.get() {
      return f64::NAN;
    }
    total += panel;
    lo += width;

    let env = peak.get();
    scale = scale.max(env);

    if panel_index > 0 && env <= tol * scale {
      let decay = (prev_env / env).ln() / width;
      if decay > 0.0 && env / decay <= tol * total.abs().max(1.0) {
        break;
      }
    }
    prev_env = env;

    width = if depth_used == 0 {
      (width * 2.0).min(MAX_WIDTH)
    } else {
      (width / f64::from(1u32 << (depth_used - 1))).max(MIN_WIDTH)
    };
  }

  total
}

/// `∫ f` on recorded panels, each at the levels its walk took; NaN as the walk would give it.
fn integrate_on<F>(f: &F, panels: &Panels) -> f64
where
  F: Fn(f64) -> f64,
{
  let poisoned = Cell::new(false);
  let watched = |u: f64| -> f64 {
    let v = f(u);
    poisoned.set(poisoned.get() || v.is_nan());
    v
  };
  let total = panels.0.iter().fold(0.0, |total, &(lo, hi, levels)| {
    total + tanh_sinh::integrate_levels(&watched, lo, hi, levels)
  });
  if poisoned.get() { f64::NAN } else { total }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The rule counts a non-finite sample as zero, so the walk is what reports a NaN integrand.
  #[test]
  fn a_nan_integrand_does_not_integrate_to_zero() {
    let v = integrate_to_convergence(|_| f64::NAN, 1e-8, 1e-8);
    assert!(v.is_nan(), "a NaN integrand must exit as NaN, got {v}");
  }

  /// A `NaN` over part of the range poisons the whole integral: once any
  /// piece of the integrand is undefined there is no integral to return, and
  /// the swallowed version silently reports the finite piece alone.
  #[test]
  fn a_partially_nan_integrand_is_poisoned_too() {
    let v = integrate_to_convergence(
      |u: f64| if u > 10.0 { f64::NAN } else { (-u).exp() },
      0.0,
      1e-8,
    );
    assert!(
      v.is_nan(),
      "a partially NaN integrand must exit as NaN, got {v}"
    );
  }

  /// One panel of a walk, `[24, 56]`: the square of its level difference would pass it 1.5e-5 off.
  #[test]
  fn a_panel_is_not_accepted_on_a_squared_level_difference() {
    let (w, a, phase) = (0.454_f64, 0.2_f64, 1.0_f64);
    let primitive = |x: f64| {
      (-a * x).exp() * (w * (w * x + phase).sin() - a * (w * x + phase).cos()) / (a * a + w * w)
    };
    let f = |x: f64| (w * x + phase).cos() * (-a * x).exp();
    let (got, _) = refine(&f, 24.0, 56.0, 1e-10, MAX_DEPTH, &mut None);
    let want = primitive(56.0) - primitive(24.0);
    assert!((got - want).abs() < 1e-10, "{got} vs {want}");
  }

  /// The poison check must leave an integrand that was never poisoned alone.
  /// $\int_0^\infty e^{-x}\,dx = 1$.
  #[test]
  fn a_finite_integrand_is_unchanged_by_the_poison_check() {
    let v = integrate_to_convergence(|x: f64| (-x).exp(), 0.0, 1e-10);
    assert!((v - 1.0).abs() < 1e-9, "expected 1, got {v}");
  }

  /// An `∞` sample counts as zero and does not end the walk; `∞` on `[0, 1)` leaves `∫_1^∞ e^{-x}`.
  #[test]
  fn an_infinite_sample_counts_as_zero() {
    let v = integrate_to_convergence(
      |u: f64| if u < 1.0 { f64::INFINITY } else { (-u).exp() },
      0.0,
      1e-10,
    );
    let expected = (-1.0_f64).exp();
    assert!((v - expected).abs() < 1e-9, "expected {expected}, got {v}");
  }

  /// The same over a slow tail, which an infinite envelope would cut short: `∫_1^∞ e^{-x/20} dx`.
  #[test]
  fn an_infinite_sample_does_not_truncate_a_slow_tail() {
    let v = integrate_to_convergence(
      |u: f64| {
        if u < 1.0 {
          f64::INFINITY
        } else {
          (-u / 20.0).exp()
        }
      },
      0.0,
      1e-10,
    );
    let expected = 20.0 * (-0.05_f64).exp();
    assert!(
      (v - expected).abs() < 1e-9 * expected,
      "expected {expected}, got {v}"
    );
  }

  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "a replay outran its recorded rules")]
  fn a_replay_past_its_record_panics_where_debug_assertions_run() {
    integrate_walk(|u: f64| (-u).exp(), 0.0, 1e-8, &mut Walk::Replay([].iter()));
  }

  /// Recorded on one integrand, the rule integrates a shifted one as a fixed rule: the same value
  /// at the point, and a difference across the walk's stop that moves only with the integrand.
  #[test]
  fn a_recorded_rule_replays_the_walk_and_holds_across_its_stop() {
    let f = |c: f64| move |u: f64| (-c * u).exp() * (u * 1.3).cos() / (1.0 + u * u);
    let mut rules = Vec::new();
    let walked = integrate_walk(f(0.05), 0.0, 1e-8, &mut Walk::Record(&mut rules));
    assert_eq!(walked, integrate_to_convergence(f(0.05), 0.0, 1e-8));
    let replayed = integrate_walk(f(0.05), 0.0, 1e-8, &mut Walk::Replay(rules.iter()));
    assert!((replayed - walked).abs() < 1e-15, "{replayed} vs {walked}");
    let h = 1e-6;
    let on_rule = |c: f64| integrate_walk(f(c), 0.0, 1e-8, &mut Walk::Replay(rules.iter()));
    let slope = (on_rule(0.05 + h) - on_rule(0.05 - h)) / (2.0 * h);
    let wide = (on_rule(0.05 + 1e3 * h) - on_rule(0.05 - 1e3 * h)) / (2e3 * h);
    assert!(
      (slope - wide).abs() < 1e-6 * wide.abs(),
      "{slope} vs {wide}"
    );
  }
}

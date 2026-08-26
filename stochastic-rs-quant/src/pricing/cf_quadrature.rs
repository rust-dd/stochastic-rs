//! Convergence-controlled quadrature for characteristic-function inversion.
//!
//! The Heston, Gil-Pelaez, Lewis and Carr-Madan pricers all invert a
//! characteristic function by integrating an oscillatory-but-decaying integrand
//! over `[a, ∞)`. The integrand's envelope only becomes negligible once `φ` is
//! past its decay length, which grows like `1/√(v·τ)` as the maturity `τ` or
//! the variance `v` shrink. A hardcoded finite upper limit therefore truncates
//! a non-negligible tail for short-dated or low-variance options, which
//! under-prices them by 15-35% and can even return arbitrage-violating negative
//! call prices.
//!
//! [`integrate_to_convergence`] replaces the fixed bound: it accumulates
//! tanh-sinh panels of geometrically growing width and stops once the tail
//! contribution is negligible, so the effective upper limit adapts to the
//! actual decay length for any `(τ, v, moneyness)`.

use std::cell::Cell;

use quadrature::double_exponential;

/// Integrate `f` over `[a, ∞)` to a relative tolerance `tol`.
///
/// Successive tanh-sinh panels of geometrically growing width are summed until
/// two consecutive panels each add less than `tol` relative to the running
/// integral. `tol` is also the per-panel tanh-sinh target. Requiring two
/// negligible panels (not one) guards against a panel that integrates to near
/// zero by oscillatory cancellation while the envelope is still significant.
///
/// Returns [`f64::NAN`] if the integrand is `NaN` anywhere the rule evaluates
/// it. That check has to live here because the third-party
/// `double_exponential::integrate` rewrites every non-finite sample to `0.0`
/// before the rule sees it, which silently turns an undefined integrand into
/// a well-scaled number: a wholly-`NaN` integrand integrated to exactly `0.0`,
/// so `GilPelaezPricer` returned `2.438528774964297` and `LewisPricer` the
/// spot for a characteristic function that had blown up. That is the
/// plausible-looking sentinel the [failure
/// convention](crate::traits::ModelPricer#how-pricing-fails) rules out, and
/// this wrapper is the only place in the path the crate owns.
///
/// `±∞` keeps the third-party behaviour deliberately. An overflowing
/// integrand is a different case from an undefined one, and the crate's own
/// Lévy loss integrand reaches `∞` transiently on unprojected calibration
/// iterates, where poisoning the loss would abort a run that currently
/// recovers.
pub(crate) fn integrate_to_convergence<F>(f: F, a: f64, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  const INITIAL_WIDTH: f64 = 50.0;
  const GROWTH: f64 = 2.0;
  const MAX_PANELS: usize = 40;

  let poisoned = Cell::new(false);
  let watched = |u: f64| -> f64 {
    let v = f(u);
    if v.is_nan() {
      poisoned.set(true);
    }
    v
  };

  let mut lo = a;
  let mut width = INITIAL_WIDTH;
  let mut total = 0.0_f64;
  let mut negligible_streak = 0u32;

  for _ in 0..MAX_PANELS {
    let panel = double_exponential::integrate(&watched, lo, lo + width, tol).integral;
    if poisoned.get() {
      return f64::NAN;
    }
    total += panel;

    if panel.abs() <= tol * total.abs().max(1.0) {
      negligible_streak += 1;
      if negligible_streak >= 2 {
        break;
      }
    } else {
      negligible_streak = 0;
    }

    lo += width;
    width *= GROWTH;
  }

  total
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The third-party `double_exponential::integrate` rewrites every
  /// non-finite sample to `0.0` before the rule ever sees it, so an
  /// integrand that is `NaN` everywhere integrates to exactly `0.0` — a
  /// plausible number carrying no trace of the undefined input that produced
  /// it. That is the [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails)'s fourth
  /// option, and this wrapper is the only place the crate controls.
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
    assert!(v.is_nan(), "a partially NaN integrand must exit as NaN, got {v}");
  }

  /// The poison check must leave an integrand that was never poisoned alone.
  /// $\int_0^\infty e^{-x}\,dx = 1$.
  #[test]
  fn a_finite_integrand_is_unchanged_by_the_poison_check() {
    let v = integrate_to_convergence(|x: f64| (-x).exp(), 0.0, 1e-10);
    assert!((v - 1.0).abs() < 1e-9, "expected 1, got {v}");
  }
}

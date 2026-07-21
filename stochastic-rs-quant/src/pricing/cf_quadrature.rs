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

use quadrature::double_exponential;

/// Integrate `f` over `[a, ∞)` to a relative tolerance `tol`.
///
/// Successive tanh-sinh panels of geometrically growing width are summed until
/// two consecutive panels each add less than `tol` relative to the running
/// integral. `tol` is also the per-panel tanh-sinh target. Requiring two
/// negligible panels (not one) guards against a panel that integrates to near
/// zero by oscillatory cancellation while the envelope is still significant.
pub(crate) fn integrate_to_convergence<F>(f: F, a: f64, tol: f64) -> f64
where
  F: Fn(f64) -> f64,
{
  const INITIAL_WIDTH: f64 = 50.0;
  const GROWTH: f64 = 2.0;
  const MAX_PANELS: usize = 40;

  let mut lo = a;
  let mut width = INITIAL_WIDTH;
  let mut total = 0.0_f64;
  let mut negligible_streak = 0u32;

  for _ in 0..MAX_PANELS {
    let panel = double_exponential::integrate(&f, lo, lo + width, tol).integral;
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

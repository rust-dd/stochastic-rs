//! Gauss–Seidel sweeps for an interpolation whose later pillars reshape its earlier intervals: each
//! quote re-solved in its slot of the whole curve, in pillar order, until no node moves.
//!
//! Reference: QuantLib, `IterativeBootstrap::calculate`, ql/termstructures/iterativebootstrap.hpp
//! Reference: QuantLib, `accuracy_` = 1e-12, ql/termstructures/yield/piecewiseyieldcurve.hpp

use super::Place;
use super::quote_node;
use crate::QuantError;
use crate::curves::discount_curve::DiscountCurve;
use crate::curves::types::BootstrapInstrument;
use crate::curves::types::CurvePoint;
use crate::curves::types::InterpolationMethod;
use crate::traits::RealExt;

/// QuantLib's `maxIterations` for a discount curve's bootstrap.
const MAX_SWEEPS: usize = 100;

/// Sweeps until no node moves by more than `2^10·ε` of itself, inside QuantLib's `1e-12` in `f64`
/// and ten times a re-solve's rounding jitter (to `1e2·ε`); `Optimizer` after `MAX_SWEEPS`.
pub(super) fn reprice_together<T: RealExt>(
  points: &mut [CurvePoint<T>],
  quotes: &[(&BootstrapInstrument<T>, usize)],
  discount: Option<&DiscountCurve<T>>,
  method: InterpolationMethod,
) -> Result<(), QuantError> {
  let tolerance = T::from_f64_fast(1024.0) * T::machine_epsilon();
  let mut largest = T::zero();
  for _ in 0..MAX_SWEEPS {
    largest = T::zero();
    for (slot, &(instrument, quote)) in quotes.iter().enumerate() {
      let node = quote_node(Place::At(points, slot), instrument, discount, method, quote)?;
      largest = largest.max_or_nan((node - points[slot].discount_factor).abs() / node);
      points[slot].discount_factor = node;
    }
    if largest <= tolerance {
      return Ok(());
    }
  }
  Err(QuantError::Optimizer {
    what: "bootstrap",
    reason: format!(
      "a node still moves by {largest:?} of itself after {MAX_SWEEPS} sweeps over the curve"
    ),
  })
}

#[cfg(test)]
mod tests;

//! Gauss–Seidel sweeps for an interpolation whose later pillars reshape its earlier intervals: each
//! quote re-solved in its slot of the whole curve, in pillar order, until no node moves.
//!
//! Reference: QuantLib, `IterativeBootstrap::calculate`, ql/termstructures/iterativebootstrap.hpp

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

/// Sweeps until no node moves by more than `4ε` of itself; `Optimizer` while one still does after
/// `MAX_SWEEPS`.
pub(super) fn reprice_together<T: RealExt>(
  points: &mut [CurvePoint<T>],
  quotes: &[(&BootstrapInstrument<T>, usize)],
  discount: Option<&DiscountCurve<T>>,
  method: InterpolationMethod,
) -> Result<(), QuantError> {
  let tolerance = T::from_f64_fast(4.0) * T::epsilon();
  for _ in 0..MAX_SWEEPS {
    let mut moved = false;
    for (slot, &(instrument, quote)) in quotes.iter().enumerate() {
      let node = quote_node(Place::At(points, slot), instrument, discount, method, quote)?;
      moved |= (node - points[slot].discount_factor).abs() > tolerance * node;
      points[slot].discount_factor = node;
    }
    if !moved {
      return Ok(());
    }
  }
  Err(QuantError::Optimizer {
    what: "bootstrap",
    reason: format!("a node still moves after {MAX_SWEEPS} sweeps of the quotes over the curve"),
  })
}

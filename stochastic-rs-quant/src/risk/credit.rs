//! Credit-aware risk metrics: a survival curve from [`crate::credit`] beside the portfolio
//! loss measures of [`crate::risk`].
//!
//! ```
//! use stochastic_rs_quant::credit::survival_curve::{
//!     HazardInterpolation, SurvivalCurve, SurvivalPoint,
//! };
//! use stochastic_rs_quant::risk::credit::expected_credit_loss;
//!
//! // In practice these nodes are bootstrapped from CDS spreads; two
//! // (time, survival) points keep this example self-contained.
//! let curve = SurvivalCurve::new(
//!     vec![
//!         SurvivalPoint { time: 1.0, survival_probability: 0.98 },
//!         SurvivalPoint { time: 5.0, survival_probability: 0.90 },
//!     ],
//!     HazardInterpolation::PiecewiseConstantHazard,
//! );
//! let lgd = 0.6;
//! let exposure = 1_000_000.0;
//! let ecl = expected_credit_loss(&curve, 1.0, lgd, exposure);
//! assert!(ecl > 0.0);
//! ```

use crate::credit::survival_curve::SurvivalCurve;
use crate::traits::RealExt;

/// Expected credit loss $(1 - Q(t)) \cdot \mathrm{LGD} \cdot \mathrm{exposure}$ over `t` years,
/// with $Q$ the survival probability and LGD = 1 − recovery.
pub fn expected_credit_loss<T: RealExt>(curve: &SurvivalCurve<T>, t: T, lgd: T, exposure: T) -> T {
  let q = curve.survival_probability(t);
  (T::one() - q) * lgd * exposure
}

/// Probability that default occurs before `t`, i.e. $1 - Q(t)$.
pub fn probability_of_default_before<T: RealExt>(curve: &SurvivalCurve<T>, t: T) -> T {
  T::one() - curve.survival_probability(t)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::credit::survival_curve::HazardInterpolation;

  #[test]
  fn ecl_grows_with_horizon() {
    use ndarray::Array1;
    let times = Array1::from(vec![1.0_f64, 5.0]);
    let hazards = Array1::from(vec![0.01_f64, 0.01]);
    let curve = SurvivalCurve::<f64>::from_hazard_rates(
      &times,
      &hazards,
      HazardInterpolation::PiecewiseConstantHazard,
    );
    let ecl_short = expected_credit_loss(&curve, 1.0, 0.6, 1_000.0);
    let ecl_long = expected_credit_loss(&curve, 5.0, 0.6, 1_000.0);
    assert!(ecl_long > ecl_short);
  }
}

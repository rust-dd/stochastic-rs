//! # Credit-aware risk metrics
//!
//! Bridges [`crate::credit`] with [`crate::risk`]: combines a survival curve
//! with portfolio loss simulation so default probability shows up alongside
//! market VaR.
//!
//! # Example
//!
//! ```ignore
//! use stochastic_rs_quant::credit::survival_curve::SurvivalCurve;
//! use stochastic_rs_quant::risk::credit::expected_credit_loss;
//!
//! let curve: SurvivalCurve<f64> = /* bootstrap from CDS */;
//! let lgd = 0.6;
//! let exposure = 1_000_000.0;
//! let ecl = expected_credit_loss(&curve, 1.0, lgd, exposure);
//! ```

use crate::credit::survival_curve::SurvivalCurve;
use crate::traits::FloatExt;

/// Expected credit loss over the horizon `t` years.
///
/// $\mathrm{ECL} = (1 - Q(t)) \cdot \mathrm{LGD} \cdot \mathrm{exposure}$,
/// where $Q$ is the survival probability and LGD = 1 − recovery.
pub fn expected_credit_loss<T: FloatExt>(
  curve: &SurvivalCurve<T>,
  t: T,
  lgd: T,
  exposure: T,
) -> T {
  let q = curve.survival_probability(t);
  (T::one() - q) * lgd * exposure
}

/// Probability that default occurs before `t`, i.e. $1 - Q(t)$.
pub fn probability_of_default_before<T: FloatExt>(curve: &SurvivalCurve<T>, t: T) -> T {
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
    let curve =
      SurvivalCurve::<f64>::from_hazard_rates(&times, &hazards, HazardInterpolation::PiecewiseConstantHazard);
    let ecl_short = expected_credit_loss(&curve, 1.0, 0.6, 1_000.0);
    let ecl_long = expected_credit_loss(&curve, 5.0, 0.6, 1_000.0);
    assert!(ecl_long > ecl_short);
  }
}

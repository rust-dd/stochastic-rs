use ndarray::Array1;

use crate::traits::FloatExt;

/// `1 − Σα − Σthreshold/2 − Σβ`, the unconditional variance's denominator; panics unless positive.
/// One check for `new` and the samplers, which a field written after `new` reaches.
#[track_caller]
pub(super) fn variance_denominator<T: FloatExt>(
  alpha: &Array1<T>,
  threshold: &Array1<T>,
  beta: &Array1<T>,
  threshold_name: &str,
) -> T {
  let sum_alpha: T = alpha.iter().cloned().sum();
  let sum_threshold_half = threshold.iter().cloned().sum::<T>() * T::from_f64_fast(0.5);
  let sum_beta: T = beta.iter().cloned().sum();
  let denom = T::one() - sum_alpha - sum_threshold_half - sum_beta;
  assert!(
    denom > T::zero(),
    "alpha must satisfy `1 - alpha.sum() - 0.5 * {threshold_name}.sum() - beta.sum() > 0`, got alpha = {:?}, {threshold_name} = {:?}, beta = {:?}",
    alpha.to_vec(),
    threshold.to_vec(),
    beta.to_vec()
  );
  denom
}

use ndarray::Array1;

use crate::precondition::assert_entries;
use crate::traits::FloatExt;

/// The positivity conditions `α ≥ 0`, `α + threshold ≥ 0`, `β ≥ 0`, then the unconditional
/// variance's denominator `1 − Σα − Σthreshold/2 − Σβ > 0`; one check for `new` and the samplers.
#[track_caller]
pub(super) fn variance_denominator<T: FloatExt>(
  alpha: &Array1<T>,
  threshold: &Array1<T>,
  beta: &Array1<T>,
  threshold_name: &str,
) -> T {
  assert_entries("alpha", "alpha[i] >= T::zero()", alpha, |v| v >= T::zero());
  let fall = alpha
    .iter()
    .zip(threshold)
    .position(|(&a, &g)| a + g < T::zero());
  if let Some(i) = fall {
    panic!(
      "{threshold_name}[{i}] must satisfy `alpha[i] + {threshold_name}[i] >= T::zero()`, got alpha[{i}] = {:?}, {threshold_name}[{i}] = {:?}",
      alpha[i], threshold[i]
    );
  }
  assert_entries("beta", "beta[i] >= T::zero()", beta, |v| v >= T::zero());
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

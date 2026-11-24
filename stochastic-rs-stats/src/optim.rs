//! Optimization the estimator modules share, and `stochastic-rs-quant` with them: the stop
//! classes and their words, the first-order check, the L-BFGS searches and the simplex.

use crate::traits::RealExt;

pub(crate) mod certificate;
mod curvature;
mod lbfgs;
mod legacy_simplex;
pub(crate) mod search;
pub mod simplex;
mod stop;

pub(crate) use lbfgs::lbfgsb;
pub(crate) use lbfgs::more_thuente;
pub(crate) use lbfgs::polish;
pub(crate) use legacy_simplex::nelder_mead;
pub(crate) use legacy_simplex::nelder_mead_vec;
pub use stop::SolverStop;

/// The start with the lowest objective; a NaN objective loses to every number.
pub(crate) fn best_start<P: AsRef<[f64]>>(
  starts: impl IntoIterator<Item = P>,
  objective: impl Fn(&[f64]) -> f64,
) -> Option<P> {
  starts
    .into_iter()
    .map(|start| (objective(start.as_ref()), start))
    .min_by(|a, b| a.0.nan_last(&b.0))
    .map(|(_, start)| start)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Arithmetic leaves a NaN's sign unspecified (x86 sets it), so both signs must lose.
  #[test]
  fn a_nan_objective_never_wins_the_start_search() {
    for nan in [f64::NAN, -f64::NAN] {
      let objective = |p: &[f64]| if p[0] == 1.0 { 3.0 } else { nan };
      assert_eq!(best_start([[0.0], [1.0], [2.0]], objective), Some([1.0]));
      assert_eq!(
        nan.nan_last(&f64::NEG_INFINITY),
        std::cmp::Ordering::Greater
      );
    }
  }
}

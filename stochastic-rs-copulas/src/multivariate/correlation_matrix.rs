//! A validated correlation matrix and the factors the elliptical copulas evaluate through.
//! Reference: Higham, N.J., Strabić, N., Šego, V. (2016), "Restoring definiteness via shrinking, with an application to correlation matrices with a fixed block", SIAM Review 58(2), 245-263, eq. (2.1), §2, Algorithm 3.1 and §5.3 eqs. (5.6), (5.8), DOI 10.1137/140996112.

use ndarray::Array2;

use super::linalg::is_spd;
use super::linalg::spd_factors;
use crate::error::CopulaError;

const SHRINK_TOLERANCE: f64 = 1e-6;
const EIGENVALUE_FLOOR: f64 = 1e-3;

/// Square with `dim >= 2`, unit diagonal, symmetric and finite to `1e-12`, positive definite; it carries its lower
/// Cholesky factor, inverse and log-determinant.
#[derive(Debug, Clone)]
pub(crate) struct CorrelationMatrix {
  corr: Array2<f64>,
  chol_lower: Array2<f64>,
  inverse: Array2<f64>,
  log_det: f64,
}

impl CorrelationMatrix {
  /// `InvalidStructure` naming the first property `corr` fails, checked in the order of the type's doc.
  pub(crate) fn try_new(corr: Array2<f64>) -> Result<Self, CopulaError> {
    let d = corr.nrows();
    if corr.ncols() != d {
      return Err(CopulaError::InvalidStructure(format!(
        "corr must satisfy `corr.nrows() == corr.ncols()` (square), got corr.shape() = {:?}",
        corr.shape()
      )));
    }
    if d < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "corr must satisfy `corr.nrows() >= 2` (dim >= 2), got corr.nrows() = {d}"
      )));
    }
    for i in 0..d {
      let diag = corr[[i, i]];
      if !diag.is_finite() || (diag - 1.0).abs() > 1e-12 {
        return Err(CopulaError::InvalidStructure(format!(
          "corr must satisfy `|corr[[i, i]] - 1.0| <= 1e-12` (unit diagonal), got corr[[{i}, {i}]] = {diag:?}"
        )));
      }
    }
    for i in 1..d {
      for j in 0..i {
        let (lower, upper) = (corr[[i, j]], corr[[j, i]]);
        if !(lower.is_finite() && upper.is_finite()) || (lower - upper).abs() > 1e-12 {
          return Err(CopulaError::InvalidStructure(format!(
            "corr must satisfy `|corr[[i, j]] - corr[[j, i]]| <= 1e-12` (symmetric and finite), got corr[[{i}, {j}]] = {lower:?} and corr[[{j}, {i}]] = {upper:?}"
          )));
        }
      }
    }
    let (chol_lower, inverse) = spd_factors(corr.view()).map_err(|index| {
      CopulaError::InvalidStructure(format!(
        "corr must satisfy `pivot[i] > 0.0` (positive definite), got pivot[{index}] <= 0.0"
      ))
    })?;
    let log_det = 2.0 * (0..d).map(|i| chol_lower[[i, i]].ln()).sum::<f64>();
    Ok(Self {
      corr,
      chol_lower,
      inverse,
      log_det,
    })
  }

  pub(crate) fn dim(&self) -> usize {
    self.corr.nrows()
  }

  pub(crate) fn corr(&self) -> &Array2<f64> {
    &self.corr
  }

  pub(crate) fn chol_lower(&self) -> &Array2<f64> {
    &self.chol_lower
  }

  pub(crate) fn inverse(&self) -> &Array2<f64> {
    &self.inverse
  }

  pub(crate) fn log_det(&self) -> f64 {
    self.log_det
  }
}

/// The smallest shrink toward the identity, `(1 − λ) R + λ I`, that leaves every eigenvalue at least `1e-3`: `λ` is
/// bisected to `1e-6` with a Cholesky test of `S(λ) − 1e-3 I` (Algorithm 3.1, §5.3); the unit diagonal survives.
pub(crate) fn shrink_to_spd(corr: Array2<f64>) -> Array2<f64> {
  let clears_floor = |matrix: &Array2<f64>| {
    let mut lowered = matrix.clone();
    lowered.diag_mut().mapv_inplace(|d| d - EIGENVALUE_FLOOR);
    is_spd(lowered.view())
  };
  if clears_floor(&corr) {
    return corr;
  }
  let identity = Array2::<f64>::eye(corr.nrows());
  let shrink = |lambda: f64| {
    let mut shrunk = &corr * (1.0 - lambda) + &identity * lambda;
    shrunk.diag_mut().fill(1.0);
    shrunk
  };
  let (mut left, mut right) = (0.0, 1.0);
  while right - left > SHRINK_TOLERANCE {
    let middle = 0.5 * (left + right);
    if clears_floor(&shrink(middle)) {
      right = middle;
    } else {
      left = middle;
    }
  }
  shrink(right)
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use crate::multivariate::linalg::condition_number;
  use crate::multivariate::linalg::smallest_eigenvalue;

  /// The documented bisection tolerance and eigenvalue floor, pinned here independently of the code's constants.
  const TOLERANCE: f64 = 1e-6;
  const FLOOR: f64 = 1e-3;

  #[test]
  fn a_correlation_matrix_is_square_symmetric_unit_diagonal_and_positive_definite() {
    let good = CorrelationMatrix::try_new(array![[1.0, 0.5], [0.5, 1.0]]).unwrap();
    assert_eq!(good.dim(), 2);
    assert!((good.log_det() - (0.75_f64).ln()).abs() < 1e-14);
    assert!((good.chol_lower()[[1, 1]] - 0.75_f64.sqrt()).abs() < 1e-14);
    assert!((good.inverse()[[0, 1]] + 0.5 / 0.75).abs() < 1e-14);
    for (bad, what) in [
      (array![[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]], "square"),
      (array![[1.0, 0.5], [0.4, 1.0]], "symmetric"),
      (array![[1.1, 0.5], [0.5, 1.0]], "unit diagonal"),
      (array![[f64::NAN, 0.5], [0.5, 1.0]], "unit diagonal"),
      (array![[1.0, f64::NAN], [f64::NAN, 1.0]], "symmetric"),
      (
        array![[1.0, f64::INFINITY], [f64::INFINITY, 1.0]],
        "symmetric",
      ),
      (
        array![[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]],
        "positive definite",
      ),
      (array![[1.0, 1.0], [1.0, 1.0]], "positive definite"),
      (array![[1.0]], "dim >= 2"),
    ] {
      match CorrelationMatrix::try_new(bad) {
        Err(CopulaError::InvalidStructure(why)) => assert!(why.contains(what), "{why}"),
        other => panic!("expected InvalidStructure about {what}, got {other:?}"),
      }
    }
  }

  #[test]
  fn a_rejected_matrix_names_the_argument_the_predicate_and_the_value() {
    let message = |corr| match CorrelationMatrix::try_new(corr) {
      Err(CopulaError::InvalidStructure(why)) => why,
      other => panic!("expected InvalidStructure, got {other:?}"),
    };
    assert_eq!(
      message(array![[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]]),
      "corr must satisfy `corr.nrows() == corr.ncols()` (square), got corr.shape() = [2, 3]"
    );
    assert_eq!(
      message(array![[1.0]]),
      "corr must satisfy `corr.nrows() >= 2` (dim >= 2), got corr.nrows() = 1"
    );
    assert_eq!(
      message(array![[1.0, 0.5], [0.5, 1.1]]),
      "corr must satisfy `|corr[[i, i]] - 1.0| <= 1e-12` (unit diagonal), got corr[[1, 1]] = 1.1"
    );
    assert_eq!(
      message(array![[1.0, 0.5], [0.4, 1.0]]),
      "corr must satisfy `|corr[[i, j]] - corr[[j, i]]| <= 1e-12` (symmetric and finite), got corr[[1, 0]] = 0.4 and corr[[0, 1]] = 0.5"
    );
    assert_eq!(
      message(array![[1.0, 1.0], [1.0, 1.0]]),
      "corr must satisfy `pivot[i] > 0.0` (positive definite), got pivot[1] <= 0.0"
    );
    assert_eq!(
      message(array![[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]),
      "corr must satisfy `pivot[i] > 0.0` (positive definite), got pivot[2] <= 0.0"
    );
  }

  #[test]
  fn a_symmetry_gap_within_the_tolerance_is_accepted() {
    assert!(CorrelationMatrix::try_new(array![[1.0, 0.5], [0.5 + 1e-13, 1.0]]).is_ok());
    assert!(CorrelationMatrix::try_new(array![[1.0 + 1e-13, 0.5], [0.5, 1.0]]).is_ok());
  }

  #[test]
  fn shrinking_keeps_the_unit_diagonal_and_reaches_positive_definiteness() {
    let indefinite = array![[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]];
    let shrunk = shrink_to_spd(indefinite);
    assert!(CorrelationMatrix::try_new(shrunk.clone()).is_ok());
    assert!(shrunk.diag().iter().all(|d| (d - 1.0).abs() < 1e-15));
  }

  /// The shrink read off the first off-diagonal of `raw` against its analytic minimum `(FLOOR - mu) / (1 - mu)` for a
  /// smallest eigenvalue `mu`, and the result's smallest eigenvalue: in `[FLOOR, FLOOR + TOLERANCE (1 - mu)]`.
  fn assert_floored(raw: &Array2<f64>, shrunk: &Array2<f64>, mu: f64) {
    let lambda = 1.0 - shrunk[[0, 1]] / raw[[0, 1]];
    let want = (FLOOR - mu) / (1.0 - mu);
    assert!(
      (want - 1e-12..=want + TOLERANCE + 1e-12).contains(&lambda),
      "{lambda} vs {want}"
    );
    let least = smallest_eigenvalue(shrunk.view());
    assert!(
      (FLOOR - 1e-12..=FLOOR + TOLERANCE * (1.0 - mu) + 1e-12).contains(&least),
      "{least}"
    );
  }

  #[test]
  fn shrinking_takes_the_smallest_lambda_to_the_bisection_tolerance() {
    let a = -0.5025;
    let equi = array![[1.0, a, a], [a, 1.0, a], [a, a, 1.0]];
    let mu = smallest_eigenvalue(equi.view());
    assert!((mu - (1.0 + 2.0 * a)).abs() < 1e-12);
    let shrunk = shrink_to_spd(equi.clone());
    assert_floored(&equi, &shrunk, mu);
    let lambda = 1.0 - shrunk[[0, 1]] / a;
    for i in 0..3 {
      assert_eq!(shrunk[[i, i]], 1.0);
      for j in 0..i {
        assert!((shrunk[[i, j]] - (1.0 - lambda) * a).abs() < 1e-15);
        assert_eq!(shrunk[[i, j]], shrunk[[j, i]]);
      }
    }
    assert!(is_spd(shrunk.view()));
  }

  #[test]
  fn an_estimate_above_the_floor_is_returned_untouched() {
    for a in [0.3, -0.4985, -0.499] {
      let spd = array![[1.0, a, a], [a, 1.0, a], [a, a, 1.0]];
      assert!(smallest_eigenvalue(spd.view()) >= FLOOR);
      assert_eq!(shrink_to_spd(spd.clone()), spd);
    }
  }

  #[test]
  fn a_positive_definite_estimate_below_the_floor_is_shrunk_to_it() {
    let a = -0.4997;
    let thin = array![[1.0, a, a], [a, 1.0, a], [a, a, 1.0]];
    let mu = smallest_eigenvalue(thin.view());
    assert!(mu > 0.0 && mu < FLOOR);
    assert_floored(&thin, &shrink_to_spd(thin.clone()), mu);
    let comonotone = array![[1.0, 0.999999], [0.999999, 1.0]];
    let mu = smallest_eigenvalue(comonotone.view());
    assert!(is_spd(comonotone.view()) && mu < 1e-5);
    let shrunk = shrink_to_spd(comonotone.clone());
    assert_floored(&comonotone, &shrunk, mu);
    assert!(
      (0.999 - 1e-6..=0.999).contains(&shrunk[[0, 1]]),
      "{}",
      shrunk[[0, 1]]
    );
  }

  #[test]
  fn the_condition_number_of_a_shrunk_estimate_is_at_most_the_dimension_over_the_floor() {
    for estimate in [
      Array2::from_shape_fn((10, 10), |(i, j)| if i == j { 1.0 } else { -1.0 }),
      Array2::from_shape_fn((6, 6), |(i, j)| if i == j { 1.0 } else { 0.999999 }),
    ] {
      let d = estimate.nrows() as f64;
      assert!(condition_number(shrink_to_spd(estimate).view()) <= d / FLOOR);
    }
  }

  #[test]
  fn a_strongly_indefinite_matrix_keeps_its_correlations_instead_of_the_identity() {
    let all_negative = Array2::from_shape_fn((10, 10), |(i, j)| if i == j { 1.0 } else { -1.0 });
    let mu = smallest_eigenvalue(all_negative.view());
    assert!((mu + 8.0).abs() < 1e-10);
    let shrunk = shrink_to_spd(all_negative.clone());
    assert_floored(&all_negative, &shrunk, mu);
    assert_ne!(shrunk, Array2::<f64>::eye(10));
    assert!(shrunk[[0, 1]] < 0.0);
    assert!(is_spd(shrunk.view()));
  }

  #[test]
  fn a_non_finite_estimate_is_left_for_the_validator_to_reject() {
    let nan = array![[1.0, f64::NAN], [f64::NAN, 1.0]];
    assert!(CorrelationMatrix::try_new(shrink_to_spd(nan)).is_err());
  }
}

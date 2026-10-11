//! A validated correlation matrix and the factors the elliptical copulas evaluate through.
//! Reference: Higham, N.J., Strabić, N., Šego, V. (2016), "Restoring definiteness via shrinking, with an application to correlation matrices with a fixed block", SIAM Review 58(2), 245-263, eq. (2.1), §2 and §3.1, DOI 10.1137/140996112.

use ndarray::Array2;

use super::linalg::is_spd;
use super::linalg::spd_cholesky_lower;
use super::linalg::spd_inverse;
use crate::error::CopulaError;

const SHRINKAGE: [f64; 7] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0];

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
    let not_positive_definite = || {
      CopulaError::InvalidStructure(
        "corr must satisfy `pivot[i] > 0.0` (positive definite), got a non-positive Cholesky pivot"
          .into(),
      )
    };
    let chol_lower = spd_cholesky_lower(corr.view()).ok_or_else(not_positive_definite)?;
    let inverse = spd_inverse(corr.view()).ok_or_else(not_positive_definite)?;
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

/// Shrinks an estimated correlation toward the identity, `(1 − λ) R + λ I`, by the least `λ` of `1e-6, 1e-5, …, 1` whose
/// Cholesky test passes (a decade grid where the paper bisects); the unit diagonal survives.
pub(crate) fn shrink_to_spd(corr: Array2<f64>) -> Array2<f64> {
  if is_spd(corr.view()) {
    return corr;
  }
  let identity = Array2::<f64>::eye(corr.nrows());
  for lambda in SHRINKAGE {
    let mut shrunk = &corr * (1.0 - lambda) + &identity * lambda;
    shrunk.diag_mut().fill(1.0);
    if is_spd(shrunk.view()) {
      return shrunk;
    }
  }
  corr
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

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

  #[test]
  fn a_matrix_beyond_the_grid_shrinks_to_the_identity_with_positive_zeros() {
    let indefinite = array![[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]];
    let shrunk = shrink_to_spd(indefinite);
    assert_eq!(shrunk, Array2::<f64>::eye(3));
    assert!(shrunk.iter().all(|v| v.is_sign_positive()));
  }

  #[test]
  fn shrinking_takes_the_least_lambda_of_the_schedule_that_works() {
    let a = -0.5005;
    let equi = array![[1.0, a, a], [a, 1.0, a], [a, a, 1.0]];
    let shrunk = shrink_to_spd(equi.clone());
    for i in 0..3 {
      assert_eq!(shrunk[[i, i]], 1.0);
      for j in 0..i {
        assert!(
          (shrunk[[i, j]] - a * (1.0 - 1e-3)).abs() < 1e-15,
          "{shrunk:?}"
        );
        assert_eq!(shrunk[[i, j]], shrunk[[j, i]]);
      }
    }
    let spd = array![[1.0, 0.3], [0.3, 1.0]];
    assert_eq!(shrink_to_spd(spd.clone()), spd);
  }
}

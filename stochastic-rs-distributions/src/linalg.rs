//! `ndarray`-view linear algebra on faer for the sibling crates; outside the stability promise.
//! SPD and eigenvalue functions read the lower triangle; symmetry and finiteness are the caller's.

use faer::Side;
use faer::linalg::solvers::ColPivQr;
use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::solvers::Llt;
use faer::linalg::solvers::LltError;
use faer::linalg::solvers::PartialPivLu;
use faer::linalg::solvers::Solve;
use faer::linalg::solvers::SolveLstsqCore;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

/// Least-squares solution of `a x = y` through a column-pivoted QR; panics unless
/// `a.nrows() >= a.ncols()` and `y.len() == a.nrows()`.
pub fn lstsq(a: ArrayView2<f64>, y: ArrayView1<f64>) -> Array1<f64> {
  let qr = ColPivQr::new(a.into_faer());
  let mut rhs = faer::Mat::from_fn(y.len(), 1, |i, _| y[i]);
  qr.solve_lstsq_in_place_with_conj(faer::Conj::No, rhs.as_mut());
  Array1::from_iter((0..a.ncols()).map(|j| rhs[(j, 0)]))
}

/// LU solve of `a x = b`; `None` for a non-finite solution, a pivoted LU's singularity signal;
/// panics unless `a` is square and `b.len() == a.nrows()`.
pub fn solve(a: ArrayView2<f64>, b: ArrayView1<f64>) -> Option<Array1<f64>> {
  let lu = PartialPivLu::new(a.into_faer());
  let rhs = faer::Mat::from_fn(b.len(), 1, |i, _| b[i]);
  let x = lu.solve(rhs.as_ref());
  let out = Array1::from_iter((0..b.len()).map(|i| x[(i, 0)]));
  out.iter().all(|v| v.is_finite()).then_some(out)
}

/// Lower Cholesky factor, strict upper triangle zeroed; `None` when `a` is not positive definite;
/// panics if `a` is not square.
pub fn spd_cholesky_lower(a: ArrayView2<f64>) -> Option<Array2<f64>> {
  let llt = Llt::new(a.into_faer(), Side::Lower).ok()?;
  let mut l = llt.L().into_ndarray().to_owned();
  let n = l.nrows();
  for i in 0..n {
    for j in (i + 1)..n {
      l[[i, j]] = 0.0;
    }
  }
  Some(l)
}

/// Lower Cholesky factor (strict upper triangle zeroed) and inverse from one factorization, or the
/// index of the first non-positive pivot; panics if `a` is not square.
pub fn spd_factors(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), usize> {
  let llt =
    Llt::new(a.into_faer(), Side::Lower).map_err(|LltError::NonPositivePivot { index }| index)?;
  let mut l = llt.L().into_ndarray().to_owned();
  let n = l.nrows();
  for i in 0..n {
    for j in (i + 1)..n {
      l[[i, j]] = 0.0;
    }
  }
  let inverse = llt.inverse().as_ref().into_ndarray().to_owned();
  Ok((l, inverse))
}

/// Positive-definiteness probe: does the Cholesky factorization succeed?
pub fn is_spd(a: ArrayView2<f64>) -> bool {
  a.nrows() == a.ncols() && Llt::new(a.into_faer(), Side::Lower).is_ok()
}

/// Eigenvalues of a symmetric matrix in nondecreasing order, the lower triangle read; `None` where
/// `a` is not square or the solver does not converge.
pub fn symmetric_eigenvalues(a: ArrayView2<f64>) -> Option<Array1<f64>> {
  if a.nrows() != a.ncols() {
    return None;
  }
  a.into_faer()
    .self_adjoint_eigenvalues(Side::Lower)
    .ok()
    .map(Array1::from)
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn lstsq_recovers_exact_coefficients() {
    let a = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y = array![1.0, 3.0, 5.0, 7.0];
    let beta = lstsq(a.view(), y.view());
    assert!((beta[0] - 1.0).abs() < 1e-12 && (beta[1] - 2.0).abs() < 1e-12);
  }

  #[test]
  fn solve_answers_a_regular_system_and_flags_a_singular_one() {
    let a = array![[2.0, 1.0], [1.0, 3.0]];
    let b = array![5.0, 10.0];
    let x = solve(a.view(), b.view()).expect("nonsingular");
    assert_eq!(x.len(), 2);
    for (got, want) in a.dot(&x).iter().zip(b.iter()) {
      assert!((got - want).abs() < 1e-12);
    }
    let sing = array![[1.0, 2.0], [2.0, 4.0]];
    assert!(solve(sing.view(), b.view()).is_none());
  }

  #[test]
  fn the_factors_reconstruct_and_invert_the_matrix() {
    let a = array![[4.0, 2.0], [2.0, 3.0]];
    let (l, inv) = spd_factors(a.view()).expect("SPD");
    let back = l.dot(&l.t());
    let id = a.dot(&inv);
    for i in 0..2 {
      for j in 0..2 {
        assert!((back[[i, j]] - a[[i, j]]).abs() < 1e-12, "L L^T mismatch");
        let want = if i == j { 1.0 } else { 0.0 };
        assert!((id[[i, j]] - want).abs() < 1e-12, "A A^-1 != I");
      }
    }
    assert_eq!(l[[0, 1]], 0.0, "strict upper triangle must be zeroed");
    assert_eq!(spd_cholesky_lower(a.view()).expect("SPD"), l);
  }

  #[test]
  fn an_indefinite_matrix_is_rejected_at_its_first_non_positive_pivot() {
    let a = array![[1.0, 2.0], [2.0, 1.0]];
    assert_eq!(spd_factors(a.view()).unwrap_err(), 1);
    assert_eq!(
      spd_factors(array![[-1.0, 0.0], [0.0, 1.0]].view()).unwrap_err(),
      0
    );
    assert!(!is_spd(a.view()));
    assert!(spd_cholesky_lower(a.view()).is_none());
  }

  #[test]
  fn equicorrelation_eigenvalues_match_the_closed_form() {
    let equi = |rho: f64| Array2::from_shape_fn((3, 3), |(i, j)| if i == j { 1.0 } else { rho });
    for (rho, want) in [(0.5, [0.5, 0.5, 2.0]), (-0.25, [0.5, 1.25, 1.25])] {
      let got = symmetric_eigenvalues(equi(rho).view()).expect("symmetric");
      assert_eq!(got.len(), 3);
      for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-12, "rho = {rho}: {g} vs {w}");
      }
    }
    assert!(symmetric_eigenvalues(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].view()).is_none());
  }
}

//! faer-backed dense-linalg glue for the estimators: OLS least squares,
//! LU solve / inverse with a finite-solution singularity probe, SPD
//! Cholesky, and complex eigenvalues — all on `ndarray` types.

use faer::Side;
use faer::linalg::solvers::ColPivQr;
use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::solvers::Eigen;
use faer::linalg::solvers::Llt;
use faer::linalg::solvers::PartialPivLu;
use faer::linalg::solvers::Solve;
use faer::linalg::solvers::SolveLstsqCore;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex;

/// Least-squares solution of `a x = y` through a column-pivoted QR.
pub(crate) fn lstsq(a: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
  let qr = ColPivQr::new(a.view().into_faer());
  let mut rhs = faer::Mat::from_fn(y.len(), 1, |i, _| y[i]);
  qr.solve_lstsq_in_place_with_conj(faer::Conj::No, rhs.as_mut());
  Array1::from_iter((0..a.ncols()).map(|j| rhs[(j, 0)]))
}

/// LU solve of `a x = b`. `None` when the solution is not finite — the
/// practical singularity signal a pivoted LU emits instead of an error.
pub(crate) fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
  let lu = PartialPivLu::new(a.view().into_faer());
  let rhs = faer::Mat::from_fn(b.len(), 1, |i, _| b[i]);
  let x = lu.solve(rhs.as_ref());
  let out = Array1::from_iter((0..b.len()).map(|i| x[(i, 0)]));
  out.iter().all(|v| v.is_finite()).then_some(out)
}

/// LU inverse. `None` when the inverse is not finite (singular input).
pub(crate) fn inverse(a: &Array2<f64>) -> Option<Array2<f64>> {
  let lu = PartialPivLu::new(a.view().into_faer());
  let inv = lu.inverse();
  let out = inv.as_ref().into_ndarray().to_owned();
  out.iter().all(|v| v.is_finite()).then_some(out)
}

/// Lower Cholesky factor of an SPD matrix, strict upper triangle zeroed.
pub(crate) fn spd_cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
  let llt = Llt::new(a.view().into_faer(), Side::Lower).ok()?;
  let mut l = llt.L().into_ndarray().to_owned();
  let n = l.nrows();
  for i in 0..n {
    for j in (i + 1)..n {
      l[[i, j]] = 0.0;
    }
  }
  Some(l)
}

/// Complex eigenvalues of a general real square matrix.
pub(crate) fn eigenvalues(a: &Array2<f64>) -> Option<Vec<Complex<f64>>> {
  let evd: Eigen<f64> = Eigen::new_from_real(a.view().into_faer()).ok()?;
  let s = evd.S();
  Some((0..a.nrows()).map(|i| s[i]).collect())
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn lstsq_recovers_exact_coefficients() {
    let a = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y = array![1.0, 3.0, 5.0, 7.0];
    let beta = lstsq(&a, &y);
    assert!((beta[0] - 1.0).abs() < 1e-12 && (beta[1] - 2.0).abs() < 1e-12);
  }

  #[test]
  fn solve_matches_direct_inverse_and_flags_singular() {
    let a = array![[2.0, 1.0], [1.0, 3.0]];
    let b = array![5.0, 10.0];
    let x = solve(&a, &b).expect("nonsingular");
    assert!((a.dot(&x)[0] - 5.0).abs() < 1e-12);
    let sing = array![[1.0, 2.0], [2.0, 4.0]];
    assert!(solve(&sing, &b).is_none());
  }

  #[test]
  fn eigenvalues_of_diagonal_matrix() {
    let a = array![[3.0, 0.0], [0.0, -2.0]];
    let mut ev: Vec<f64> = eigenvalues(&a).expect("evd").iter().map(|c| c.re).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!((ev[0] + 2.0).abs() < 1e-12 && (ev[1] - 3.0).abs() < 1e-12);
  }
}

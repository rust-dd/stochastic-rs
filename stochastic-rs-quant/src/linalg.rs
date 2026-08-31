//! faer-backed dense-linalg glue: least squares, SPD Cholesky, LU
//! inverse/solve and thin SVD on `ndarray` types, plus `FloatExt`-generic
//! wrappers that round-trip through `f64` (the only scalars in play are
//! `f32`/`f64`, and factoring an `f32` matrix in `f64` loses nothing).

use faer::Side;
use faer::linalg::solvers::ColPivQr;
use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::solvers::Llt;
use faer::linalg::solvers::PartialPivLu;
use faer::linalg::solvers::Solve;
use faer::linalg::solvers::SolveLstsqCore;
use faer::linalg::solvers::Svd;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array1;
use ndarray::Array2;

use crate::traits::RealExt;

/// Least-squares solution of `a x = y` through a column-pivoted QR.
pub(crate) fn lstsq(a: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
  let qr = ColPivQr::new(a.view().into_faer());
  let mut rhs = faer::Mat::from_fn(y.len(), 1, |i, _| y[i]);
  qr.solve_lstsq_in_place_with_conj(faer::Conj::No, rhs.as_mut());
  Array1::from_iter((0..a.ncols()).map(|j| rhs[(j, 0)]))
}

/// LU solve of `a x = b`. `None` when the solution is not finite.
pub(crate) fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
  let lu = PartialPivLu::new(a.view().into_faer());
  let rhs = faer::Mat::from_fn(b.len(), 1, |i, _| b[i]);
  let x = lu.solve(rhs.as_ref());
  let out = Array1::from_iter((0..b.len()).map(|i| x[(i, 0)]));
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

/// Thin SVD: `(u, s, vt)` with `u` carrying `min(m, n)` columns.
pub(crate) fn svd_thin(a: &Array2<f64>) -> Option<(Array2<f64>, Array1<f64>, Array2<f64>)> {
  let svd = Svd::new(a.view().into_faer()).ok()?;
  let r = a.nrows().min(a.ncols());
  let u_full = svd.U().into_ndarray();
  let v_full = svd.V().into_ndarray();
  let s = svd.S();
  let u = u_full.slice(ndarray::s![.., ..r]).to_owned();
  let sigma = Array1::from_iter((0..r).map(|i| s[i]));
  let vt = v_full.slice(ndarray::s![.., ..r]).t().to_owned();
  Some((u, sigma, vt))
}

/// `FloatExt`-generic SPD Cholesky through an `f64` round trip.
pub(crate) fn spd_cholesky_lower_t<T: RealExt>(a: &Array2<T>) -> Option<Array2<T>> {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  spd_cholesky_lower(&a64).map(|l| l.mapv(T::from_f64_fast))
}

/// `FloatExt`-generic SPD probe.
pub(crate) fn is_spd_t<T: RealExt>(a: &Array2<T>) -> bool {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  a64.iter().all(|v| v.is_finite()) && Llt::new(a64.view().into_faer(), Side::Lower).is_ok()
}

/// `FloatExt`-generic LU inverse through an `f64` round trip. `None` when
/// the inverse is not finite (singular input).
pub(crate) fn inverse_t<T: RealExt>(a: &Array2<T>) -> Option<Array2<T>> {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  let lu = PartialPivLu::new(a64.view().into_faer());
  let inv = lu.inverse();
  let out = inv.as_ref().into_ndarray().to_owned();
  out
    .iter()
    .all(|v| v.is_finite())
    .then(|| out.mapv(T::from_f64_fast))
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn svd_thin_reconstructs_the_matrix() {
    let a = array![[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]];
    let (u, s, vt) = svd_thin(&a).expect("svd");
    let mut back = Array2::<f64>::zeros((3, 2));
    for i in 0..3 {
      for j in 0..2 {
        for k in 0..2 {
          back[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
        }
      }
    }
    for (want, got) in a.iter().zip(back.iter()) {
      assert!((want - got).abs() < 1e-12, "reconstruction mismatch");
    }
  }

  #[test]
  fn generic_cholesky_round_trips_f32() {
    let a = array![[4.0_f32, 2.0], [2.0, 3.0]];
    let l = spd_cholesky_lower_t(&a).expect("SPD");
    let back = l.dot(&l.t());
    for (want, got) in a.iter().zip(back.iter()) {
      assert!((want - got).abs() < 1e-6, "L L^T mismatch");
    }
  }

  #[test]
  fn singular_generic_inverse_is_none() {
    let a = array![[1.0, 2.0], [2.0, 4.0]];
    assert!(inverse_t(&a).is_none());
  }
}

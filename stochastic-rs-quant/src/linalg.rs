//! faer-backed thin SVD and truncated least squares, plus `RealExt`-generic wrappers that
//! round-trip the shared `f64` factorisations (factoring an `f32` matrix in `f64` loses nothing).

use faer::linalg::solvers::Svd;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::linalg::inverse;
use stochastic_rs_distributions::linalg::is_spd;
use stochastic_rs_distributions::linalg::solve;
use stochastic_rs_distributions::linalg::spd_cholesky_lower;

use crate::traits::RealExt;

/// LU solve of the row-major `N`×`N` system `a x = b`; `None` for a singular system or an `a_flat`
/// that is not `N * N` long; panics unless `b.len() == N`.
pub(crate) fn solve_small<const N: usize>(a_flat: &[f64], b: &[f64]) -> Option<[f64; N]> {
  let a = ArrayView2::from_shape((N, N), a_flat).ok()?;
  let x = solve(a, ArrayView1::from(b))?;
  Some(std::array::from_fn(|i| x[i]))
}

/// Thin SVD: `(u, s, vt)` with `u` carrying `min(m, n)` columns.
pub(crate) fn svd_thin(a: &Array2<f64>) -> Option<(Array2<f64>, Array1<f64>, Array2<f64>)> {
  let svd = Svd::new_thin(a.view().into_faer()).ok()?;
  let r = a.nrows().min(a.ncols());
  let s = svd.S();
  let u = svd.U().into_ndarray().to_owned();
  let sigma = Array1::from_iter((0..r).map(|i| s[i]));
  let vt = svd.V().into_ndarray().t().to_owned();
  Some((u, sigma, vt))
}

/// Least-squares solution of `a x = y` through a truncated pseudo-inverse:
/// singular values at or below `eps` are dropped, so a rank-deficient design
/// gets the minimum-norm solution instead of an exploding one. `None` when the
/// factorisation fails or the solution is not finite.
pub(crate) fn lstsq_svd(a: &Array2<f64>, y: &Array1<f64>, eps: f64) -> Option<Array1<f64>> {
  let (u, s, vt) = svd_thin(a)?;
  let mut projected = u.t().dot(y);
  for (value, sigma) in projected.iter_mut().zip(s.iter()) {
    *value = if *sigma > eps { *value / sigma } else { 0.0 };
  }
  let x = vt.t().dot(&projected);
  x.iter().all(|v| v.is_finite()).then_some(x)
}

/// SPD Cholesky through an `f64` round trip.
pub(crate) fn spd_cholesky_lower_t<T: RealExt>(a: &Array2<T>) -> Option<Array2<T>> {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  spd_cholesky_lower(a64.view()).map(|l| l.mapv(T::from_f64_fast))
}

/// SPD probe; `false` for a non-finite entry, which the shared probe leaves to its caller.
pub(crate) fn is_spd_t<T: RealExt>(a: &Array2<T>) -> bool {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  a64.iter().all(|v| v.is_finite()) && is_spd(a64.view())
}

/// LU inverse through an `f64` round trip; `None` when it is not finite (singular input).
pub(crate) fn inverse_t<T: RealExt>(a: &Array2<T>) -> Option<Array2<T>> {
  let a64 = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
  inverse(a64.view()).map(|inv| inv.mapv(T::from_f64_fast))
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn solve_small_answers_a_regular_system_and_flags_a_singular_or_short_one() {
    let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
    let x = solve_small::<3>(&a, &[4.0, 10.0, 14.0]).expect("nonsingular");
    for (got, want) in x.iter().zip([1.0, 2.0, 3.0]) {
      assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }
    assert!(solve_small::<2>(&[1.0, 2.0, 2.0, 4.0], &[5.0, 10.0]).is_none());
    assert!(solve_small::<3>(&a[..8], &[4.0, 10.0, 14.0]).is_none());
  }

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
  fn svd_least_squares_recovers_a_tall_regression() {
    let a = Array2::from_shape_fn((20_000, 3), |(i, j)| {
      let x = i as f64 / 20_000.0;
      x.powi(j as i32)
    });
    let expected = array![1.5, -2.0, 0.75];
    let y = a.dot(&expected);
    let actual = lstsq_svd(&a, &y, 1e-10).expect("least squares");
    for (actual, expected) in actual.iter().zip(expected.iter()) {
      assert!((actual - expected).abs() < 1e-10);
    }
  }

  #[test]
  fn svd_least_squares_returns_minimum_norm_for_dependent_columns() {
    let a = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
    let y = array![3.0, 6.0, 9.0];
    let actual = lstsq_svd(&a, &y, 1e-10).expect("rank-deficient least squares");
    assert!((actual[0] - 0.6).abs() < 1e-12);
    assert!((actual[1] - 1.2).abs() < 1e-12);
  }

  #[test]
  fn svd_least_squares_returns_minimum_norm_for_a_wide_system() {
    let a = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
    let y = array![1.0, 1.0];
    let actual = lstsq_svd(&a, &y, 1e-10).expect("underdetermined least squares");
    for (actual, expected) in actual.iter().zip([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]) {
      assert!((actual - expected).abs() < 1e-12);
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

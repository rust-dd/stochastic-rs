//! # Correlation
//!
//! $$
//! \rho_{ij}=\frac{\operatorname{Cov}(X_i,X_j)}{\sigma_i\sigma_j}
//! $$
//!
//! Kendall τ-b, and its conversion between the bivariate copulas (parametrised by τ) and the Gaussian multivariate copula
//! (parametrised by a correlation matrix), exact for the Gaussian copula and the canonical approximation for elliptical ones:
//!
//! $$
//! \rho = \sin\!\left(\frac{\pi\tau}{2}\right)
//! $$
//!
//! Reference: Knight, W. R. (1966), "A Computer Method for Calculating Kendall's Tau with Ungrouped Data", Journal of the American Statistical Association 61(314), 436–439, doi:10.1080/01621459.1966.10480879.
//! Reference: Kendall, M. G. (1945), "The treatment of ties in ranking problems", Biometrika 33(3), 239–251, doi:10.1093/biomet/33.3.239.

use std::borrow::Cow;
use std::cmp::Ordering;

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::error::CopulaError;

/// Kendall's τ-b of two columns (Knight's O(n log n) count, ties counted per margin): NaN where it is undefined (a
/// constant column or a NaN entry); `DimensionMismatch` on unequal lengths, `InsufficientData` below two rows.
pub fn kendall_tau_b(u: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<f64, CopulaError> {
  if u.len() != v.len() {
    return Err(CopulaError::DimensionMismatch {
      expected: u.len(),
      got: v.len(),
    });
  }
  if u.len() < 2 {
    return Err(CopulaError::InsufficientData {
      needed: 2,
      got: u.len(),
    });
  }
  if u.iter().any(|x| x.is_nan()) || v.iter().any(|x| x.is_nan()) {
    return Ok(f64::NAN);
  }
  let (tau, ..) = kendalls::tau_b_with_comparator(&as_slice(u), &as_slice(v), |a, b| {
    a.partial_cmp(b).unwrap_or(Ordering::Greater)
  })
  .map_err(|_| CopulaError::InsufficientData {
    needed: 2,
    got: u.len(),
  })?;
  Ok(tau)
}

/// The column as `kendalls` takes it, copied only when its elements are not contiguous.
fn as_slice(column: ArrayView1<'_, f64>) -> Cow<'_, [f64]> {
  column
    .to_slice()
    .map_or_else(|| Cow::Owned(column.to_vec()), Cow::Borrowed)
}

/// Pairwise Kendall τ-b matrix of the columns of `data` ([`kendall_tau_b`] per pair, ones on the diagonal);
/// `InsufficientData` below two rows.
pub fn kendall_tau(data: ArrayView2<f64>) -> Result<Array2<f64>, CopulaError> {
  if data.nrows() < 2 {
    return Err(CopulaError::InsufficientData {
      needed: 2,
      got: data.nrows(),
    });
  }
  let cols = data.ncols();
  let mut tau = Array2::<f64>::eye(cols);
  for i in 0..cols {
    for j in (i + 1)..cols {
      let t = kendall_tau_b(data.column(i), data.column(j))?;
      tau[[i, j]] = t;
      tau[[j, i]] = t;
    }
  }
  Ok(tau)
}

/// `InvalidStructure` naming the columns of the τ-b matrix `tau` whose entries with every other column are NaN (a constant
/// or NaN column): the fits that invert τ need every column to vary.
pub(crate) fn check_defined(tau: ArrayView2<f64>) -> Result<(), CopulaError> {
  match undefined_columns(tau) {
    Some(columns) => Err(undefined_tau_b(&columns, "")),
    None => Ok(()),
  }
}

/// The columns `check_defined` names: those NaN against every other column, else the first NaN pair; `None` when all are defined.
pub(crate) fn undefined_columns(tau: ArrayView2<f64>) -> Option<Vec<usize>> {
  let d = tau.nrows();
  let ((i, j), _) = tau.indexed_iter().find(|(_, t)| t.is_nan())?;
  let culprits: Vec<usize> = (0..d)
    .filter(|&c| (0..d).filter(|&r| r != c).all(|r| tau[[r, c]].is_nan()))
    .collect();
  Some(if culprits.is_empty() {
    vec![i, j]
  } else {
    culprits
  })
}

/// The `InvalidStructure` of `check_defined` for the data columns `columns`, `given` qualifying them.
pub(crate) fn undefined_tau_b(columns: &[usize], given: &str) -> CopulaError {
  CopulaError::InvalidStructure(match columns {
    [column] => {
      format!(
        "Kendall's tau-b is undefined for column {column}{given}: it is constant or holds a NaN"
      )
    }
    columns => format!(
      "Kendall's tau-b is undefined for columns {columns:?}{given}: at least one of them is constant or holds a NaN"
    ),
  })
}

/// $\rho = \sin(\pi\tau/2)$: the correlation of the Gaussian or t copula with Kendall's `tau`;
/// feeds a `BivariateExt` fit into the `MultivariateExt` Gaussian constructor.
pub fn tau_to_corr(tau: f64) -> f64 {
  (std::f64::consts::FRAC_PI_2 * tau).sin()
}

/// Inverse of [`tau_to_corr`]: $\tau = \frac{2}{\pi}\arcsin(\rho)$; NaN outside `[-1, 1]`.
pub fn corr_to_tau(rho: f64) -> f64 {
  2.0 / std::f64::consts::PI * rho.asin()
}

/// Apply [`tau_to_corr`] elementwise to a Kendall-tau matrix to obtain a
/// Gaussian-copula correlation matrix on the same indexing.
pub fn tau_matrix_to_corr_matrix(tau: ArrayView2<f64>) -> Array2<f64> {
  tau.mapv(tau_to_corr)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tau_corr_round_trip() {
    for &t in &[-0.7, -0.3, 0.0, 0.2, 0.8] {
      let r = tau_to_corr(t);
      let t2 = corr_to_tau(r);
      assert!(
        (t - t2).abs() < 1e-12,
        "round-trip failed: {t} -> {r} -> {t2}"
      );
    }
  }

  #[test]
  fn tau_zero_means_corr_zero() {
    assert!(tau_to_corr(0.0).abs() < 1e-15);
  }

  #[test]
  fn tau_one_implies_corr_one() {
    assert!((tau_to_corr(1.0) - 1.0).abs() < 1e-15);
    assert!((tau_to_corr(-1.0) + 1.0).abs() < 1e-15);
  }

  #[test]
  fn a_correlation_outside_the_unit_interval_has_no_tau() {
    assert!(corr_to_tau(1.5).is_nan());
    assert!((corr_to_tau(1.0) - 1.0).abs() < 1e-15);
  }

  /// τ-b counts ties in each margin (Kendall 1945): on `(1,1,2,3)` × `(1,2,2,3)` it is `0.8` where τ-a is `2/3`.
  #[test]
  fn kendall_tau_is_tau_b() {
    let data = ndarray::array![[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]];
    let tau = kendall_tau(data.view()).unwrap();
    assert!((tau[[0, 1]] - 0.8).abs() < 1e-15, "{}", tau[[0, 1]]);
    assert_eq!(tau[[0, 0]], 1.0);
    assert!(kendall_tau(ndarray::Array2::<f64>::zeros((1, 2)).view()).is_err());
    assert!((kendall_tau_b(data.column(0), data.column(1)).unwrap() - 0.8).abs() < 1e-15);
  }

  /// scipy 1.17 `kendalltau(x, y, variant="b")` on samples with ties in both margins (τ-a of the first is -0.249).
  #[test]
  fn kendall_tau_b_agrees_with_scipy() {
    let x = ndarray::array![
      12.0, 14.0, 14.0, 17.0, 19.0, 19.0, 19.0, 19.0, 19.0, 20.0, 21.0, 21.0, 21.0, 21.0, 21.0,
      22.0, 23.0, 24.0, 24.0, 24.0, 26.0, 26.0, 27.0
    ];
    let y = ndarray::array![
      11.0, 4.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0,
      0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let tau = kendall_tau_b(x.view(), y.view()).unwrap();
    assert!((tau + 0.37620154104750975).abs() < 1e-14, "{tau}");
    let x = ndarray::array![12.0, 2.0, 1.0, 12.0, 2.0];
    let y = ndarray::array![1.0, 4.0, 7.0, 1.0, 0.0];
    let tau = kendall_tau_b(x.view(), y.view()).unwrap();
    assert!((tau + 0.4714045207910316).abs() < 1e-14, "{tau}");
  }

  /// Knight's O(n log n) count against `(P − Q) / √((P + Q + T)(P + Q + U))` from the pair definition, with `T` and `U`
  /// the pairs tied in `u` only and in `v` only.
  #[test]
  fn kendall_tau_b_matches_the_pair_count_on_tied_samples() {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut draw = |levels: u64| {
      state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      ((state >> 33) % levels) as f64
    };
    for n in [2usize, 3, 5, 12, 40, 97] {
      for levels in [2u64, 3, 6, 50] {
        let u = ndarray::Array1::from_shape_fn(n, |_| draw(levels));
        let v = ndarray::Array1::from_shape_fn(n, |_| draw(levels));
        let (mut p, mut q, mut t, mut s) = (0.0_f64, 0.0, 0.0, 0.0);
        for i in 0..n {
          for j in (i + 1)..n {
            let (du, dv) = (u[i] - u[j], v[i] - v[j]);
            match (du == 0.0, dv == 0.0) {
              (true, true) => {}
              (true, false) => t += 1.0,
              (false, true) => s += 1.0,
              (false, false) if du * dv > 0.0 => p += 1.0,
              (false, false) => q += 1.0,
            }
          }
        }
        let want = (p - q) / ((p + q + t) * (p + q + s)).sqrt();
        let got = kendall_tau_b(u.view(), v.view()).unwrap();
        assert!(
          (got.is_nan() && want.is_nan()) || (got - want).abs() < 1e-14,
          "n = {n}, levels = {levels}: {got} vs {want}"
        );
      }
    }
  }

  /// A constant column or a NaN entry leaves τ-b undefined, which is NaN; the sort behind it panics on a NaN.
  #[test]
  fn kendall_tau_b_is_nan_where_it_is_undefined() {
    let ramp = ndarray::Array1::from_shape_fn(50, |i| i as f64);
    let constant = ndarray::Array1::from_elem(50, 1.0);
    assert!(
      kendall_tau_b(constant.view(), ramp.view())
        .unwrap()
        .is_nan()
    );
    assert!(
      kendall_tau_b(ramp.view(), constant.view())
        .unwrap()
        .is_nan()
    );
    let mut holed = ramp.clone();
    holed[25] = f64::NAN;
    assert!(kendall_tau_b(holed.view(), ramp.view()).unwrap().is_nan());
    assert!(kendall_tau_b(ramp.view(), holed.view()).unwrap().is_nan());
    let tau = kendall_tau(ndarray::stack![ndarray::Axis(1), ramp, constant].view()).unwrap();
    assert!(tau[[0, 1]].is_nan() && tau[[1, 0]].is_nan());
    assert_eq!((tau[[0, 0]], tau[[1, 1]]), (1.0, 1.0));
  }

  #[test]
  fn fewer_than_two_rows_are_insufficient_data() {
    let one = ndarray::array![1.0];
    let want = Err(CopulaError::InsufficientData { needed: 2, got: 1 });
    assert_eq!(kendall_tau_b(one.view(), one.view()), want);
    assert_eq!(
      kendall_tau(ndarray::Array2::<f64>::zeros((1, 1)).view()),
      Err(CopulaError::InsufficientData { needed: 2, got: 1 })
    );
    assert_eq!(
      kendall_tau(ndarray::Array2::<f64>::zeros((0, 3)).view()),
      Err(CopulaError::InsufficientData { needed: 2, got: 0 })
    );
  }

  /// The guard names the column whose τ-b with every other column is NaN, not the first NaN pair row by row; where
  /// several columns qualify (two columns, or two constant ones) it lists them.
  #[test]
  fn an_undefined_tau_names_the_column_that_has_none() {
    let ramp = ndarray::Array1::from_shape_fn(20, |i| i as f64);
    let flat = ndarray::Array1::from_elem(20, 1.0);
    let why = |columns: ndarray::Array2<f64>| match check_defined(
      kendall_tau(columns.view()).unwrap().view(),
    ) {
      Err(CopulaError::InvalidStructure(why)) => why,
      other => panic!("expected InvalidStructure, got {other:?}"),
    };
    let reversed = ramp.mapv(|v| 19.0 - v);
    let one = ndarray::stack![ndarray::Axis(1), ramp, reversed, flat, ramp];
    assert_eq!(
      why(one),
      "Kendall's tau-b is undefined for column 2: it is constant or holds a NaN"
    );
    let two = ndarray::stack![ndarray::Axis(1), ramp, flat, flat, reversed];
    assert!(why(two).contains("undefined for columns [1, 2]: at least one of them"));
    let pair = ndarray::stack![ndarray::Axis(1), ramp, flat];
    assert!(why(pair).contains("undefined for columns [0, 1]: at least one of them"));
    let fine = ndarray::stack![ndarray::Axis(1), ramp, reversed];
    assert_eq!(
      check_defined(kendall_tau(fine.view()).unwrap().view()),
      Ok(())
    );
    let stray = ndarray::array![[1.0, f64::NAN, 0.5], [f64::NAN, 1.0, 0.5], [0.5, 0.5, 1.0]];
    assert!(
      matches!(check_defined(stray.view()), Err(CopulaError::InvalidStructure(why)) if why.contains("columns [0, 1]"))
    );
  }

  #[test]
  fn columns_of_different_lengths_are_a_dimension_mismatch() {
    let got = kendall_tau_b(
      ndarray::array![1.0, 2.0].view(),
      ndarray::array![1.0].view(),
    );
    assert_eq!(
      got,
      Err(CopulaError::DimensionMismatch {
        expected: 2,
        got: 1
      })
    );
  }
}

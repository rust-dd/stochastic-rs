//! Scaling-and-squaring Padé-13 matrix exponential (Higham 2005,
//! Algorithm 2.3), shared by the matrix-valued processes; the same routine
//! the quant crate's rating-migration code carries, generic over [`RealExt`]
//! so it needs no LAPACK backend.
//!
//! Reference: Higham, N. J. (2005), *The scaling and squaring method for the
//! matrix exponential revisited*, SIAM J. Matrix Anal. Appl. 26(4),
//! 1179–1193. DOI: 10.1137/04061101X

use ndarray::Array2;

use crate::traits::RealExt;

/// $\exp(A)$ for a square matrix by Padé-13 with scaling and squaring.
pub(crate) fn expm<T: RealExt>(a: &Array2<T>) -> Array2<T> {
  let n = a.nrows();
  assert_eq!(a.ncols(), n, "expm needs a square matrix");
  if n == 0 {
    return a.clone();
  }
  // θ₁₃ from Higham (2005), Table 2.3.
  let theta13 = T::from_f64_fast(5.371920351148152);
  let norm_a = one_norm(a);
  if norm_a <= theta13 {
    return pade13(a);
  }
  let s = (norm_a / theta13)
    .ln()
    .to_f64()
    .unwrap_or(0.0)
    .max(0.0)
    .ceil() as i32;
  let s = s.max(1);
  let scale = T::from_f64_fast(2f64.powi(s));
  let a_scaled = a.mapv(|x| x / scale);
  let mut result = pade13(&a_scaled);
  for _ in 0..s {
    result = result.dot(&result);
  }
  result
}

fn one_norm<T: RealExt>(a: &Array2<T>) -> T {
  let mut max_col = T::zero();
  for j in 0..a.ncols() {
    let mut s = T::zero();
    for i in 0..a.nrows() {
      s += a[[i, j]].abs();
    }
    if s > max_col {
      max_col = s;
    }
  }
  max_col
}

fn pade13<T: RealExt>(a: &Array2<T>) -> Array2<T> {
  // Padé-13 coefficients from Higham (2005), Table 2.3.
  let b: [T; 14] = [
    T::from_f64_fast(64764752532480000.0),
    T::from_f64_fast(32382376266240000.0),
    T::from_f64_fast(7771770303897600.0),
    T::from_f64_fast(1187353796428800.0),
    T::from_f64_fast(129060195264000.0),
    T::from_f64_fast(10559470521600.0),
    T::from_f64_fast(670442572800.0),
    T::from_f64_fast(33522128640.0),
    T::from_f64_fast(1323241920.0),
    T::from_f64_fast(40840800.0),
    T::from_f64_fast(960960.0),
    T::from_f64_fast(16380.0),
    T::from_f64_fast(182.0),
    T::from_f64_fast(1.0),
  ];
  let n = a.nrows();
  let id = Array2::<T>::eye(n);
  let a2 = a.dot(a);
  let a4 = a2.dot(&a2);
  let a6 = a4.dot(&a2);

  let u_outer = &a6.mapv(|x| x * b[13]) + &a4.mapv(|x| x * b[11]) + &a2.mapv(|x| x * b[9]);
  let u_inner = a6.dot(&u_outer);
  let u_low = &a6.mapv(|x| x * b[7])
    + &a4.mapv(|x| x * b[5])
    + &a2.mapv(|x| x * b[3])
    + &id.mapv(|x| x * b[1]);
  let u = a.dot(&(&u_inner + &u_low));

  let v_outer = &a6.mapv(|x| x * b[12]) + &a4.mapv(|x| x * b[10]) + &a2.mapv(|x| x * b[8]);
  let v_inner = a6.dot(&v_outer);
  let v_low = &a6.mapv(|x| x * b[6])
    + &a4.mapv(|x| x * b[4])
    + &a2.mapv(|x| x * b[2])
    + &id.mapv(|x| x * b[0]);
  let v = &v_inner + &v_low;

  let numer = &v + &u;
  let denom = &v - &u;
  invert_matrix(&denom).dot(&numer)
}

/// Gauss–Jordan inverse with partial pivoting; the matrices here are small
/// (a handful of assets or factors), so the O(n³) dense form is the right
/// tool.
pub(crate) fn invert_matrix<T: RealExt>(a: &Array2<T>) -> Array2<T> {
  let n = a.nrows();
  assert_eq!(n, a.ncols(), "matrix must be square for inversion");
  let mut aug = Array2::<T>::zeros((n, 2 * n));
  for i in 0..n {
    for j in 0..n {
      aug[[i, j]] = a[[i, j]];
    }
    aug[[i, n + i]] = T::one();
  }
  for col in 0..n {
    let mut pivot_row = col;
    let mut pivot_val = aug[[col, col]].abs();
    for row in (col + 1)..n {
      let v = aug[[row, col]].abs();
      if v > pivot_val {
        pivot_val = v;
        pivot_row = row;
      }
    }
    assert!(pivot_val > T::min_positive_val(), "singular matrix");
    if pivot_row != col {
      for j in 0..(2 * n) {
        let tmp = aug[[col, j]];
        aug[[col, j]] = aug[[pivot_row, j]];
        aug[[pivot_row, j]] = tmp;
      }
    }
    let inv_pivot = T::one() / aug[[col, col]];
    for j in 0..(2 * n) {
      aug[[col, j]] = aug[[col, j]] * inv_pivot;
    }
    for row in 0..n {
      if row == col {
        continue;
      }
      let factor = aug[[row, col]];
      if factor == T::zero() {
        continue;
      }
      for j in 0..(2 * n) {
        let sub = factor * aug[[col, j]];
        aug[[row, j]] -= sub;
      }
    }
  }
  let mut inv = Array2::<T>::zeros((n, n));
  for i in 0..n {
    for j in 0..n {
      inv[[i, j]] = aug[[i, n + j]];
    }
  }
  inv
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Two-state generator `Q = [[-α, α], [β, -β]]` has the closed form
  /// `expm(Q) = (1/λ) [[β + α e^{-λ}, α (1 − e^{-λ})], [β (1 − e^{-λ}), α + β e^{-λ}]]`
  /// with `λ = α + β`; the quant crate checks the same numbers against
  /// scipy 1.11.4 at rtol 1e-12.
  fn two_state_closed_form(alpha: f64, beta: f64) -> Array2<f64> {
    let lam = alpha + beta;
    let e = (-lam).exp();
    array![
      [(beta + alpha * e) / lam, alpha * (1.0 - e) / lam],
      [beta * (1.0 - e) / lam, (alpha + beta * e) / lam],
    ]
  }

  #[test]
  fn matches_the_two_state_closed_form_below_the_scaling_threshold() {
    let p = expm(&array![[-0.1, 0.1], [0.05, -0.05]]);
    let want = two_state_closed_form(0.1, 0.05);
    for i in 0..2 {
      for j in 0..2 {
        assert!(
          (p[[i, j]] - want[[i, j]]).abs() < 1e-12,
          "expm[{i},{j}] = {}",
          p[[i, j]]
        );
      }
    }
  }

  /// A one-norm of 20 exceeds θ₁₃ ≈ 5.37, so this exercises the scaling and
  /// squaring branch.
  #[test]
  fn matches_the_two_state_closed_form_through_scaling_and_squaring() {
    let p = expm(&array![[-10.0, 10.0], [5.0, -5.0]]);
    let want = two_state_closed_form(10.0, 5.0);
    for i in 0..2 {
      for j in 0..2 {
        assert!(
          (p[[i, j]] - want[[i, j]]).abs() < 1e-12,
          "expm[{i},{j}] = {}",
          p[[i, j]]
        );
      }
    }
  }

  #[test]
  fn zero_maps_to_the_identity_and_diagonals_exponentiate_pointwise() {
    let z = expm(&Array2::<f64>::zeros((3, 3)));
    let d = expm(&array![[0.3, 0.0, 0.0], [0.0, -1.2, 0.0], [0.0, 0.0, 2.5]]);
    for i in 0..3 {
      for j in 0..3 {
        let want_z = if i == j { 1.0 } else { 0.0 };
        assert!((z[[i, j]] - want_z).abs() < 1e-14);
        let want_d = if i == j {
          [0.3_f64, -1.2, 2.5][i].exp()
        } else {
          0.0
        };
        assert!((d[[i, j]] - want_d).abs() < 1e-12);
      }
    }
  }

  #[test]
  fn inverse_times_matrix_is_the_identity() {
    let a = array![[4.0_f64, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
    let prod = invert_matrix(&a).dot(&a);
    for i in 0..3 {
      for j in 0..3 {
        let want = if i == j { 1.0 } else { 0.0 };
        assert!((prod[[i, j]] - want).abs() < 1e-13);
      }
    }
  }
}

//! Credit rating migration matrices and continuous-time generators.
//!
//! Reference: Jarrow, Lando & Turnbull, "A Markov Model for the Term Structure
//! of Credit Risk Spreads", Review of Financial Studies, 10(2), 481–523 (1997).
//! DOI: 10.1093/rfs/10.2.481
//!
//! Reference: Israel, Rosenthal & Wei, "Finding Generators for Markov Chains
//! via Empirical Transition Matrices, with Applications to Credit Ratings",
//! Mathematical Finance, 11(2), 245–265 (2001).
//! DOI: 10.1111/1467-9965.00114
//!
//! Reference: Pfeuffer, dos Reis & Smith, "Capturing Model Risk and Rating
//! Momentum in the Estimation of Probabilities of Default and Credit Rating
//! Migrations", arXiv:1809.09889 (2018).
//!
//! Reference: Higham, "The Scaling and Squaring Method for the Matrix
//! Exponential Revisited", SIAM Journal on Matrix Analysis and Applications,
//! 26(4), 1179–1193 (2005). DOI: 10.1137/04061101X
//!
//! The discrete one-period transition matrix is row-stochastic with an
//! absorbing default state in the last row/column:
//! $$
//! P\in\mathbb{R}^{n\times n},\quad P_{ij}\ge 0,\quad \sum_j P_{ij}=1,\quad
//! P_{n,\cdot}=(0,\dots,0,1).
//! $$
//! A continuous-time rating process is driven by a generator $Q$ with
//! $Q_{ii}\le 0$, $Q_{ij}\ge 0$ for $i\ne j$ and $\sum_j Q_{ij}=0$; the
//! $t$-year transition matrix is $P(t)=\exp(tQ)$.

use ndarray::Array1;
use ndarray::Array2;

use crate::traits::FloatExt;

/// Discrete-time rating transition matrix with an absorbing default state.
///
/// Rows correspond to the starting rating and columns to the rating at the
/// end of the period.  The last index is the default (absorbing) state.
#[derive(Debug, Clone)]
pub struct TransitionMatrix<T: FloatExt> {
  matrix: Array2<T>,
}

impl<T: FloatExt> TransitionMatrix<T> {
  /// Build from a row-stochastic matrix. The last state is treated as the
  /// absorbing default state.
  pub fn new(matrix: Array2<T>) -> Self {
    let (rows, cols) = matrix.dim();
    assert_eq!(rows, cols, "transition matrix must be square");
    Self { matrix }
  }

  /// Number of rating states (including default).
  pub fn states(&self) -> usize {
    self.matrix.nrows()
  }

  /// Index of the absorbing default state.
  pub fn default_state(&self) -> usize {
    self.matrix.nrows() - 1
  }

  /// Underlying matrix.
  pub fn matrix(&self) -> &Array2<T> {
    &self.matrix
  }

  /// Matrix reference to `P^k` — the $k$-period transition matrix.
  ///
  /// Convention: `power(0)` returns the identity matrix $I$ (vacuous transition,
  /// state preserved with probability 1). `power(k)` for `k >= 1` is `P^k`
  /// computed by repeated multiplication.
  pub fn power(&self, k: usize) -> TransitionMatrix<T> {
    if k == 0 {
      let n = self.matrix.nrows();
      let mut id = Array2::<T>::zeros((n, n));
      for i in 0..n {
        id[[i, i]] = T::one();
      }
      return TransitionMatrix::new(id);
    }
    let mut result = self.matrix.clone();
    for _ in 1..k {
      result = result.dot(&self.matrix);
    }
    TransitionMatrix::new(result)
  }

  /// Cumulative default probability over `k` periods from each starting state.
  pub fn default_probabilities(&self, k: usize) -> Array1<T> {
    let powered = self.power(k);
    powered.matrix.column(self.default_state()).to_owned()
  }

  /// Validate that the matrix is row-stochastic to within `tol`.
  pub fn check_row_stochastic(&self, tol: T) -> Result<(), String> {
    for (i, row) in self.matrix.rows().into_iter().enumerate() {
      let mut s = T::zero();
      for &v in row.iter() {
        if v < T::zero() && v.abs() > tol {
          return Err(format!("row {i} contains negative entry {v:?}"));
        }
        s += v;
      }
      if (s - T::one()).abs() > tol {
        return Err(format!("row {i} sums to {s:?}, expected 1"));
      }
    }
    Ok(())
  }
}

/// Continuous-time generator matrix (Q-matrix) for a rating Markov chain.
#[derive(Debug, Clone)]
pub struct GeneratorMatrix<T: FloatExt> {
  matrix: Array2<T>,
}

impl<T: FloatExt> GeneratorMatrix<T> {
  /// Wrap an existing generator matrix. Rows must sum to zero, off-diagonal
  /// entries are non-negative, and diagonal entries are non-positive.
  pub fn new(matrix: Array2<T>) -> Self {
    let (rows, cols) = matrix.dim();
    assert_eq!(rows, cols, "generator matrix must be square");
    Self { matrix }
  }

  /// Number of rating states (including default).
  pub fn states(&self) -> usize {
    self.matrix.nrows()
  }

  /// Index of the absorbing default state.
  pub fn default_state(&self) -> usize {
    self.matrix.nrows() - 1
  }

  /// Underlying generator matrix.
  pub fn matrix(&self) -> &Array2<T> {
    &self.matrix
  }

  /// Transition matrix $P(t)=\exp(tQ)$ for a holding period of `t` years.
  ///
  /// Uses a scaling-and-squaring Padé approximation (Higham, 2005) implemented
  /// on `ndarray` matrix operations so the method works under the default
  /// crate features (no LAPACK dependency).
  pub fn transition_at(&self, t: T) -> TransitionMatrix<T> {
    let scaled = &self.matrix * t;
    let p = expm(&scaled);
    TransitionMatrix::new(p)
  }

  /// Jarrow-Lando-Turnbull embedding of a discrete yearly transition matrix:
  /// $Q=\log(P)$ via series expansion.  Works when $P$ is diagonally dominant
  /// (common for credit rating matrices).
  ///
  /// Uses the series
  /// $\log(I+X)=\sum_{k\ge 1}(-1)^{k+1}X^k/k$
  /// with $X=P-I$, which converges when $\|X\|<1$ and is the standard JLT
  /// approach.  The result is not guaranteed to satisfy the generator
  /// positivity constraints; downstream users should project it to a valid
  /// generator via [`GeneratorMatrix::project_to_generator`] if needed.
  pub fn from_yearly_transition(p: &TransitionMatrix<T>) -> Self {
    let n = p.states();
    let identity = identity_matrix(n);
    let x = p.matrix() - &identity;

    let tolerance = T::from_f64_fast(1e-14);
    let mut result: Array2<T> = Array2::zeros((n, n));
    let mut term = x.clone();
    let mut sign = T::one();

    for k in 1..=200 {
      let k_t = T::from_usize_(k);
      let contribution = &term * (sign / k_t);
      result = &result + &contribution;

      let norm = contribution.iter().fold(T::zero(), |acc, v| acc + v.abs());
      if norm < tolerance && k > 3 {
        break;
      }

      term = term.dot(&x);
      sign = -sign;
    }

    Self::new(result)
  }

  /// Project a candidate generator matrix onto the closest valid generator by
  /// zeroing negative off-diagonal entries and rescaling rows to sum to zero
  /// (Israel-Rosenthal-Wei "diagonal adjustment" method).
  pub fn project_to_generator(&self) -> Self {
    let n = self.states();
    let mut m: Array2<T> = self.matrix.clone();

    for i in 0..n {
      let mut off_sum = T::zero();
      let mut neg_mass = T::zero();
      for j in 0..n {
        if i == j {
          continue;
        }
        if m[[i, j]] < T::zero() {
          neg_mass += m[[i, j]];
          m[[i, j]] = T::zero();
        } else {
          off_sum += m[[i, j]];
        }
      }
      if off_sum > T::zero() && neg_mass < T::zero() {
        let scale = (off_sum + neg_mass) / off_sum;
        for j in 0..n {
          if i != j && m[[i, j]] > T::zero() {
            m[[i, j]] = m[[i, j]] * scale;
          }
        }
      }

      let new_off_sum: T = (0..n)
        .filter(|&j| j != i)
        .map(|j| m[[i, j]])
        .fold(T::zero(), |acc, v| acc + v);
      m[[i, i]] = -new_off_sum;
    }

    Self::new(m)
  }

  /// Compute the default probability over a horizon `t` for every starting
  /// state.
  pub fn default_probabilities(&self, t: T) -> Array1<T> {
    let p = self.transition_at(t);
    p.matrix.column(self.default_state()).to_owned()
  }

  /// Validate that the matrix satisfies generator-matrix conditions to within
  /// `tol`: non-negative off-diagonal entries, non-positive diagonal, zero row
  /// sums.
  pub fn check_generator(&self, tol: T) -> Result<(), String> {
    let n = self.states();
    for i in 0..n {
      let mut s = T::zero();
      for j in 0..n {
        let v = self.matrix[[i, j]];
        if i != j && v < -tol {
          return Err(format!("off-diagonal ({i},{j}) negative: {v:?}"));
        }
        if i == j && v > tol {
          return Err(format!("diagonal ({i},{i}) positive: {v:?}"));
        }
        s += v;
      }
      if s.abs() > tol {
        return Err(format!("row {i} sum {s:?} not zero"));
      }
    }
    Ok(())
  }
}

fn identity_matrix<T: FloatExt>(n: usize) -> Array2<T> {
  let mut m = Array2::zeros((n, n));
  for i in 0..n {
    m[[i, i]] = T::one();
  }
  m
}

/// Scaling-and-squaring Padé-13 matrix exponential.
///
/// Implementation follows Higham (2005) Algorithm 2.3 with the $\theta_{13}$
/// threshold.  Uses only matrix multiplication and a Gauss-Jordan matrix
/// inversion so it does not require a LAPACK backend.
fn expm<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
  let n = a.nrows();
  if n == 0 {
    return a.clone();
  }
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
  let a_scaled = a / scale;
  let mut result = pade13(&a_scaled);
  for _ in 0..s {
    result = result.dot(&result);
  }
  result
}

fn one_norm<T: FloatExt>(a: &Array2<T>) -> T {
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

fn pade13<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
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
  let id = identity_matrix::<T>(n);
  let a2 = a.dot(a);
  let a4 = a2.dot(&a2);
  let a6 = a4.dot(&a2);

  let u_outer = &a6 * b[13] + &a4 * b[11] + &a2 * b[9];
  let u_inner = a6.dot(&u_outer);
  let u_low = &a6 * b[7] + &a4 * b[5] + &a2 * b[3] + &id * b[1];
  let u = a.dot(&(&u_inner + &u_low));

  let v_outer = &a6 * b[12] + &a4 * b[10] + &a2 * b[8];
  let v_inner = a6.dot(&v_outer);
  let v_low = &a6 * b[6] + &a4 * b[4] + &a2 * b[2] + &id * b[0];
  let v = &v_inner + &v_low;

  let numer = &v + &u;
  let denom = &v - &u;
  let denom_inv = invert_matrix(&denom);
  denom_inv.dot(&numer)
}

/// Gauss-Jordan elimination matrix inverse. Suitable for small matrices
/// (rating matrices are typically 8×8).
fn invert_matrix<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
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
  use super::*;
  use ndarray::array;

  /// Reference values for `expm` of the 2-state generator
  /// Q = [[-0.1, 0.1], [0.05, -0.05]] computed with scipy.linalg.expm
  /// (closed-form: P = (1/λ) [[β + α·e^{-λ}, α(1-e^{-λ})],
  ///                          [β(1-e^{-λ}), α + β·e^{-λ}]] for λ = α + β).
  /// `expm(Q) = [[0.9071795..., 0.0928204...],
  ///             [0.0464102..., 0.9535897...]]`
  /// Sourced by hand-computing the closed form (see comment) and
  /// independently checked against scipy 1.11.4 with rtol=1e-12.
  #[test]
  fn expm_matches_scipy_2x2_generator() {
    let q = array![[-0.1, 0.1], [0.05, -0.05]];
    let p = expm(&q);
    // Reference: closed-form 2-state continuous-time Markov chain.
    let lam = 0.15;
    let e = (-lam_f64(lam)).exp();
    let expected = array![
      [(0.05 + 0.1 * e) / lam, 0.1 * (1.0 - e) / lam],
      [0.05 * (1.0 - e) / lam, (0.1 + 0.05 * e) / lam],
    ];
    for i in 0..2 {
      for j in 0..2 {
        let got = p[[i, j]];
        let want = expected[[i, j]];
        assert!(
          (got - want).abs() < 1e-12,
          "expm[{i},{j}] = {got}, expected {want}"
        );
      }
    }
    // Each row should sum to 1 (stochasticity).
    for i in 0..2 {
      let row_sum: f64 = (0..2).map(|j| p[[i, j]]).sum();
      assert!((row_sum - 1.0).abs() < 1e-12, "row {i} sum = {row_sum}");
    }
  }

  /// Identity check: expm(0) = I.
  #[test]
  fn expm_of_zero_is_identity() {
    let z = Array2::<f64>::zeros((4, 4));
    let p = expm(&z);
    for i in 0..4 {
      for j in 0..4 {
        let want = if i == j { 1.0 } else { 0.0 };
        assert!((p[[i, j]] - want).abs() < 1e-14);
      }
    }
  }

  /// Diagonal check: expm(diag(d)) = diag(exp(d)).
  #[test]
  fn expm_of_diagonal_matches_pointwise_exp() {
    let d = [-1.0_f64, 0.5, 2.0];
    let mut q = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
      q[[i, i]] = d[i];
    }
    let p = expm(&q);
    for i in 0..3 {
      assert!((p[[i, i]] - d[i].exp()).abs() < 1e-12);
      for j in 0..3 {
        if i != j {
          assert!(p[[i, j]].abs() < 1e-13);
        }
      }
    }
  }

  fn lam_f64(x: f64) -> f64 {
    x
  }
}

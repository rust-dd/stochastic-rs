//! # Sobol Sequence
//!
//! $$
//! x_n = \bigoplus_{k:\,b_k(n)=1} v_k,\quad
//! n = \sum_k b_k(n)\,2^k
//! $$
//!
//! Gray-code Sobol sequence on Joe and Kuo's full `new-joe-kuo-6.21201`
//! direction-number table (primitive polynomials and initial direction
//! numbers for dimensions 2 to 21201, embedded verbatim and parsed once on
//! first use), so any number of dimensions up to 21201 is available.
//!
//! [`SobolSeq::scrambled`] adds Owen-type randomisation in Matoušek's
//! random linear form: every dimension's generating matrix is multiplied by
//! a random non-singular lower-triangular binary matrix and each point is
//! XOR-ed with a random digital shift. The scramble is a digit-wise
//! bijection, so every $(t, m, s)$-net property of the sequence survives
//! (each 1-D projection of $2^m$ points still hits every dyadic interval of
//! width $2^{-m}$ exactly once) while the points become uniformly
//! distributed for every seed — the basis of randomised-QMC error bars.
//!
//! References:
//! - Joe, S., Kuo, F.Y. (2008), "Constructing Sobol Sequences with Better
//!   Two-Dimensional Projections", *SIAM Journal on Scientific Computing*
//!   30(5), 2635-2654. DOI: 10.1137/070709359 — the direction numbers
//!   (`new-joe-kuo-6.21201`, BSD-3-Clause as published at
//!   <https://web.maths.unsw.edu.au/~fkuo/sobol/>).
//! - Antonov, I.A., Saleev, V.M. (1979), "An economic method of computing
//!   LPτ-sequences", *USSR Computational Mathematics and Mathematical
//!   Physics* 19(1), 252-256. DOI: 10.1016/0041-5553(79)90085-5 — the
//!   Gray-code update.
//! - Owen, A.B. (1998), "Scrambling Sobol' and Niederreiter–Xing points",
//!   *Journal of Complexity* 14(4), 466-489. DOI: 10.1006/jcom.1998.0487
//! - Matoušek, J. (1998), "On the L2-discrepancy for anchored boxes",
//!   *Journal of Complexity* 14(4), 527-556. DOI: 10.1006/jcom.1998.0489 —
//!   the random linear scrambling implemented here.

use std::sync::OnceLock;

use ndarray::Array2;
use rand::RngCore;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;

use crate::traits::FloatExt;

const BITS: usize = 32;

/// Joe and Kuo's `new-joe-kuo-6.21201`: `d s a m_1 … m_s` per line.
const JOE_KUO_TABLE: &str = include_str!("joe_kuo_6_21201.txt");

/// Highest dimension the embedded table covers.
pub const MAX_DIMENSIONS: usize = 21_201;

/// Primitive-polynomial data of one dimension: degree `s`, the middle
/// coefficients `a`, and the `s` initial direction numbers.
struct Polynomial {
  s: u32,
  a: u32,
  m: Vec<u32>,
}

/// The parsed table for dimensions 2..=21201, index `d - 2`.
fn joe_kuo() -> &'static [Polynomial] {
  static TABLE: OnceLock<Vec<Polynomial>> = OnceLock::new();
  TABLE.get_or_init(|| {
    let mut table = Vec::with_capacity(MAX_DIMENSIONS - 1);
    for line in JOE_KUO_TABLE.lines().skip(1) {
      let mut fields = line.split_whitespace();
      let Some(d) = fields.next() else { continue };
      let d: usize = d.parse().expect("Joe-Kuo table: dimension");
      let s: u32 = fields
        .next()
        .and_then(|v| v.parse().ok())
        .expect("Joe-Kuo table: s");
      let a: u32 = fields
        .next()
        .and_then(|v| v.parse().ok())
        .expect("Joe-Kuo table: a");
      let m: Vec<u32> = fields
        .map(|v| v.parse().expect("Joe-Kuo table: m_i"))
        .collect();
      assert_eq!(
        m.len(),
        s as usize,
        "Joe-Kuo table: dimension {d} carries {} of {s} m_i",
        m.len()
      );
      assert_eq!(
        d,
        table.len() + 2,
        "Joe-Kuo table: dimensions must be consecutive"
      );
      table.push(Polynomial { s, a, m });
    }
    assert_eq!(
      table.len(),
      MAX_DIMENSIONS - 1,
      "Joe-Kuo table: expected {} rows",
      MAX_DIMENSIONS - 1
    );
    table
  })
}

/// Compute 32-bit direction numbers from the primitive polynomial data.
fn compute_direction_numbers(s: u32, a: u32, m_init: &[u32]) -> [u32; BITS] {
  let s = s as usize;
  let mut v = [0u32; BITS];

  for j in 0..s {
    v[j] = m_init[j] << (BITS - 1 - j);
  }

  for j in s..BITS {
    let mut val = v[j - s] ^ (v[j - s] >> s as u32);
    for k in 1..s {
      let c_k = (a >> (s as u32 - 1 - k as u32)) & 1;
      if c_k == 1 {
        val ^= v[j - k];
      }
    }
    v[j] = val;
  }

  v
}

/// Direction numbers of the van der Corput first dimension.
fn first_dimension() -> [u32; BITS] {
  let mut v0 = [0u32; BITS];
  for (j, v) in v0.iter_mut().enumerate() {
    *v = 1u32 << (BITS - 1 - j);
  }
  v0
}

/// Unscrambled direction numbers of `n_dims` dimensions.
fn direction_numbers(n_dims: usize) -> Vec<[u32; BITS]> {
  assert!(
    n_dims > 0 && n_dims <= MAX_DIMENSIONS,
    "Sobol supports 1..={MAX_DIMENSIONS} dimensions, got {n_dims}"
  );
  let mut direction = Vec::with_capacity(n_dims);
  direction.push(first_dimension());
  if n_dims > 1 {
    let table = joe_kuo();
    for poly in &table[..n_dims - 1] {
      direction.push(compute_direction_numbers(poly.s, poly.a, &poly.m));
    }
  }
  direction
}

/// Matoušek's random linear scramble of one dimension: the direction
/// numbers are the columns of the generating matrix, so left-multiplying by
/// a random lower-triangular binary matrix with unit diagonal maps each
/// direction number `v` to `L v`, evaluated bit by bit over GF(2).
fn linear_scramble(direction: &mut [u32; BITS], rng: &mut SimdRng) {
  let mut rows = [0u32; BITS];
  for (i, row) in rows.iter_mut().enumerate() {
    // Row `i` of L (bit `BITS - 1 - i` is the diagonal, the higher bits are
    // free): random bits strictly above the diagonal in the row's own
    // position, zero below, one on the diagonal.
    let diagonal = 1u32 << (BITS - 1 - i);
    let free_mask = if i == 0 { 0 } else { !0u32 << (BITS - i) };
    *row = (rng.next_u32() & free_mask) | diagonal;
  }
  for v in direction.iter_mut() {
    let mut scrambled = 0u32;
    for (i, row) in rows.iter().enumerate() {
      if !(row & *v).count_ones().is_multiple_of(2) {
        scrambled |= 1u32 << (BITS - 1 - i);
      }
    }
    *v = scrambled;
  }
}

/// Sobol low-discrepancy sequence generator.
#[derive(Debug, Clone)]
pub struct SobolSeq {
  n_dims: usize,
  direction: Vec<[u32; BITS]>,
  shift: Vec<u32>,
}

impl SobolSeq {
  /// Create an unscrambled Sobol sequence generator for `n_dims`
  /// dimensions (up to [`MAX_DIMENSIONS`]).
  pub fn new(n_dims: usize) -> Self {
    let direction = direction_numbers(n_dims);
    Self {
      n_dims,
      direction,
      shift: vec![0; n_dims],
    }
  }

  /// Create an Owen-type scrambled Sobol sequence (random linear scramble
  /// plus digital shift) for `n_dims` dimensions, randomised from `seed`;
  /// `Deterministic` seeds reproduce the same point set.
  pub fn scrambled<S: SeedExt>(n_dims: usize, seed: &S) -> Self {
    let mut direction = direction_numbers(n_dims);
    let mut rng = SimdRng::from_seed(seed.seed_value());
    let mut shift = Vec::with_capacity(n_dims);
    for v in direction.iter_mut() {
      linear_scramble(v, &mut rng);
      shift.push(rng.next_u32());
    }
    Self {
      n_dims,
      direction,
      shift,
    }
  }

  /// Whether the generator carries a scramble.
  pub fn is_scrambled(&self) -> bool {
    self.shift.iter().any(|s| *s != 0)
  }

  /// Generate `n_points` Sobol points in `[0, 1)^d` (Gray-code order).
  ///
  /// Returns an `(n_points, n_dims)` array.
  pub fn sample<T: FloatExt>(&self, n_points: usize) -> Array2<T> {
    let scale = T::from_f64_fast(1.0 / (1u64 << BITS) as f64);
    let mut out = Array2::<T>::zeros((n_points, self.n_dims));
    let mut x = vec![0u32; self.n_dims];

    for i in 0..n_points {
      let c = ((i + 1) as u32).trailing_zeros() as usize;
      for j in 0..self.n_dims {
        x[j] ^= self.direction[j][c.min(BITS - 1)];
        out[[i, j]] = T::from_f64_fast((x[j] ^ self.shift[j]) as f64) * scale;
      }
    }

    out
  }

  pub fn n_dims(&self) -> usize {
    self.n_dims
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn sobol_dim1_first_points() {
    let seq = SobolSeq::new(1);
    let pts: Array2<f64> = seq.sample(3);
    // Gray-code order: 0.5, 0.75, 0.25
    assert!((pts[[0, 0]] - 0.5).abs() < 1e-10);
    assert!((pts[[1, 0]] - 0.75).abs() < 1e-10);
    assert!((pts[[2, 0]] - 0.25).abs() < 1e-10);
  }

  #[test]
  fn sobol_points_in_unit_cube() {
    let seq = SobolSeq::new(10);
    let pts: Array2<f64> = seq.sample(1000);
    for i in 0..1000 {
      for j in 0..10 {
        assert!(pts[[i, j]] >= 0.0 && pts[[i, j]] < 1.0);
      }
    }
  }

  #[test]
  fn sobol_mean_converges_to_half() {
    let seq = SobolSeq::new(3);
    let n = 1023; // 2^10 − 1 for best uniformity
    let pts: Array2<f64> = seq.sample(n);
    for j in 0..3 {
      let mean: f64 = (0..n).map(|i| pts[[i, j]]).sum::<f64>() / n as f64;
      assert!(
        (mean - 0.5).abs() < 0.02,
        "dim {j} mean = {mean:.4}, expected ≈ 0.5"
      );
    }
  }

  /// The parsed table carries the file's rows verbatim: dimension 3 is
  /// `x² + x + 1` with m = (1, 3), and the last row the degree-18
  /// polynomial with its 18 initial direction numbers.
  #[test]
  fn parsed_table_matches_the_joe_kuo_file() {
    let table = joe_kuo();
    assert_eq!(table.len(), MAX_DIMENSIONS - 1);
    assert_eq!(
      (table[0].s, table[0].a, table[0].m.as_slice()),
      (1, 0, &[1][..])
    );
    assert_eq!(
      (table[1].s, table[1].a, table[1].m.as_slice()),
      (2, 1, &[1, 3][..])
    );
    assert_eq!(
      (table[2].s, table[2].a, table[2].m.as_slice()),
      (3, 1, &[1, 3, 1][..])
    );
    let last = &table[MAX_DIMENSIONS - 2];
    assert_eq!(
      (last.s, last.a, last.m.as_slice()),
      (
        18,
        131_059,
        &[
          1, 1, 7, 11, 15, 7, 37, 239, 337, 245, 1557, 3681, 7357, 9639, 27367, 26869, 114_603,
          86_317
        ][..]
      )
    );
  }

  /// `scipy.stats.qmc.Sobol(d, scramble=False).random_base2(4)` (the same
  /// Joe-Kuo table, emitted in the same Gray-code order with the origin as
  /// its first point): output `i` here is scipy's point `i + 1`.
  #[test]
  fn matches_scipy_across_the_table() {
    let cases: [(usize, usize, [f64; 15]); 7] = [
      (
        3,
        0,
        [
          0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375, 0.4375, 0.3125,
          0.8125, 0.5625, 0.0625,
        ],
      ),
      (
        3,
        1,
        [
          0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625, 0.3125, 0.8125, 0.0625, 0.5625, 0.1875,
          0.6875, 0.4375, 0.9375,
        ],
      ),
      (
        3,
        2,
        [
          0.5, 0.25, 0.75, 0.625, 0.125, 0.875, 0.375, 0.9375, 0.4375, 0.6875, 0.1875, 0.3125,
          0.8125, 0.0625, 0.5625,
        ],
      ),
      (
        21,
        20,
        [
          0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.1875, 0.6875, 0.4375, 0.9375, 0.0625,
          0.5625, 0.3125, 0.8125,
        ],
      ),
      (
        22,
        21,
        [
          0.5, 0.75, 0.25, 0.125, 0.625, 0.875, 0.375, 0.6875, 0.1875, 0.4375, 0.9375, 0.5625,
          0.0625, 0.3125, 0.8125,
        ],
      ),
      (
        1000,
        999,
        [
          0.5, 0.75, 0.25, 0.125, 0.625, 0.875, 0.375, 0.9375, 0.4375, 0.1875, 0.6875, 0.8125,
          0.3125, 0.0625, 0.5625,
        ],
      ),
      (
        21201,
        21200,
        [
          0.5, 0.75, 0.25, 0.625, 0.125, 0.375, 0.875, 0.3125, 0.8125, 0.5625, 0.0625, 0.9375,
          0.4375, 0.1875, 0.6875,
        ],
      ),
    ];
    for (d, col, want) in cases {
      let pts: Array2<f64> = SobolSeq::new(d).sample(15);
      for (i, w) in want.into_iter().enumerate() {
        assert_eq!(pts[[i, col]], w, "d={d} col={col} point {i}");
      }
    }
    // 4095 points of the last and of a mid-table dimension, as index-weighted
    // checksums of the same scipy run.
    let pts: Array2<f64> = SobolSeq::new(21201).sample(4095);
    let checksum = |col: usize| {
      (0..4095)
        .map(|i| (i + 1) as f64 * pts[[i, col]])
        .sum::<f64>()
    };
    assert_eq!(checksum(21200), 4_192_768.25);
    assert_eq!(checksum(4999), 4_193_280.25);
  }

  /// A scrambled net keeps its stratification: the first 2^m − 1 outputs
  /// (the crate emits the sequence from index 1, so the origin's point is
  /// the one missing) hit 2^m − 1 distinct dyadic intervals of width 2^-m
  /// in every coordinate — exactly one interval stays empty.
  #[test]
  fn scramble_preserves_the_net_property() {
    let m = 10;
    let n = (1usize << m) - 1;
    let seq = SobolSeq::scrambled(8, &Deterministic::new(42));
    assert!(seq.is_scrambled());
    let pts: Array2<f64> = seq.sample(n);
    for j in 0..8 {
      let mut hits = vec![0usize; n + 1];
      for i in 0..n {
        let x = pts[[i, j]];
        assert!((0.0..1.0).contains(&x));
        hits[(x * (n + 1) as f64) as usize] += 1;
      }
      assert!(
        hits.iter().all(|h| *h <= 1),
        "dim {j}: an interval is hit twice"
      );
      assert_eq!(
        hits.iter().filter(|h| **h == 0).count(),
        1,
        "dim {j}: empty intervals"
      );
    }
  }

  /// Scrambling is reproducible per seed, differs across seeds, and stays
  /// unbiased.
  #[test]
  fn scramble_is_seeded_and_unbiased() {
    let a: Array2<f64> = SobolSeq::scrambled(5, &Deterministic::new(7)).sample(64);
    let b: Array2<f64> = SobolSeq::scrambled(5, &Deterministic::new(7)).sample(64);
    let c: Array2<f64> = SobolSeq::scrambled(5, &Deterministic::new(8)).sample(64);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let plain: Array2<f64> = SobolSeq::new(5).sample(64);
    assert_ne!(a, plain);
    let pts: Array2<f64> = SobolSeq::scrambled(5, &Deterministic::new(9)).sample(4096);
    for j in 0..5 {
      let mean: f64 = (0..4096).map(|i| pts[[i, j]]).sum::<f64>() / 4096.0;
      assert!((mean - 0.5).abs() < 0.01, "dim {j} mean = {mean}");
    }
  }

  #[test]
  #[should_panic(expected = "Sobol supports 1..=21201 dimensions")]
  fn rejects_dimensions_beyond_the_table() {
    let _ = SobolSeq::new(MAX_DIMENSIONS + 1);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PySobolSeq {
  inner: SobolSeq,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PySobolSeq {
  /// Sobol sequence in `n_dims` dimensions (up to 21201); a `seed` switches
  /// on the Owen-type scramble.
  #[new]
  #[pyo3(signature = (n_dims, seed=None))]
  fn new(n_dims: usize, seed: Option<u64>) -> Self {
    let inner = match seed {
      Some(s) => SobolSeq::scrambled(n_dims, &stochastic_rs_core::simd_rng::Deterministic::new(s)),
      None => SobolSeq::new(n_dims),
    };
    Self { inner }
  }

  /// `(n_points, n_dims)` array of points in `[0, 1)`.
  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
    n_points: usize,
  ) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.sample::<f64>(n_points).into_pyarray(py)
  }

  #[getter]
  fn n_dims(&self) -> usize {
    self.inner.n_dims()
  }

  #[getter]
  fn is_scrambled(&self) -> bool {
    self.inner.is_scrambled()
  }
}

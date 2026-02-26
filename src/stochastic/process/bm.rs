//! # BM
//!
//! $$
//! B_t=\int_0^t dW_s,\quad B_t-B_s\sim\mathcal N(0,t-s)
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct BM<T: FloatExt> {
  /// Number of discrete time points in the generated path.
  pub n: usize,
  /// Total simulation horizon (defaults to `1` if `None`).
  pub t: Option<T>,
}

impl<T: FloatExt> BM<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Self { n, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for BM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut bm = Array1::<T>::zeros(self.n);
    if self.n <= 1 {
      return bm;
    }

    let n_increments = self.n - 1;
    let std_dev = (self.t.unwrap_or(T::one()) / T::from_usize_(n_increments)).sqrt();
    let mut tail_view = bm.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("BM output tail must be contiguous");
    T::fill_standard_normal_scaled_slice(tail, std_dev);

    let mut acc = T::zero();
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }

    bm
  }
}

py_process_1d!(PyBM, BM,
  sig: (n, t=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use std::time::Instant;

  use super::*;

  #[test]
  fn test_bm() {
    let start = Instant::now();
    let bm = BM::new(10000, Some(1.0));
    for _ in 0..10000 {
      let m = bm.sample();
      assert_eq!(m.len(), 10000);
    }
    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());

    let start = Instant::now();
    let bm = BM::new(10000, Some(1.0));
    for _ in 0..10000 {
      let m = bm.sample();
      assert_eq!(m.len(), 10000);
    }
    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());
  }

  #[test]
  fn test_bm_movement_1000_iterations() {
    let bm = BM::new(1000, Some(1.0));

    let mut max_abs_value: f64 = 0.0;
    let mut min_abs_value: f64 = f64::MAX;
    let mut last_value_sum: f64 = 0.0;

    for _ in 0..1000 {
      let path = bm.sample();
      assert_eq!(path.len(), 1000);

      let last_value = path[999];
      last_value_sum += last_value;

      let abs_last = last_value.abs();
      max_abs_value = max_abs_value.max(abs_last);
      min_abs_value = min_abs_value.min(abs_last);
    }

    let avg_last_value = last_value_sum / 1000.0;
    println!("BM Movement Test (1000 iterations):");
    println!("  Average last value: {}", avg_last_value);
    println!("  Maximum absolute last value: {}", max_abs_value);
    println!("  Minimum absolute last value: {}", min_abs_value);

    assert!(max_abs_value > 0.0, "BM should have non-zero movement");
    assert!(
      avg_last_value.abs() < 2.0,
      "Average position should stay relatively close to zero"
    );
  }
}

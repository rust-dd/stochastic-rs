//! # BM
//!
//! $$
//! B_t=\int_0^t dW_s,\quad B_t-B_s\sim\mathcal N(0,t-s)
//! $$
//!
use ndarray::s;
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct BM<T: FloatExt> {
  /// Number of discrete time points in the generated path.
  pub n: usize,
  /// Total simulation horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> BM<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Self {
      n,
      t,
      gn: Gn::new(n.saturating_sub(1), t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for BM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut bm = Array1::<T>::zeros(self.n);
    if self.n <= 1 {
      return bm;
    }

    let std_dev = self.gn.dt().sqrt();
    let mut tail_view = bm.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("BM output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    let mut acc = T::zero();
    for x in tail.iter_mut() {
      acc += *x * std_dev;
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
}

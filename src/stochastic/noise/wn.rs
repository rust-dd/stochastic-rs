//! # WN
//!
//! $$
//! \xi_i\stackrel{iid}{\sim}\mathcal N(0,1)
//! $$
//!
use ndarray::Array1;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Wn<T: FloatExt> {
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Target mean level for generated noise samples.
  pub mean: Option<T>,
  /// Standard deviation of generated noise samples.
  pub std_dev: Option<T>,
}

impl<T: FloatExt> Wn<T> {
  pub fn new(n: usize, mean: Option<T>, std_dev: Option<T>) -> Self {
    Wn { n, mean, std_dev }
  }
}

impl<T: FloatExt> ProcessExt<T> for Wn<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    T::normal_array(
      self.n,
      self.mean.unwrap_or(T::zero()),
      self.std_dev.unwrap_or(T::one()),
    )
  }
}

py_process_1d!(PyWn, Wn,
  sig: (n, mean=None, std_dev=None, dtype=None),
  params: (n: usize, mean: Option<f64>, std_dev: Option<f64>)
);

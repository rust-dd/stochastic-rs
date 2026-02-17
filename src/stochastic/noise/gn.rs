use ndarray::Array1;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Gn<T: FloatExt> {
  pub n: usize,
  pub t: Option<T>,
}

impl<T: FloatExt> Gn<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Gn { n, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for Gn<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    let out_slice = out.as_slice_mut().expect("Gn output must be contiguous");
    self.fill_slice(out_slice);
    out
  }
}

impl<T: FloatExt> Gn<T> {
  pub fn fill_slice(&self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    T::fill_standard_normal_slice(&mut out[..len]);
    let std_dev = self.dt().sqrt();
    for x in out[..len].iter_mut() {
      *x = *x * std_dev;
    }
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

py_process_1d!(PyGn, Gn,
  sig: (n, t=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);

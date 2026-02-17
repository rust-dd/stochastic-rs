use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct BM<T: FloatExt> {
  pub n: usize,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> BM<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Self {
      n,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for BM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let gn = &self.gn.sample();
    let mut bm = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      bm[i] = bm[i - 1] + gn[i - 1];
    }

    bm
  }
}

py_process_1d!(PyBM, BM,
  sig: (n, t=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);

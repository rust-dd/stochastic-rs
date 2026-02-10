use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct BM<T: Float> {
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
  gn: Gn<T>,
}

impl<T: Float> Process<T> for BM<T> {
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

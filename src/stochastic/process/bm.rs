use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct BM<T: Float> {
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
}

impl<T: Float> Process<T> for BM<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(&self, noise_fn: impl FnOnce(&Self::Noise) -> Self::Output) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let gn = noise_fn(&gn);
    let mut bm = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      bm[i] += bm[i - 1] + gn[i - 1];
    }

    bm
  }
}

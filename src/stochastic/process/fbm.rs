use ndarray::Array1;
use statrs::function::gamma;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FBM<T: FloatExt> {
  pub hurst: T,
  pub n: usize,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> FBM<T> {
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    Self {
      hurst,
      n,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FBM<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let fgn = &self.fgn.sample();
    let mut fbm = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    fbm
  }
}

impl<T: FloatExt> FBM<T> {
  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  pub fn malliavin(&self) -> Array1<T> {
    let dt = self.fgn.dt();
    let mut m = Array1::zeros(self.n);
    let g = gamma::gamma(self.hurst.to_f64().unwrap() + 0.5);

    for i in 0..self.n {
      m[i] = T::one() / T::from_f64_fast(g)
        * (T::from_usize_(i) * dt).powf(self.hurst - T::from_f64_fast(0.5));
    }

    m
  }
}

#[cfg(test)]
mod tests {
  use std::time::Instant;

  use super::*;

  #[test]
  fn test_fbm() {
    let start = Instant::now();
    let fbm = FBM::new(0.7, 10000, None);
    for _ in 0..10000 {
      let m = fbm.sample();
      assert_eq!(m.len(), 10000);
    }
    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());
  }
}

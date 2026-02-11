use ndarray::Array1;
use statrs::function::gamma;

use crate::f;
use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct FBM<T: Float> {
  pub hurst: T,
  pub n: usize,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: Float> FBM<T> {
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    Self {
      hurst,
      n,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FBM<T> {
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

impl<T: Float> FBM<T> {
  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  fn malliavin(&self) -> Array1<T> {
    let dt = self.fgn.dt();
    let mut m = Array1::zeros(self.n);

    for i in 0..self.n {
      m[i] = f!(1) / f!(gamma::gamma(self.hurst.to_f64().unwrap() + 0.5))
        * (f!(i) * dt).powf(self.hurst - f!(0.5));
    }

    m
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::plot_2d;
  use crate::stochastic::N;

  #[test]
  fn fbm_length_equals_n() {
    let fbm = FBM::new(0.7, N, Some(1.0));
    assert_eq!(fbm.sample().len(), N);
  }

  #[test]
  fn fbm_starts_with_x0() {
    let fbm = FBM::new(0.7, N, Some(1.0));
    assert_eq!(fbm.sample()[0], 0.0);
  }

  #[test]
  fn fbm_plot() {
    let fbm = FBM::new(0.1, N, Some(1.0));
    plot_1d!(fbm.sample(), "Fractional Brownian Motion (H = 0.7)");
  }

  #[test]
  fn fbm_malliavin() {
    let fbm = FBM::new(0.7, N, Some(1.0));
    let process = fbm.sample();
    let malliavin = fbm.malliavin();
    plot_2d!(
      process,
      "Fractional Brownian Motion (H = 0.7)",
      malliavin,
      "Malliavin derivative of Fractional Brownian Motion (H = 0.7)"
    );
  }
}

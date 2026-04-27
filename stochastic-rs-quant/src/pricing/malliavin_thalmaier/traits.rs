//! Generic multi-asset stochastic volatility trait.
//!
//! Any SV model (Heston, Sabr, 3/2, rough Bergomi, …) can implement
//! [`MultiSvPaths`] to plug into the M-T Greeks engine.

use ndarray::Array2;

use crate::traits::FloatExt;

/// Generic sampled paths from a multi-asset SV model.
pub trait MultiSvPaths<T: FloatExt> {
  fn n_assets(&self) -> usize;
  fn n_steps(&self) -> usize;
  fn terminal_price(&self, asset: usize) -> T;
  fn variance(&self, asset: usize, step: usize) -> T;
  fn price(&self, asset: usize, step: usize) -> T;
}

impl<T: FloatExt> MultiSvPaths<T> for super::heston::MultiHestonPaths<T> {
  fn n_assets(&self) -> usize {
    self.n_assets
  }

  fn n_steps(&self) -> usize {
    self.n_steps
  }

  fn terminal_price(&self, asset: usize) -> T {
    self.prices[[asset, self.n_steps - 1]]
  }

  fn variance(&self, asset: usize, step: usize) -> T {
    self.vols[[asset, step]]
  }

  fn price(&self, asset: usize, step: usize) -> T {
    self.prices[[asset, step]]
  }
}

/// Malliavin covariance matrix from generic paths.
pub fn malliavin_cov_generic<T: FloatExt>(
  paths: &dyn MultiSvPaths<T>,
  cross_corr: &Array2<T>,
  tau: T,
) -> Array2<T> {
  let d = paths.n_assets();
  let n = paths.n_steps();
  let dt = tau / T::from_usize_(n - 1);

  let mut gamma = Array2::<T>::zeros((d, d));
  for i in 0..d {
    for j in 0..d {
      let st_i = paths.terminal_price(i);
      let st_j = paths.terminal_price(j);
      let integral: T = (0..n - 1)
        .map(|k| {
          (paths.variance(i, k).max(T::zero()) * paths.variance(j, k).max(T::zero())).sqrt() * dt
        })
        .fold(T::zero(), |a, b| a + b);
      gamma[[i, j]] = st_i * st_j * cross_corr[[i, j]] * integral;
    }
  }
  gamma
}

/// Itô integral `∫√V_j dWⱼˢ` from generic paths.
pub fn ito_integral_generic<T: FloatExt>(
  paths: &dyn MultiSvPaths<T>,
  asset: usize,
  r: T,
  tau: T,
) -> T {
  let n = paths.n_steps();
  let dt = tau / T::from_usize_(n - 1);
  (0..n - 1)
    .map(|k| {
      let s = paths.price(asset, k);
      if s.abs() > T::from_f64_fast(1e-14) {
        paths.price(asset, k + 1) / s - T::one() - r * dt
      } else {
        T::zero()
      }
    })
    .fold(T::zero(), |a, b| a + b)
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::*;
  use crate::pricing::malliavin_thalmaier::heston::*;

  #[test]
  fn generic_trait_works() {
    let a = AssetParams {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
    };
    let mut cross = Array2::<f64>::eye(2);
    cross[[0, 1]] = 0.5;
    cross[[1, 0]] = 0.5;
    let p = MultiHestonParams {
      assets: vec![a.clone(), a],
      cross_corr: cross.clone(),
      r: 0.05,
      tau: 1.0,
      n_steps: 100,
    };
    let paths = p.sample();

    let gamma = malliavin_cov_generic(&paths as &dyn MultiSvPaths<f64>, &cross, 1.0);
    assert!(gamma[[0, 0]] > 0.0);

    let ito = ito_integral_generic(&paths as &dyn MultiSvPaths<f64>, 0, 0.05, 1.0);
    assert!(ito.is_finite());
  }
}

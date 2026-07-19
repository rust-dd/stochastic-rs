//! Generic multi-asset path access for Malliavin–Thalmaier utilities.
//!
//! Implementing [`MultiSvPaths`] supplies path data only. Exact conditional
//! Malliavin weights additionally require volatility paths independent of the
//! price Brownian motions and an exponential log-price update.

use ndarray::Array2;

use crate::traits::FloatExt;

/// Read-only access to sampled multi-asset stochastic-volatility paths.
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

/// Conditional Malliavin covariance matrix from generic paths.
///
/// This is exact when the variance paths are independent of the price noise.
/// `impl Trait` lets the call site inline the per-step accessors.
pub fn malliavin_cov_generic<T: FloatExt>(
  paths: &(impl MultiSvPaths<T> + ?Sized),
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
      let integral = (0..n - 1)
        .map(|k| {
          (paths.variance(i, k).max(T::zero()) * paths.variance(j, k).max(T::zero())).sqrt() * dt
        })
        .fold(T::zero(), |a, b| a + b);
      gamma[[i, j]] = st_i * st_j * cross_corr[[i, j]] * integral;
    }
  }
  gamma
}

/// Reconstruct `integral sqrt(V_j) dW_j^S` from generic log-Euler paths.
///
/// The reconstruction is exact for the exponential update
/// `S_{k+1}/S_k = exp((r - V_k/2) dt + sqrt(V_k) dW_k)`.
pub fn ito_integral_generic<T: FloatExt>(
  paths: &(impl MultiSvPaths<T> + ?Sized),
  asset: usize,
  r: T,
  tau: T,
) -> T {
  let n = paths.n_steps();
  let dt = tau / T::from_usize_(n - 1);
  (0..n - 1)
    .map(|k| {
      let s = paths.price(asset, k);
      let next = paths.price(asset, k + 1);
      if s > T::zero() && next > T::zero() {
        let variance = paths.variance(asset, k).max(T::zero());
        (next / s).ln() - (r - T::from_f64_fast(0.5) * variance) * dt
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
      rho: 0.0,
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
    let paths = p.sample_with_seed(0x7a_17_01);

    let gamma = malliavin_cov_generic(&paths, &cross, 1.0);
    assert!(gamma[[0, 0]] > 0.0);

    let ito = ito_integral_generic(&paths, 0, 0.05, 1.0);
    assert!(ito.is_finite());
  }
}

//! Multi-asset Heston simulation for M-T Greeks.
//!
//! See Malliavin & Thalmaier (2006), Chapter 4.

use ndarray::Array2;

use crate::simd_rng::SeedExt;
use crate::stochastic::volatility::heston::Heston;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Parameters for a single asset in a multi-asset Heston model.
#[derive(Clone, Debug)]
pub struct AssetParams<T: FloatExt> {
  pub s0: T,
  pub v0: T,
  pub kappa: T,
  pub theta: T,
  pub xi: T,
  /// Within-asset price–variance correlation ρ.
  pub rho: T,
}

/// Multi-asset Heston model.
///
/// ```text
/// dSᵢ = r·Sᵢ·dt + √Vᵢ·Sᵢ·dWᵢˢ
/// dVᵢ = κᵢ(θᵢ − Vᵢ)dt + ξᵢ√Vᵢ·(ρᵢ·dWᵢˢ + √(1−ρᵢ²)·dWᵢᵛ)
/// ```
///
/// Cross-asset correlation: `Cor(Wᵢˢ, Wⱼˢ) = cross_corr[[i,j]]`.
#[derive(Clone, Debug)]
pub struct MultiHestonParams<T: FloatExt> {
  pub assets: Vec<AssetParams<T>>,
  pub cross_corr: Array2<T>,
  pub r: T,
  pub tau: T,
  pub n_steps: usize,
}

/// Simulated paths from a multi-asset Heston model.
pub struct MultiHestonPaths<T: FloatExt> {
  pub prices: Array2<T>,
  pub vols: Array2<T>,
  pub n_assets: usize,
  pub n_steps: usize,
}

impl<T: FloatExt> MultiHestonParams<T> {
  pub fn n_assets(&self) -> usize {
    self.assets.len()
  }

  /// Simulate one set of paths. Requires LAPACK for Cholesky.
  pub fn sample(&self) -> MultiHestonPaths<T>
  where
    T: ndarray_linalg::Lapack,
  {
    let d = self.n_assets();
    let n = self.n_steps;
    let dt = self.tau / T::from_usize_(n - 1);
    let sqrt_dt = <T as num_traits::Float>::sqrt(dt);
    let m = 2 * d;

    let chol = self.brownian_cholesky();

    let mut prices = Array2::<T>::zeros((d, n));
    let mut vols = Array2::<T>::zeros((d, n));
    for i in 0..d {
      prices[[i, 0]] = self.assets[i].s0;
      vols[[i, 0]] = self.assets[i].v0.max(T::zero());
    }

    for step in 1..n {
      let mut z_ind = vec![T::zero(); m];
      T::fill_standard_normal_slice(&mut z_ind);

      let mut db = vec![T::zero(); m];
      for i in 0..m {
        let mut s = T::zero();
        for j in 0..=i {
          s = s + chol[[i, j]] * z_ind[j];
        }
        db[i] = s * sqrt_dt;
      }

      for i in 0..d {
        let v_prev = vols[[i, step - 1]].max(T::zero());
        let sqrt_v = <T as num_traits::Float>::sqrt(v_prev);
        prices[[i, step]] = prices[[i, step - 1]]
          + self.r * prices[[i, step - 1]] * dt
          + sqrt_v * prices[[i, step - 1]] * db[i];
        let dv = self.assets[i].kappa * (self.assets[i].theta - v_prev) * dt
          + self.assets[i].xi * sqrt_v * db[d + i];
        vols[[i, step]] = (v_prev + dv).max(T::zero());
      }
    }

    MultiHestonPaths {
      prices,
      vols,
      n_assets: d,
      n_steps: n,
    }
  }

  fn brownian_cholesky(&self) -> Array2<T>
  where
    T: ndarray_linalg::Lapack,
  {
    use ndarray_linalg::Cholesky;
    use ndarray_linalg::UPLO;

    let d = self.n_assets();
    let m = 2 * d;
    let mut corr = Array2::<T>::eye(m);
    for i in 0..d {
      for j in 0..d {
        corr[[i, j]] = self.cross_corr[[i, j]];
      }
    }
    for i in 0..d {
      corr[[i, d + i]] = self.assets[i].rho;
      corr[[d + i, i]] = self.assets[i].rho;
    }
    corr
      .cholesky(UPLO::Lower)
      .expect("correlation matrix not positive-definite")
  }
}

impl<T: FloatExt> MultiHestonPaths<T> {
  /// Build from existing single-asset `Heston` instances.
  ///
  /// Each `Heston` is sampled independently — use this when cross-asset
  /// correlation is zero. For correlated assets use
  /// [`MultiHestonParams::sample`] instead.
  pub fn from_hestons<S: SeedExt>(hestons: &[Heston<T, S>]) -> Self {
    let d = hestons.len();
    assert!(d > 0, "need at least one Heston instance");
    let n = hestons[0].n;

    let mut prices = Array2::<T>::zeros((d, n));
    let mut vols = Array2::<T>::zeros((d, n));

    for (i, h) in hestons.iter().enumerate() {
      assert_eq!(h.n, n, "all Heston instances must have the same n_steps");
      let [s, v] = h.sample();
      for j in 0..n {
        prices[[i, j]] = s[j];
        vols[[i, j]] = v[j];
      }
    }

    Self { prices, vols, n_assets: d, n_steps: n }
  }

  pub fn terminal_prices(&self) -> Vec<T> {
    (0..self.n_assets)
      .map(|i| self.prices[[i, self.n_steps - 1]])
      .collect()
  }

  /// Malliavin covariance matrix `γ_F` (d × d).
  pub fn malliavin_cov(&self, cross_corr: &Array2<T>, tau: T) -> Array2<T> {
    let d = self.n_assets;
    let n = self.n_steps;
    let dt = tau / T::from_usize_(n - 1);
    let st = self.terminal_prices();

    let mut g = Array2::<T>::zeros((d, d));
    for i in 0..d {
      for j in 0..d {
        let mut integral = T::zero();
        for k in 0..(n - 1) {
          integral = integral
            + (self.vols[[i, k]].max(T::zero()) * self.vols[[j, k]].max(T::zero())).sqrt() * dt;
        }
        g[[i, j]] = st[i] * st[j] * cross_corr[[i, j]] * integral;
      }
    }
    g
  }

  /// Stochastic integral `I_j = ∫₀ᵀ √V_j dW_jˢ` reconstructed from prices.
  pub fn ito_integral(&self, asset: usize, r: T, tau: T) -> T {
    let n = self.n_steps;
    let dt = tau / T::from_usize_(n - 1);
    let mut sum = T::zero();
    for k in 0..(n - 1) {
      let s_prev = self.prices[[asset, k]];
      if s_prev.abs() > T::from_f64_fast(1e-14) {
        sum = sum + self.prices[[asset, k + 1]] / s_prev - T::one() - r * dt;
      }
    }
    sum
  }

  /// Malliavin weight vector `H_{(i)}` for Delta of asset `p`.
  pub fn malliavin_weights(
    &self,
    gamma_inv: &Array2<T>,
    param_asset: usize,
    r: T,
    tau: T,
    spots: &[T],
  ) -> Vec<T> {
    let d = self.n_assets;
    let st = self.terminal_prices();
    let tangent = st[param_asset] / spots[param_asset];
    let ito: Vec<T> = (0..d).map(|j| self.ito_integral(j, r, tau)).collect();

    (0..d)
      .map(|i| {
        let s: T = (0..d)
          .map(|j| gamma_inv[[i, j]] * st[j] * ito[j])
          .fold(T::zero(), |a, b| a + b);
        tangent * s
      })
      .collect()
  }

  /// Inverse of the Malliavin covariance matrix. Requires LAPACK.
  pub fn gamma_inv(&self, cross_corr: &Array2<T>, tau: T) -> Array2<T>
  where
    T: ndarray_linalg::Lapack,
  {
    use ndarray_linalg::Inverse;
    self
      .malliavin_cov(cross_corr, tau)
      .inv()
      .expect("Malliavin covariance matrix is singular")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn two_asset_params() -> MultiHestonParams<f64> {
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
    MultiHestonParams {
      assets: vec![a.clone(), a],
      cross_corr: cross,
      r: 0.05,
      tau: 1.0,
      n_steps: 100,
    }
  }

  #[test]
  fn simulation_positive_prices() {
    let st = two_asset_params().sample().terminal_prices();
    assert!(st[0] > 0.0, "S1_T = {}", st[0]);
    assert!(st[1] > 0.0, "S2_T = {}", st[1]);
  }

  #[test]
  fn malliavin_cov_positive_definite() {
    let p = two_asset_params();
    let paths = p.sample();
    let g = paths.malliavin_cov(&p.cross_corr, p.tau);
    let det = g[[0, 0]] * g[[1, 1]] - g[[0, 1]] * g[[1, 0]];
    assert!(det > 0.0, "det(γ) = {det}");
    assert!(g[[0, 0]] > 0.0);
  }

  #[test]
  fn cholesky_roundtrip() {
    use ndarray_linalg::Cholesky;
    use ndarray_linalg::UPLO;

    let mut a = Array2::<f64>::zeros((3, 3));
    a[[0, 0]] = 4.0;
    a[[0, 1]] = 2.0;
    a[[0, 2]] = 0.5;
    a[[1, 0]] = 2.0;
    a[[1, 1]] = 5.0;
    a[[1, 2]] = 1.0;
    a[[2, 0]] = 0.5;
    a[[2, 1]] = 1.0;
    a[[2, 2]] = 3.0;

    let l = a.cholesky(UPLO::Lower).unwrap();
    for i in 0..3 {
      for j in 0..3 {
        let s: f64 = (0..3).map(|k| l[[i, k]] * l[[j, k]]).sum();
        assert!(
          (s - a[[i, j]]).abs() < 1e-10,
          "LLᵀ[{i},{j}]={s} ≠ {}",
          a[[i, j]]
        );
      }
    }
  }

  #[test]
  fn invert_roundtrip() {
    use ndarray_linalg::Inverse;

    let mut a = Array2::<f64>::zeros((2, 2));
    a[[0, 0]] = 3.0;
    a[[0, 1]] = 1.0;
    a[[1, 0]] = 1.0;
    a[[1, 1]] = 2.0;
    let inv = a.inv().unwrap();
    for i in 0..2 {
      for j in 0..2 {
        let s: f64 = (0..2).map(|k| a[[i, k]] * inv[[k, j]]).sum();
        let e = if i == j { 1.0 } else { 0.0 };
        assert!((s - e).abs() < 1e-10, "AA⁻¹[{i},{j}]={s} ≠ {e}");
      }
    }
  }
}

//! Multi-asset Heston simulation for M-T Greeks.
//!
//! See Malliavin & Thalmaier (2006), Chapter 4.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;

use crate::QuantError;
use crate::error::check_grid_shape;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

mod weights;

/// Parameters for a single asset in a multi-asset Heston model.
#[derive(Clone, Debug)]
pub struct AssetParams<T: FloatExt> {
  pub s: T,
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
#[derive(Debug, Clone)]
pub struct MultiHestonPaths<T: FloatExt> {
  pub prices: Array2<T>,
  pub vols: Array2<T>,
  pub n_assets: usize,
  pub n_steps: usize,
  conditional_weights_exact: bool,
}

impl<T: FloatExt> MultiHestonParams<T> {
  pub fn n_assets(&self) -> usize {
    self.assets.len()
  }

  /// Validate model, time-grid and joint-correlation inputs, which [`Self::sample`] needs for
  /// user-supplied parameters and [`Self::try_sample`] runs itself.
  pub fn validate(&self) -> Result<(), QuantError> {
    const CORR: &str = "cross-asset correlation";
    const ASSETS: &str = "asset parameters";
    let d = self.n_assets();
    if d == 0 {
      return Err(QuantError::invalid(
        ASSETS,
        "need at least one asset, got none",
      ));
    }
    if self.n_steps < 2 {
      return Err(QuantError::invalid(
        "n_steps",
        format!("must be >= 2, got {}", self.n_steps),
      ));
    }
    if !num_traits::Float::is_finite(self.r) {
      return Err(QuantError::invalid(
        "r",
        format!("must be finite, got {:?}", self.r),
      ));
    }
    if !num_traits::Float::is_finite(self.tau) || self.tau <= T::zero() {
      return Err(QuantError::invalid(
        "tau",
        format!("must be finite and positive, got {:?}", self.tau),
      ));
    }
    check_grid_shape(
      [
        "cross_corr rows, one per asset",
        "cross_corr columns, one per asset",
      ],
      (d, d),
      self.cross_corr.dim(),
    )?;
    let one = T::one();
    let neg_one = -one;
    let tol = T::from_f64_fast(1e-10);
    for i in 0..d {
      for j in 0..d {
        let v = self.cross_corr[[i, j]];
        if !num_traits::Float::is_finite(v) {
          return Err(QuantError::invalid(
            CORR,
            format!("cross_corr[{i},{j}] must be finite, got {v:?}"),
          ));
        }
        if i == j {
          if num_traits::Float::abs(v - one) > tol {
            return Err(QuantError::invalid(
              CORR,
              format!("cross_corr[{i},{i}] = {v:?} is not 1"),
            ));
          }
        } else {
          if v < neg_one || v > one {
            return Err(QuantError::invalid(
              CORR,
              format!("cross_corr[{i},{j}] = {v:?} is outside [-1, 1]"),
            ));
          }
          let v_sym = self.cross_corr[[j, i]];
          if num_traits::Float::abs(v - v_sym) > tol {
            return Err(QuantError::invalid(
              CORR,
              format!("cross_corr is not symmetric at ({i},{j}): {v:?} against {v_sym:?}"),
            ));
          }
        }
      }
    }
    for (i, a) in self.assets.iter().enumerate() {
      if !num_traits::Float::is_finite(a.s) || a.s <= T::zero() {
        return Err(QuantError::invalid(
          ASSETS,
          format!("assets[{i}].s must be finite and positive, got {:?}", a.s),
        ));
      }
      for (name, value) in [
        ("v0", a.v0),
        ("kappa", a.kappa),
        ("theta", a.theta),
        ("xi", a.xi),
      ] {
        if !num_traits::Float::is_finite(value) || value < T::zero() {
          return Err(QuantError::invalid(
            ASSETS,
            format!("assets[{i}].{name} must be finite and non-negative, got {value:?}"),
          ));
        }
      }
      if !num_traits::Float::is_finite(a.rho) || num_traits::Float::abs(a.rho) >= one {
        return Err(QuantError::invalid(
          ASSETS,
          format!(
            "assets[{i}].rho must be finite with |rho| < 1, got {:?}",
            a.rho
          ),
        ));
      }
    }

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
    if !crate::linalg::is_spd_t(corr.view()) {
      return Err(QuantError::NotPositiveDefinite {
        what: "the joint Brownian correlation matrix",
      });
    }
    Ok(())
  }

  fn sample_with_fill<F>(&self, mut fill_standard_normals: F) -> MultiHestonPaths<T>
  where
    F: FnMut(&mut [T]),
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
      prices[[i, 0]] = self.assets[i].s;
      vols[[i, 0]] = self.assets[i].v0.max(T::zero());
    }

    for step in 1..n {
      let mut z_ind = Array1::<T>::zeros(m);
      fill_standard_normals(
        z_ind
          .as_slice_mut()
          .expect("freshly allocated normal array must be contiguous"),
      );

      let mut db = Array1::<T>::zeros(m);
      for i in 0..m {
        let mut s = T::zero();
        for j in 0..=i {
          s += chol[[i, j]] * z_ind[j];
        }
        db[i] = s * sqrt_dt;
      }

      for i in 0..d {
        let v_prev = vols[[i, step - 1]].max(T::zero());
        let sqrt_v = <T as num_traits::Float>::sqrt(v_prev);
        let log_return = (self.r - T::from_f64_fast(0.5) * v_prev) * dt + sqrt_v * db[i];
        prices[[i, step]] = prices[[i, step - 1]] * <T as num_traits::Float>::exp(log_return);
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
      conditional_weights_exact: self
        .assets
        .iter()
        .all(|params| params.xi == T::zero() || params.rho == T::zero()),
    }
  }

  /// One set of paths drawn from `seed`, which advances; the Cholesky factorisation panics unless
  /// [`Self::validate`] passes, which [`Self::try_sample`] reports as an error instead.
  pub fn sample(&self, seed: &impl SeedExt) -> MultiHestonPaths<T> {
    let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(seed);
    self.sample_with_fill(|z| normal.fill_slice(z))
  }

  pub fn try_sample(&self, seed: &impl SeedExt) -> Result<MultiHestonPaths<T>, QuantError> {
    self.validate()?;
    Ok(self.sample(seed))
  }

  fn brownian_cholesky(&self) -> Array2<T> {
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
    crate::linalg::spd_cholesky_lower_t(corr.view()).expect(
      "correlation matrix not positive-definite — call MultiHestonParams::validate() up-front",
    )
  }
}

impl<T: FloatExt> MultiHestonPaths<T> {
  /// From independent single-asset `Heston`s, so uncorrelated (else [`MultiHestonParams::sample`]);
  /// their arithmetic Euler update keeps the paths out of [`Self::try_malliavin_weights`].
  pub fn from_hestons<S: SeedExt>(hestons: &[Heston<T, Euler, S>]) -> Self {
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

    Self {
      prices,
      vols,
      n_assets: d,
      n_steps: n,
      conditional_weights_exact: false,
    }
  }

  /// Whether the paths satisfy the independence assumptions of the
  /// conditional Malliavin weights.
  pub fn supports_conditional_malliavin_weights(&self) -> bool {
    self.conditional_weights_exact
  }

  pub fn terminal_prices(&self) -> Vec<T> {
    self.terminal_prices_array().to_vec()
  }

  /// Terminal prices as a contiguous numeric array.
  pub fn terminal_prices_array(&self) -> Array1<T> {
    (0..self.n_assets)
      .map(|i| self.prices[[i, self.n_steps - 1]])
      .collect()
  }
}

#[cfg(test)]
#[path = "heston/tests.rs"]
mod tests;

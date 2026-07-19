//! Multi-asset Heston simulation for M-T Greeks.
//!
//! See Malliavin & Thalmaier (2006), Chapter 4.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::volatility::heston::Heston;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

mod weights;

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

  /// Validate model, time-grid and joint-correlation inputs.
  ///
  /// This must be called before [`Self::sample`] or [`Self::sample_with_seed`]
  /// for user-supplied parameters. Their fallible variants call it directly.
  pub fn validate(&self) -> anyhow::Result<()>
  where
    T: ndarray_linalg::Lapack,
  {
    use ndarray_linalg::Cholesky;
    use ndarray_linalg::UPLO;

    let d = self.n_assets();
    if d == 0 {
      anyhow::bail!("need at least one asset");
    }
    if self.n_steps < 2 {
      anyhow::bail!("n_steps must be >= 2, got {}", self.n_steps);
    }
    if !num_traits::Float::is_finite(self.r) {
      anyhow::bail!("r must be finite");
    }
    if !num_traits::Float::is_finite(self.tau) || self.tau <= T::zero() {
      anyhow::bail!("tau must be finite and positive");
    }
    if self.cross_corr.shape() != [d, d] {
      anyhow::bail!(
        "cross_corr shape {:?} does not match n_assets={d}",
        self.cross_corr.shape()
      );
    }
    let one = T::one();
    let neg_one = -one;
    let tol = T::from_f64_fast(1e-10);
    for i in 0..d {
      for j in 0..d {
        let v = self.cross_corr[[i, j]];
        if !num_traits::Float::is_finite(v) {
          anyhow::bail!("cross_corr[{i},{j}] must be finite");
        }
        if i == j {
          if num_traits::Float::abs(v - one) > tol {
            anyhow::bail!("cross_corr[{i},{i}]={:?} is not 1", v);
          }
        } else {
          if v < neg_one || v > one {
            anyhow::bail!("cross_corr[{i},{j}]={:?} is outside [-1, 1]", v);
          }
          let v_sym = self.cross_corr[[j, i]];
          if num_traits::Float::abs(v - v_sym) > tol {
            anyhow::bail!("cross_corr is not symmetric at ({i},{j})");
          }
        }
      }
    }
    for (i, a) in self.assets.iter().enumerate() {
      if !num_traits::Float::is_finite(a.s0) || a.s0 <= T::zero() {
        anyhow::bail!("assets[{i}].s0 must be finite and positive");
      }
      for (name, value) in [
        ("v0", a.v0),
        ("kappa", a.kappa),
        ("theta", a.theta),
        ("xi", a.xi),
      ] {
        if !num_traits::Float::is_finite(value) || value < T::zero() {
          anyhow::bail!("assets[{i}].{name} must be finite and non-negative");
        }
      }
      if !num_traits::Float::is_finite(a.rho) {
        anyhow::bail!("assets[{i}].rho must be finite");
      }
      if num_traits::Float::abs(a.rho) >= one {
        anyhow::bail!("assets[{i}].rho={:?} must satisfy |rho| < 1", a.rho);
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
    corr.cholesky(UPLO::Lower).map_err(|e| {
      anyhow::anyhow!("joint Brownian correlation matrix is not positive definite: {e}")
    })?;
    Ok(())
  }

  fn sample_with_fill<F>(&self, mut fill_standard_normals: F) -> MultiHestonPaths<T>
  where
    F: FnMut(&mut [T]),
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

  /// Simulate one set of paths. Requires LAPACK for Cholesky.
  ///
  /// **Precondition:** `self.cross_corr` must be a valid `d×d` symmetric
  /// positive-definite correlation matrix (jointly with the per-asset `rho`s).
  /// Call [`Self::validate`] up-front when the matrix is user-supplied —
  /// the underlying Cholesky factorisation panics on a non-SPD input. Prefer
  /// [`Self::try_sample`] when the matrix comes from external user input.
  pub fn sample(&self) -> MultiHestonPaths<T>
  where
    T: ndarray_linalg::Lapack,
  {
    self.sample_with_fill(T::fill_standard_normal_slice)
  }

  /// Fallible variant of [`Self::sample`]. Runs [`Self::validate`] first and
  /// returns an error when the joint Brownian correlation matrix fails the
  /// positive-definite check (otherwise [`Self::sample`] would panic on the
  /// internal Cholesky factorisation).
  pub fn try_sample(&self) -> anyhow::Result<MultiHestonPaths<T>>
  where
    T: ndarray_linalg::Lapack,
  {
    self.validate()?;
    Ok(self.sample_with_fill(T::fill_standard_normal_slice))
  }

  /// Deterministic variant of [`sample`](Self::sample), intended for reproducible
  /// tests and benchmark comparisons.
  pub fn sample_with_seed(&self, seed: u64) -> MultiHestonPaths<T>
  where
    T: ndarray_linalg::Lapack,
  {
    let normal = SimdNormal::<T>::new(T::zero(), T::one(), &Deterministic::new(seed));
    self.sample_with_fill(|z| normal.fill_slice_fast(z))
  }

  /// Fallible variant of [`Self::sample_with_seed`]. See [`Self::try_sample`]
  /// for the failure mode (non-SPD joint Brownian correlation).
  pub fn try_sample_with_seed(&self, seed: u64) -> anyhow::Result<MultiHestonPaths<T>>
  where
    T: ndarray_linalg::Lapack,
  {
    self.validate()?;
    let normal = SimdNormal::<T>::new(T::zero(), T::one(), &Deterministic::new(seed));
    Ok(self.sample_with_fill(|z| normal.fill_slice_fast(z)))
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
    corr.cholesky(UPLO::Lower).expect(
      "correlation matrix not positive-definite — call MultiHestonParams::validate() up-front",
    )
  }
}

impl<T: FloatExt> MultiHestonPaths<T> {
  /// Build from existing single-asset `Heston` instances.
  ///
  /// Each `Heston` is sampled independently — use this when cross-asset
  /// correlation is zero. For correlated assets use
  /// [`MultiHestonParams::sample`] instead.
  ///
  /// The upstream `Heston` Euler sampler uses an arithmetic price update, so
  /// paths built by this adapter are not eligible for the exact conditional
  /// weights in [`Self::try_malliavin_weights`].
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

//! Stein's (2002) isotropic fractional Brownian field, not the product-form sheet, by his exact
//! intrinsic embedding. The correction is eq. (16)'s; the listed MATLAB's `kron` is not Gaussian.
//!
//! $$
//! \operatorname{Cov}(X_s, X_t)=\|s\|^{\alpha}+\|t\|^{\alpha}-\|s-t\|^{\alpha},\qquad \alpha=2H,
//! \qquad \mathbb E[(X_s-X_t)^2]=2\|s-t\|^{\alpha}\ \text{for}\ \|s-t\|\le 1\ \text{on}\ (0,r]^2
//! $$
//!
//! $$
//! \operatorname{Cov}(Y_s,Y_t)=\rho(s-t),\quad \rho(h)=\begin{cases}
//! c_0+c_2\|h\|^2-\|h\|^{\alpha},&\|h\|\le 1\\ \beta\,(r-\|h\|)^3/\|h\|,&1<\|h\|\le r
//! \end{cases}\quad(\beta=0,\ \alpha\le 1.5),\qquad
//! X_t=Y_t-Y_{t_1}+\sqrt{2c_2}\,t^\top Z,\quad Z\sim\mathcal N(0,I_2)
//! $$
//!
//! Reference: Stein (2002), J. Comput. Graph. Statist. 11(3), 587–599
//! Reference: Kroese & Botev (2015), Spatial process simulation, §4.4, eq. (16), arXiv:1308.0399
//!

use std::sync::Arc;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::SheetBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(metal_backend)]
pub(crate) mod metal;

#[derive(Debug, Clone)]
pub struct Fbs<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent `H`, `α = 2H` in the covariance; the field is isotropic,
  /// one exponent for both coordinates.
  pub hurst: T,
  /// Grid points along the sheet's second axis (rows of the output).
  pub rows: usize,
  /// Grid points along the sheet's first axis (columns of the output).
  pub cols: usize,
  /// Extent of the simulation grid: the covariance helper `rho` is defined on `[0, r]²` with a
  /// cutoff at distance `r`. Not a financial rate.
  pub r: T,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
  /// Square roots of the circulant embedding's `2(rows − 1) × 2(cols − 1)` eigenvalues, which
  /// scale every sample's complex noise; computed once, uploaded once per device configuration.
  pub(crate) lam: Arc<Array2<T>>,
  /// The intrinsic embedding's `c₂` at this Hurst exponent and cutoff: the linear term
  /// `√(2c₂) (t₁ Z₁ + t₂ Z₂)` restores the `c₂‖s − t‖²` the embedding took out of the variogram.
  pub(crate) c2: T,
}

impl<T: FloatExt, S: SeedExt> Fbs<T, S> {
  /// Panics unless `rows >= 2`, `cols >= 2` and every parameter is finite.
  pub fn new(hurst: T, rows: usize, cols: usize, r: T, seed: S) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      r.is_finite(),
      "r must satisfy `r.is_finite()`, got r = {r:?}"
    );
    assert!(
      rows >= 2,
      "rows must satisfy `rows >= 2`, got rows = {rows:?}"
    );
    assert!(
      cols >= 2,
      "cols must satisfy `cols >= 2`, got cols = {cols:?}"
    );
    let (lam, c2) = Self::embedding(hurst, rows, cols, r);
    Self {
      backend: Cpu,
      hurst,
      rows,
      cols,
      r,
      seed,
      lam: Arc::new(lam),
      c2,
    }
  }

  /// Replace `rows` and rebuild the embedding, so it is defined on `Cpu` (move with `on`
  /// afterwards); panics unless `rows >= 2`.
  pub fn with_rows(self, rows: usize) -> Self {
    Self::new(self.hurst, rows, self.cols, self.r, self.seed)
  }

  /// Replace `cols` and rebuild the embedding, so it is defined on `Cpu` (move with `on`
  /// afterwards); panics unless `cols >= 2`.
  pub fn with_cols(self, cols: usize) -> Self {
    Self::new(self.hurst, self.rows, cols, self.r, self.seed)
  }

  /// The square roots of the positive eigenvalues of the `2(m − 1) × 2(n − 1)` circulant that
  /// mirrors the covariance block against the first grid point, with the embedding's `c₂`.
  fn embedding(hurst: T, m: usize, n: usize, r: T) -> (Array2<T>, T) {
    let alpha = T::from_usize_(2) * hurst;

    let tx = Array1::linspace(r / T::from_usize_(n), r, n);
    let ty = Array1::linspace(r / T::from_usize_(m), r, m);

    let mut cov = Array2::<T>::zeros((m, n));
    for i in 0..n {
      for j in 0..m {
        cov[[j, i]] = Self::rho((tx[i], ty[j]), (tx[0], ty[0]), r, alpha).0;
      }
    }

    let big_m = 2 * (m - 1);
    let big_n = 2 * (n - 1);
    let mut blk = Array2::<T>::zeros((big_m, big_n));

    blk.slice_mut(s![..m, ..n]).assign(&cov);

    blk
      .slice_mut(s![..m, n..])
      .assign(&cov.slice(s![.., 1..n - 1;-1]));

    blk
      .slice_mut(s![m.., ..n])
      .assign(&cov.slice(s![1..m - 1;-1, ..]));

    blk
      .slice_mut(s![m.., n..])
      .assign(&cov.slice(s![1..m - 1, 1..n - 1]).slice(s![..;-1, ..;-1]));

    let scale = T::from_usize_(4) * T::from_usize_(m - 1) * T::from_usize_(n - 1);
    let fft_handler0 = FftHandler::<T>::new(big_m);
    let fft_handler1 = FftHandler::<T>::new(big_n);

    let blk_c = blk.mapv(|v| Complex::new(v, T::zero()));
    let mut fft_tmp = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&blk_c, &mut fft_tmp, &fft_handler0, 0);
    let mut fft_freq = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp, &mut fft_freq, &fft_handler1, 1);

    let lam = fft_freq.mapv(|c| (c.re / scale).max(T::zero()).sqrt());
    let (_, _, c2) = Self::rho((T::zero(), T::zero()), (T::zero(), T::zero()), r, alpha);
    (lam, c2)
  }
}

impl<T: FloatExt, S: SeedExt, B: SheetBackend<T>> Fbs<T, S, B> {
  /// One sheet from this process's own sampler, the seed advanced as [`ProcessExt::sample`]
  /// advances it: the host devices' output, and a device build's off the powers of two.
  pub(crate) fn host_sheet(&self) -> Array2<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fbs<T, S> { hurst, rows, cols, r, seed, lam, c2 } via sheet);

impl<T: FloatExt, S: SeedExt, B: SheetBackend<T>> crate::traits::Sealed for Fbs<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: SheetBackend<T>> ProcessExt<T> for Fbs<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = FbsSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> FbsSampler<T, S> {
    FbsSampler {
      lam: Arc::clone(&self.lam),
      c2: self.c2,
      m: self.rows,
      n: self.cols,
      r: self.r,
      seed: self.seed.derive(),
    }
  }

  /// Through the backend's sheet pipeline when both embedding sides are powers of two; any other
  /// grid samples on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array2<T> {
    if self.device_ready() {
      self
        .backend
        .try_sheet(self)
        .unwrap_or_else(crate::device::device_panic)
    } else {
      self.host_sheet()
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self
        .backend
        .try_sheets_map(self, m, f)
        .unwrap_or_else(crate::device::device_panic)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    if self.device_ready() {
      self
        .backend
        .try_sheets(self, m)
        .unwrap_or_else(crate::device::device_panic)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array2<T>, DeviceError> {
    if self.device_ready() {
      self.backend.try_sheet(self)
    } else {
      Ok(self.host_sheet())
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, DeviceError> {
    if self.device_ready() {
      self.backend.try_sheets(self, m)
    } else {
      Ok(crate::traits::process::sample_par_chunked(self, m))
    }
  }

  /// What a device runs here: both embedding sides, `2(rows − 1)` and `2(cols − 1)`, powers of
  /// two, as the kernels' radix-2 transforms take; any other grid samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!((self.rows - 1).is_power_of_two() && (self.cols - 1).is_power_of_two()))
      .then_some("an embedding side that is not a power of two")
  }
}

/// Reusable [`Fbs`] sampling state: owns the seed source and shares the process's embedding. The
/// matrix draw, then the two normals of the linear correction come from the derived seed.
#[doc(hidden)]
#[derive(Debug)]
pub struct FbsSampler<T: FloatExt, S: SeedExt> {
  lam: Arc<Array2<T>>,
  c2: T,
  m: usize,
  n: usize,
  r: T,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FbsSampler<T, S> {
  fn sample_inner(&mut self) -> Array2<T> {
    let (m, n, r) = (self.m, self.n, self.r);
    let (big_m, big_n) = self.lam.dim();

    let fft_handler0 = FftHandler::<T>::new(big_m);
    let fft_handler1 = FftHandler::<T>::new(big_n);

    let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);
    let z = Array2::from_shape_fn((big_m, big_n), |_| {
      Complex::new(normal.sample(), normal.sample())
    });

    let prod = self.lam.mapv(|v| Complex::new(v, T::zero())) * z;
    let mut fft_tmp2 = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&prod, &mut fft_tmp2, &fft_handler0, 0);
    let mut result = Array2::<Complex<T>>::zeros((big_m, big_n));
    ndfft(&fft_tmp2, &mut result, &fft_handler1, 1);

    let mut field = Array2::<T>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] = result[[i, j]].re;
      }
    }

    let shift = field[[0, 0]];
    field.mapv_inplace(|v| v - shift);

    // Stein's correction: the random linear function √(2c₂) (t₁ Z₁ + t₂ Z₂),
    // whose increments carry exactly the c₂‖s − t‖² the embedding took out.
    let mut normal_scalar = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);
    let mut z_buf = [T::zero(); 2];
    normal_scalar.fill_slice(&mut z_buf);
    let z1 = z_buf[0];
    let z2 = z_buf[1];

    let tx = Array1::linspace(r / T::from_usize_(n), r, n);
    let ty = Array1::linspace(r / T::from_usize_(m), r, m);
    let scale = (T::from_usize_(2) * self.c2).sqrt();
    for i in 0..m {
      for j in 0..n {
        field[[i, j]] += scale * (ty[i] * z1 + tx[j] * z2);
      }
    }

    field
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for FbsSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FbsSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> Array2<T> {
    self.sample_inner()
  }
}

impl<T: FloatExt, S: SeedExt> Fbs<T, S> {
  fn rho(x: (T, T), y: (T, T), r: T, alpha: T) -> (T, T, T) {
    let one = T::one();
    let two = T::from_usize_(2);
    let three = T::from_usize_(3);
    let half = one / two;
    let one_point_five = three * half;

    let (beta, c2, c0) = if alpha <= one_point_five {
      let c2 = alpha * half;
      let c0 = one - alpha * half;
      (T::zero(), c2, c0)
    } else {
      let beta = alpha * (two - alpha) / (three * r * (r * r - one));
      let c2 = (alpha - beta * (r - one).powi(2) * (r + two)) * half;
      let c0 = beta * (r - one).powi(3) + one - c2;
      (beta, c2, c0)
    };

    let dx = x.0 - y.0;
    let dy = x.1 - y.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let out = if dist <= one {
      c0 - dist.powf(alpha) + c2 * dist * dist
    } else if dist <= r {
      beta * (r - dist).powi(3) / dist
    } else {
      T::zero()
    };
    (out, c0, c2)
  }
}

/// The per-launch view the device pipelines share: the eigenvalue roots in the device's
/// precision, the grid, the extent, `√(2 c₂)`, and a key that tells embeddings apart in a cache.
#[cfg(device_backend)]
pub(crate) struct SheetLaunch<'a, F> {
  pub(crate) lam: &'a [F],
  pub(crate) m: usize,
  pub(crate) n: usize,
  pub(crate) r: F,
  pub(crate) corr: F,
  pub(crate) key: (u64, u64),
}

#[cfg(device_backend)]
impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// The embedding's cells: `2(rows − 1) · 2(cols − 1)`.
  #[cfg(feature = "cuda")]
  pub(crate) fn cells(&self) -> usize {
    4 * (self.rows - 1) * (self.cols - 1)
  }

  /// The key a per-size cache tells this embedding by: the Hurst exponent's
  /// and the extent's bits.
  pub(crate) fn launch_key(&self) -> (u64, u64) {
    (
      self.hurst.to_f64().unwrap_or(0.0).to_bits(),
      self.r.to_f64().unwrap_or(0.0).to_bits(),
    )
  }

  /// `√(2 c₂)`, the linear correction's coefficient.
  pub(crate) fn correction(&self) -> T {
    (T::from_usize_(2) * self.c2).sqrt()
  }

  /// A batch's flat `sheets · rows · cols` values as one array per sheet.
  pub(crate) fn sheets_from_flat<F: Copy + Into<f64>>(&self, flat: &[F]) -> Vec<Array2<T>> {
    let (m, n) = (self.rows, self.cols);
    flat
      .chunks_exact(m * n)
      .map(|sheet| {
        Array2::from_shape_fn((m, n), |(i, j)| T::from_f64_fast(sheet[i * n + j].into()))
      })
      .collect()
  }
}

py_process_2d!(PyFbs, Fbs,
  sig: (hurst, rows, cols, r, seed=None, dtype=None),
  params: (hurst: f64, rows: usize, cols: usize, r: f64),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn the_grid_setters_rebuild_the_embedding() {
    let resized = Fbs::<f64, _>::new(0.7, 9, 5, 1.0, Deterministic::new(3))
      .with_rows(17)
      .with_cols(9);
    let fresh = Fbs::<f64, _>::new(0.7, 17, 9, 1.0, Deterministic::new(3));
    assert_eq!((resized.rows, resized.cols), (17, 9));
    assert_eq!(resized.sample(), fresh.sample());
  }
}

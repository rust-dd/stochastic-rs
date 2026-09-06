//! # Fbs
//!
//! The isotropic fractional Brownian field (fractional Brownian surface) of
//! Stein (2002): the zero-mean Gaussian field on the plane with
//!
//! $$
//! \operatorname{Cov}(X_s, X_t)=\|s\|^{\alpha}+\|t\|^{\alpha}-\|s-t\|^{\alpha},\qquad \alpha=2H,
//! $$
//!
//! so that `E[(X_s − X_t)²] = 2‖s − t‖^α` — the isotropic generalisation of
//! fractional Brownian motion, not the product-form sheet. Sampled exactly by
//! Stein's intrinsic embedding: a stationary field with the covariance
//! `c₀ + c₂‖h‖² − ‖h‖^α` (cubic tail out to `r` when `α > 1.5`) is drawn by
//! circulant embedding and a two-dimensional FFT, its value at the first grid
//! point is subtracted, and the random linear term `√(2c₂) tᵀZ`, `Z` a pair
//! of independent standard normals, restores the `c₂‖s − t‖²` the embedding
//! took out. The law holds for pairs of points within unit distance of each
//! other; the grid spans `(0, r]²`.
//!
//! References: Stein, M. L. (2002), *Fast and exact simulation of fractional
//! Brownian surfaces*, J. Comput. Graph. Statist. 11(3), 587–599; Kroese, D.
//! P. & Botev, Z. I. (2015), *Spatial process simulation*, §4.4, in
//! Stochastic Geometry, Spatial Statistics and Random Fields, Springer
//! (arXiv:1308.0399). The linear correction is the one in their eq. (16)
//! construction; their listed MATLAB adds `kron(ty'·Z₁, tx·Z₂)`, a product of
//! two normals, which is neither Gaussian nor the right variogram.
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
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::SheetBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

#[derive(Debug, Clone)]
pub struct Fbs<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent `H`, `α = 2H` in the covariance; the field is isotropic,
  /// one exponent for both coordinates.
  pub hurst: T,
  /// Grid resolution along the sheet's second coordinate axis (rows of
  /// the output `Array2`).
  pub m: usize,
  /// Grid resolution along the sheet's first coordinate axis (columns of
  /// the output `Array2`).
  pub n: usize,
  /// Physical domain extent for the simulation grid — the covariance
  /// kernel in this type's internal `rho` helper is defined on `[0, r]²`
  /// with a cutoff at distance `r`. Not a financial rate; this is a pure
  /// covariance-field simulator.
  pub r: T,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
  /// The square roots of the circulant embedding's eigenvalues: the
  /// `2(m − 1) × 2(n − 1)` block every sample scales its complex Gaussian
  /// noise by, computed once here and uploaded once per device configuration.
  pub(crate) lam: Arc<Array2<T>>,
  /// The `c₂` of the intrinsic embedding at this Hurst exponent and cutoff:
  /// the random linear term `√(2c₂) (t₁ Z₁ + t₂ Z₂)` added after the shift
  /// restores the `c₂‖s − t‖²` the embedding took out of the variogram.
  pub(crate) c2: T,
}

impl<T: FloatExt, S: SeedExt> Fbs<T, S> {
  pub fn new(hurst: T, m: usize, n: usize, r: T, seed: S) -> Self {
    assert!(
      m >= 2 && n >= 2,
      "Fbs: the grid needs at least two points per axis"
    );
    let (lam, c2) = Self::embedding(hurst, m, n, r);
    Self {
      backend: Cpu,
      hurst,
      m,
      n,
      r,
      seed,
      lam: Arc::new(lam),
      c2,
    }
  }

  /// The circulant embedding of the covariance on the `m × n` grid: the
  /// covariance block against the first grid point, mirrored into the
  /// `2(m − 1) × 2(n − 1)` circulant whose two-dimensional FFT gives the
  /// eigenvalues; the square roots of their positive parts come back with
  /// the embedding's `c₂`.
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
  /// One sheet from this process's own sampler, the seed advanced as
  /// [`ProcessExt::sample`] advances it: what the host devices produce, and
  /// what a device build falls back to off the powers of two.
  pub(crate) fn host_sheet(&self) -> Array2<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fbs<T, S> { hurst, m, n, r, seed, lam, c2 } via sheet);

impl<T: FloatExt, S: SeedExt, B: SheetBackend<T>> ProcessExt<T> for Fbs<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = FbsSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> FbsSampler<T, S> {
    FbsSampler {
      lam: Arc::clone(&self.lam),
      c2: self.c2,
      m: self.m,
      n: self.n,
      r: self.r,
      seed: self.seed.derive(),
    }
  }

  /// Through the backend's sheet pipeline when both embedding sides are
  /// powers of two; any other grid samples on the host, chunked exactly as
  /// [`ProcessExt`] chunks.
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

  /// Whether a device can run this sheet: both embedding sides, `2(m − 1)`
  /// and `2(n − 1)`, powers of two, which is what the kernels' radix-2
  /// transforms take. Any other grid samples on the host.
  fn device_ready(&self) -> bool {
    (self.m - 1).is_power_of_two() && (self.n - 1).is_power_of_two()
  }
}

/// Reusable [`Fbs`] sampling state: owns the seed source so a Monte-Carlo loop
/// reuses the output field, and shares the process's embedding, so a sample
/// is the noise transform alone. The matrix draw and the two normals of the
/// linear correction come from the derived seed in that order.
#[doc(hidden)]
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

    let normal = SimdNormal::<T, 64>::new(T::zero(), T::one(), &self.seed);
    let z = Array2::from_shape_fn((big_m, big_n), |_| {
      Complex::new(normal.sample_fast(), normal.sample_fast())
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
    let normal_scalar = SimdNormal::<T>::new(T::zero(), T::one(), &self.seed);
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

/// The per-launch view of a sheet the device pipelines share: the embedding's
/// eigenvalue roots in the device's precision, the grid, the domain extent,
/// the linear correction's coefficient `√(2 c₂)`, and a key that tells one
/// embedding from another in a per-size cache.
#[cfg(any(
  feature = "metal",
  feature = "cuda"
))]
pub(crate) struct SheetLaunch<'a, F> {
  pub(crate) lam: &'a [F],
  pub(crate) m: usize,
  pub(crate) n: usize,
  pub(crate) r: F,
  pub(crate) corr: F,
  pub(crate) key: (u64, u64),
}

#[cfg(any(
  feature = "metal",
  feature = "cuda"
))]
impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// The embedding's cells: `2(m − 1) · 2(n − 1)`.
  pub(crate) fn cells(&self) -> usize {
    4 * (self.m - 1) * (self.n - 1)
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

  /// A batch's flat `sheets · m · n` values as one array per sheet.
  pub(crate) fn sheets_from_flat<F: Copy + Into<f64>>(&self, flat: &[F]) -> Vec<Array2<T>> {
    let (m, n) = (self.m, self.n);
    flat
      .chunks_exact(m * n)
      .map(|sheet| Array2::from_shape_fn((m, n), |(i, j)| T::from_f64_fast(sheet[i * n + j].into())))
      .collect()
  }
}

py_process_2d!(PyFbs, Fbs,
  sig: (hurst, m, n, r, seed=None, dtype=None),
  params: (hurst: f64, m: usize, n: usize, r: f64),
  device
);

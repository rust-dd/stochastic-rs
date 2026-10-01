//! # Core
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
use std::sync::Arc;

use ndarray::prelude::*;
use ndrustfft::FftHandler;
use ndrustfft::ndfft_inplace_par;
use num_complex::Complex;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::euler::FgnSpec;
use crate::traits::FloatExt;

/// Fractional Gaussian noise: `n` increments over `[0, t]`. Fields are private because the FFT
/// plan and spectrum are cached from them; read them through getters, change them with `with_*`.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::noise::fgn::Fgn;
/// let mut p = Fgn::<f64>::new(0.7, 10, None, Unseeded);
/// p.hurst = 0.3;
/// ```
#[derive(Clone)]
pub struct Fgn<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  /// Internal FFT length (power-of-two padded).
  pub(super) padded_n: usize,
  t: Option<T>,
  /// Padding introduced by rounding the requested length up to the next
  /// power of two (`n.next_power_of_two() - n`); the first `offset`
  /// entries of the padded circulant sample are discarded.
  pub(super) offset: usize,
  out_len: usize,
  pub(super) scale: T,
  /// Precomputed square roots of the circulant-embedding eigenvalues
  /// (Davies–Harte), shared via `Arc` across samplers so repeated calls
  /// reuse the same FFT-derived spectrum.
  sqrt_eigenvalues: Arc<Array1<T>>,
  /// Precomputed FFT plan for the padded circulant length, shared via
  /// `Arc` across samplers.
  fft_handler: Arc<FftHandler<T>>,
  seed: S,
  /// Compile-time sampling backend marker (default [`Cpu`]).
  pub(crate) backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Fgn::default().with_hurst(0.3)`.
///
/// **Cache note**: `sqrt_eigenvalues`/`fft_handler` (the Davies-Harte
/// circulant-embedding FFT plan and eigenvalues) are an expensive, pure
/// function of `hurst` and the *requested* length (`out_len`, not
/// `padded_n`) and `t`. Rather than hand-duplicating `new()`'s ~60-line
/// FFT setup (and risking it drifting out of sync), the three setters
/// that feed the cache call `Self::new(..)` again wholesale, reusing
/// `out_len` (not `padded_n`, which is already power-of-two padded) as the
/// constructor's own `n` argument. `with_seed` is the one exception:
/// neither cached array depends on the seed, so it stays a plain field
/// write.
impl<T: FloatExt, S: SeedExt> Fgn<T, S, Cpu> {
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Fgn: Hurst parameter must be in [0, 1]"
    );

    let offset = n.next_power_of_two() - n;
    let out_len = n;
    let n = n.next_power_of_two();
    let circ_len = 2 * n;
    // The embedding is built in double precision whatever `T` is, and only
    // the square roots that come out of it are narrowed.
    //
    // The autocovariance is a second difference of `k^{2H}`, and second
    // differences cancel: at `H = 0.7` and `k = 4000` the three terms are
    // each about 2.6e5 and their combination about 1e-1, six digits gone. In
    // `f32` that leaves nothing — the value is 12 % wrong by `k = 1000` and
    // *exactly zero* past `k ≈ 4000`, so the kernel flattens, every
    // eigenvalue but the first collapses, and the sampler returns white
    // noise: lag-one autocorrelation −0.006 against a theoretical 0.3195 at
    // `n = 16384`. It is a setup cost paid once per parameter set, so there
    // is nothing to save by narrowing it early.
    let f2h = 2.0 * hurst.to_f64().unwrap_or(0.5);

    let mut buf = Array1::<Complex<f64>>::zeros(circ_len);
    let buf_slice = buf.as_slice_mut().unwrap();
    buf_slice[0] = Complex::new(1.0, 0.0);
    for k in 1..=n {
      let kf = k as f64;
      let val = 0.5 * ((kf + 1.0).powf(f2h) - 2.0 * kf.powf(f2h) + (kf - 1.0).powf(f2h));
      buf_slice[k] = Complex::new(val, 0.0);
      if k > 0 && k < n {
        buf_slice[circ_len - k] = Complex::new(val, 0.0);
      }
    }

    let setup_handler = FftHandler::<f64>::new(circ_len);
    let mut buf_view = buf.view_mut();
    ndfft_inplace_par(&mut buf_view, &setup_handler, 0);

    let fft_handler = Arc::new(FftHandler::new(circ_len));
    let norm = circ_len as f64;
    let mut sqrt_eigenvalues = Array1::<T>::uninit(circ_len);
    let eig_slice =
      unsafe { std::slice::from_raw_parts_mut(sqrt_eigenvalues.as_mut_ptr() as *mut T, circ_len) };
    let buf_slice = buf.as_slice().unwrap();
    for (dst, src) in eig_slice.iter_mut().zip(buf_slice.iter()) {
      let lambda = src.re / norm;
      *dst = T::from_f64_fast(if lambda > 0.0 { lambda.sqrt() } else { 0.0 });
    }
    let sqrt_eigenvalues = unsafe { sqrt_eigenvalues.assume_init() };

    let scale_n = out_len.max(1);

    Self {
      hurst,
      padded_n: n,
      offset,
      out_len,
      t,
      scale: T::from_usize_(scale_n).powf(-hurst) * t.unwrap_or(T::one()).powf(hurst),
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler,
      seed,
      backend: Cpu,
    }
  }

  /// Replace `hurst`; rebuilds the FFT/eigenvalue cache.
  pub fn with_hurst(self, hurst: T) -> Self {
    Self::new(hurst, self.out_len, self.t, self.seed)
  }

  /// Replace the (requested, pre-padding) number of simulation steps;
  /// rebuilds the FFT/eigenvalue cache.
  pub fn with_steps(self, n: usize) -> Self {
    Self::new(self.hurst, n, self.t, self.seed)
  }

  /// Replace the simulation horizon `t`; rebuilds the FFT/eigenvalue cache
  /// (`scale` depends on `t`).
  pub fn with_horizon(self, t: Option<T>) -> Self {
    Self::new(self.hurst, self.out_len, t, self.seed)
  }

  /// Replace the seed strategy's value, all else unchanged. Neither cached
  /// array depends on the seed, so this is a plain field write.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// H=0.7 — this crate's own long-memory-example convention (used
/// throughout the fractional-process `Default` impls). t=1, n=252 — one
/// trading year of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for Fgn<T, Unseeded, Cpu> {
  fn default() -> Self {
    Self::new(T::from_f64_fast(0.7), 252, Some(T::one()), Unseeded)
  }
}

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// Hurst exponent controlling roughness and long-memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Number of increments per path, as passed to `new()` (the FFT pads it internally).
  pub fn n(&self) -> usize {
    self.out_len
  }

  /// Simulation horizon `[0, t]` the increments span (defaults to `1` if `None`).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub fn seed(&self) -> &S {
    &self.seed
  }

  /// The circulant-embedding spectrum, read-only, for tests and probes.
  #[doc(hidden)]
  pub fn sqrt_eigenvalues(&self) -> &[T] {
    self.sqrt_eigenvalues.as_slice().expect("contiguous")
  }

  pub(crate) fn fgn_spec(&self, streams: usize) -> FgnSpec<'_, T> {
    FgnSpec {
      sqrt_eigenvalues: self.sqrt_eigenvalues(),
      n: self.padded_n,
      offset: self.offset,
      hurst: self.hurst.to_f64().unwrap_or(0.5),
      t: self.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams,
    }
  }

  pub fn dt(&self) -> T {
    let step_count = self.out_len.max(1);
    self.t.unwrap_or(T::one()) / T::from_usize_(step_count)
  }

  /// Sample fGn using a specific deterministic seed.
  pub fn sample_cpu_with_seed(&self, seed: u64) -> Array1<T> {
    self.sample_cpu_impl(&Deterministic::new(seed))
  }

  /// Test-only convenience: `sample_cpu_impl(&self.seed)`, identical to what
  /// `ProcessExt::sample()` does for the `Cpu` backend. `FgnBackend::generate_batch`
  /// used to call this once per path; it now builds one `SimdNormal` per
  /// chunk and calls `fill_cpu` directly (see `device.rs`'s `Cpu` impl), so
  /// this has no production caller left — kept `#[cfg(test)]` for the
  /// covariance/marginal tests in this module and the CPU-vs-CUDA comparison
  /// tests in `cuda/tests.rs`, rather than duplicating this one-liner
  /// in both places.
  #[cfg(test)]
  pub(crate) fn sample_cpu(&self) -> Array1<T> {
    self.sample_cpu_impl(&self.seed)
  }

  /// Core fGn sampling — monomorphised per seed strategy, zero runtime branching.
  #[inline]
  pub(crate) fn sample_cpu_impl<S2: SeedExt>(&self, seed: &S2) -> Array1<T> {
    let len = 2 * self.padded_n;
    let mut fgn = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      debug_assert_eq!(rnd.len(), len);
      // SAFETY: Complex<T> is repr(C) with the layout of [T; 2], so the scratch is twice its
      // length in scalars, measured on the slice handed over, not on the length asked for.
      let flat =
        unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * rnd.len()) };
      let normal =
        stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), T::one(), seed);
      normal.fill_slice(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for (dst, c) in fgn.iter_mut().zip(src.iter()) {
        *dst = c.re * self.scale;
      }
    });

    fgn
  }

  /// Fill `out` (length `out_len`) with one fGn path using a caller-owned
  /// Gaussian source. Lets a [`FgnSampler`](super::FgnSampler) amortise the
  /// `SimdNormal` construction across a Monte-Carlo loop; the FFT plan and
  /// eigenvalues are already `Arc`-shared on the process and the complex
  /// scratch is thread-local, so this allocates nothing.
  #[inline]
  pub(crate) fn fill_cpu(
    &self,
    normal: &mut stochastic_rs_distributions::normal::SimdNormal<T>,
    out: &mut [T],
  ) {
    let len = 2 * self.padded_n;
    T::with_fgn_complex_scratch(len, |rnd| {
      debug_assert_eq!(rnd.len(), len);
      // SAFETY: Complex<T> is repr(C) with the layout of [T; 2], so the scratch is twice its
      // length in scalars, measured on the slice handed over, not on the length asked for.
      let flat =
        unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * rnd.len()) };
      normal.fill_slice(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for (dst, c) in out.iter_mut().zip(src.iter()) {
        *dst = c.re * self.scale;
      }
    });
  }

  /// Sample a pair of independent fGn paths using a specific deterministic seed.
  pub(crate) fn sample_pair_cpu_with_seed(&self, seed: u64) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_impl(&Deterministic::new(seed))
  }

  pub(crate) fn sample_pair_cpu(&self) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_impl(&self.seed)
  }

  /// Two independent fGn paths per FFT call. Re and Im of the circulant
  /// output are independent zero-mean Gaussians with the same target
  /// covariance — Dietrich & Newsam (1997), Kroese & Botev (2013 §2.2
  /// Step 4, MATLAB listing "two independent fields").
  #[inline]
  pub(crate) fn sample_pair_cpu_impl<S2: SeedExt>(&self, seed: &S2) -> (Array1<T>, Array1<T>) {
    let len = 2 * self.padded_n;
    let mut fgn_re = Array1::<T>::zeros(self.out_len);
    let mut fgn_im = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      debug_assert_eq!(rnd.len(), len);
      // SAFETY: Complex<T> is repr(C) with the layout of [T; 2], so the scratch is twice its
      // length in scalars, measured on the slice handed over, not on the length asked for.
      let flat =
        unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * rnd.len()) };
      let normal =
        stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), T::one(), seed);
      normal.fill_slice(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for ((r, i), c) in fgn_re.iter_mut().zip(fgn_im.iter_mut()).zip(src.iter()) {
        *r = c.re * self.scale;
        *i = c.im * self.scale;
      }
    });

    (fgn_re, fgn_im)
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fgn<T, S> { hurst, padded_n, t, offset, out_len, scale, sqrt_eigenvalues, fft_handler, seed } via phantom);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> Fgn<T, S, B> {
  /// One fGN increment vector on backend `B`, the device's error instead of
  /// its panic. The host-side `seed` drives the CPU path and the GPU launch
  /// seed alike.
  pub(crate) fn try_noise<S2: SeedExt>(&self, seed: &S2) -> Result<Array1<T>, DeviceError> {
    self.backend.try_generate(self, seed)
  }

  /// Two independent fGN paths in one pass on backend `B`, the device's
  /// error instead of its panic.
  pub(crate) fn try_noise_pair<S2: SeedExt>(
    &self,
    seed: &S2,
  ) -> Result<(Array1<T>, Array1<T>), DeviceError> {
    self.backend.try_generate_pair(self, seed)
  }
}

#[cfg(test)]
mod tests;

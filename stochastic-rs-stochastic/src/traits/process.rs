//! `ProcessExt` and dimensional output markers.

#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use anyhow::Result;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use either::Either;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_distributions::traits::FloatExt;

/// Stochastic process simulation trait.
///
/// Each process exposes `sample()` returning a [`Self::Output`] and
/// `sample_par(m)` returning `m` independent samples via Rayon.
///
/// ## Time-horizon (`t`) convention
///
/// Most process structs hold `t: Option<T>`. When `t.is_none()`, implementations
/// fall back to `T::one()` (i.e. one unit of time, conventionally one year for
/// finance models). This matches the convention used across the workspace and
/// the audit document; do **not** rely on it implicitly for interest-rate or
/// volatility models where the horizon meaningfully drives parameter scaling
/// (Vasicek, CIR, HJM, Heston, Bergomi). For those, set `t` explicitly. Note
/// that [`crate::interest::bgm::Bgm`] despite its name is **not** a coupled
/// LMM/BGM (see its module doc); it is a parallel array of independent
/// Euler-stepped multiplicative martingales.
///
/// ## GPU coverage
///
/// The `sample_gpu` / `sample_cuda_native` / `sample_metal` / `sample_accelerate`
/// methods are opt-in overrides; default impls return `Err`. Currently the
/// only processes with GPU implementations are [`Fgn`](crate::noise::fgn::Fgn)
/// and [`Fbm`](crate::process::fbm::Fbm), which both rely on the FFT-based
/// circulant-embedding path under `feature = "gpu"`.
///
/// **Roadmap (P3/18):** GPU implementations for Heston, RoughBergomi, Cir2F,
/// and Hjm remain TODO. These processes use *standard* Brownian noise,
/// not fractional, so the Fgn GPU path is not directly reusable — they
/// need bespoke kernels for variance updates (e.g. Andersen QE for Cir-type
/// dynamics) and correlated noise generation. ([`Bgm`](crate::interest::bgm::Bgm)
/// is excluded from the GPU roadmap on purpose: it is a parallel array of
/// uncoupled Euler-stepped multiplicative martingales, not a market model
/// — see its module doc — so a GPU sampler would just be `xn` independent
/// `Gbm`-style streams and offers no algorithmic interest.)
pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
  }

  #[cfg(feature = "gpu")]
  fn sample_gpu(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("CubeCL GPU sampling is not supported for this process")
  }

  #[cfg(feature = "cuda-native")]
  fn sample_cuda_native(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("cudarc native CUDA sampling is not supported for this process")
  }

  #[cfg(feature = "accelerate")]
  fn sample_accelerate(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("Accelerate/vDSP sampling is not supported for this process")
  }

  #[cfg(feature = "metal")]
  fn sample_metal(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("Metal GPU sampling is not supported for this process")
  }
}

/// Marker for processes whose [`ProcessExt::sample`] returns a single
/// 1D trajectory `Array1<T>`.
///
/// Auto-implemented via a blanket impl for any `P: ProcessExt<T, Output = Array1<T>>`,
/// so the user only needs to query the marker — no manual `impl` lines on each
/// process struct. Use this in generic code that should only operate on single-path
/// processes (e.g. `Bm`, `Ou`, `Gbm`, `Vasicek`).
///
/// ```ignore
/// fn last_value<T: FloatExt, P: OneDimensional<T>>(p: &P) -> T {
///   *p.sample().last().unwrap()
/// }
/// ```
pub trait OneDimensional<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {}

impl<T: FloatExt, P> OneDimensional<T> for P where P: ProcessExt<T, Output = Array1<T>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns `N` aligned
/// 1D trajectories `[Array1<T>; N]`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = [Array1<T>; N]>`.
/// Stochastic-volatility models (`Heston`, `Bergomi`, `Sabr`, `RBergomi`)
/// use `N = 2` (asset + variance); 3-state models (`HestonStochCorr`,
/// `DoubleHeston`, `Hjm`) use `N = 3`.
pub trait MultiDimensional<T: FloatExt, const N: usize>:
  ProcessExt<T, Output = [Array1<T>; N]>
{
}

impl<T: FloatExt, P, const N: usize> MultiDimensional<T, N> for P where
  P: ProcessExt<T, Output = [Array1<T>; N]>
{
}

/// Convenience marker for the common 2-state case `[Array1<T>; 2]`.
///
/// Subtrait of [`MultiDimensional<T, 2>`]. Useful for asset-plus-variance
/// stochastic-vol models like `Heston`, `Bergomi`, `Sabr`.
pub trait TwoDimensional<T: FloatExt>: MultiDimensional<T, 2> {}

impl<T: FloatExt, P> TwoDimensional<T> for P where P: MultiDimensional<T, 2> {}

/// Marker for processes whose [`ProcessExt::sample`] returns an `Array2<T>`
/// matrix — a discretised curve or sheet rather than a single path.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Array2<T>>`. Used by
/// interest-rate term-structure models (`Hjm`-with-tenors, `WuZhangD`),
/// stochastic-sheet processes (`Fbs`), and the parallel-rate primitive
/// `Bgm` (which despite its name is not a coupled BGM/LMM — see its module
/// doc).
pub trait CurveOutput<T: FloatExt>: ProcessExt<T, Output = Array2<T>> {}

impl<T: FloatExt, P> CurveOutput<T> for P where P: ProcessExt<T, Output = Array2<T>> {}

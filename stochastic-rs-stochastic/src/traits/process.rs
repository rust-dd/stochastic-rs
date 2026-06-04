//! `ProcessExt` and dimensional output markers.

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
/// ## Backend selection
///
/// Re-type a process to a compile-time sampling backend with the turbofish
/// `process.on::<B>()` where `B: `[`Backend`](crate::device::Backend) (e.g.
/// `process.on::<CudaNative>()`); the backend marker propagates to the
/// process's noise source with no runtime branch. Only the fractional family
/// (built on [`Fgn`](crate::noise::fgn::Fgn)) exposes GPU backends today, and a
/// GPU marker only exists when its feature is compiled.
pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
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

/// Marker for processes whose [`ProcessExt::sample`] returns a runtime-sized
/// collection of 1D trajectories `Vec<Array1<T>>`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Vec<Array1<T>>>`.
/// Used when the dimensionality `D` is determined by a runtime parameter and
/// each component carries its own (possibly variable-length) trace —
/// e.g. [`crate::process::multivariate_hawkes::MultivariateHawkes`], whose
/// per-component event-time vectors have process-dependent lengths.
pub trait VariableDimensional<T: FloatExt>: ProcessExt<T, Output = Vec<Array1<T>>> {}

impl<T: FloatExt, P> VariableDimensional<T> for P where P: ProcessExt<T, Output = Vec<Array1<T>>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns a complex-valued
/// 1D trajectory `Array1<Complex<T>>`.
///
/// Auto-implemented for any `P: ProcessExt<T, Output = Array1<Complex<T>>>`.
/// Used by complex-state diffusions such as
/// [`crate::diffusion::cfou::Cfou`], where the joint dynamics of two
/// real OU components are expressed as a single complex Ornstein-Uhlenbeck
/// `Z_t = X_1(t) + i X_2(t)`.
pub trait ComplexPathOutput<T: FloatExt>:
  ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}

impl<T: FloatExt, P> ComplexPathOutput<T> for P where
  P: ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}

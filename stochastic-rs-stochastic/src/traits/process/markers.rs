use ndarray::Array2;

use super::*;

/// Marker for processes whose [`ProcessExt::sample`] returns a single
/// 1D trajectory `Array1<T>`.
///
/// Auto-implemented via a blanket impl for any `P: ProcessExt<T, Output = Array1<T>>`,
/// so the user only needs to query the marker — no manual `impl` lines on each
/// process struct. Use this in generic code that should only operate on single-path
/// processes (e.g. `Bm`, `Ou`, `Gbm`, `Vasicek`).
///
/// ```
/// use stochastic_rs_stochastic::process::bm::Bm;
/// use stochastic_rs_stochastic::simd_rng::Deterministic;
/// use stochastic_rs_stochastic::traits::{FloatExt, OneDimensional, ProcessExt};
///
/// fn last_value<T: FloatExt, P: OneDimensional<T>>(p: &P) -> T {
///   *p.sample().last().unwrap()
/// }
///
/// // `Bm` has `Output = Array1<T>`, so the blanket impl gives it
/// // `OneDimensional<T>` for free — no manual `impl` needed.
/// let bm = Bm::<f64, _>::new(64, Some(1.0), Deterministic::new(1));
/// let v = last_value(&bm);
/// assert!(v.is_finite());
/// ```
pub trait OneDimensional<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {}

impl<T: FloatExt, P> OneDimensional<T> for P where P: ProcessExt<T, Output = Array1<T>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns `N` aligned paths `[Array1<T>; N]`,
/// such as `Heston`, `Bergomi` and `Sabr` (`N = 2`); implemented from `Output` alone.
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

/// Marker for processes whose [`ProcessExt::sample`] returns an `Array2<T>`, a discretised curve or
/// sheet: `WuZhang`, `Fbs`.
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

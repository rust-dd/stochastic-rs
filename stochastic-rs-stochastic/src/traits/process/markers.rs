use ndarray::Array2;

use super::*;

/// Marker for processes whose [`ProcessExt::sample`] returns one path `Array1<T>`, such as `Bm`,
/// `Ou` and `Gbm`; implemented from `Output` alone.
///
/// ```
/// use stochastic_rs_stochastic::process::bm::Bm;
/// use stochastic_rs_core::simd_rng::Deterministic;
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

/// [`MultiDimensional<T, 2>`] by name, for asset-plus-variance models such as `Heston` and `Sabr`.
pub trait TwoDimensional<T: FloatExt>: MultiDimensional<T, 2> {}

impl<T: FloatExt, P> TwoDimensional<T> for P where P: MultiDimensional<T, 2> {}

/// Marker for processes whose [`ProcessExt::sample`] returns an `Array2<T>`, a discretised curve or
/// sheet: `WuZhang`, `Fbs`.
pub trait CurveOutput<T: FloatExt>: ProcessExt<T, Output = Array2<T>> {}

impl<T: FloatExt, P> CurveOutput<T> for P where P: ProcessExt<T, Output = Array2<T>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns `Vec<Array1<T>>`, one trace of its own
/// length per component, as `MultivariateHawkes`'s event times; implemented from `Output` alone.
pub trait VariableDimensional<T: FloatExt>: ProcessExt<T, Output = Vec<Array1<T>>> {}

impl<T: FloatExt, P> VariableDimensional<T> for P where P: ProcessExt<T, Output = Vec<Array1<T>>> {}

/// Marker for processes whose [`ProcessExt::sample`] returns a complex path `Array1<Complex<T>>`,
/// as `Cfou`'s `Z_t = X_1(t) + i X_2(t)`; implemented from `Output` alone.
pub trait ComplexPathOutput<T: FloatExt>:
  ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}

impl<T: FloatExt, P> ComplexPathOutput<T> for P where
  P: ProcessExt<T, Output = ndarray::Array1<num_complex::Complex<T>>>
{
}

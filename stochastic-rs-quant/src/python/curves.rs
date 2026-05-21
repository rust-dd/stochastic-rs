use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "DiscountCurve", unsendable)]
pub struct PyDiscountCurve {
  pub(super) inner: crate::curves::discount_curve::DiscountCurve<f64>,
}

#[pymethods]
impl PyDiscountCurve {
  /// Build from `(maturities, zero_rates)` arrays under continuous compounding.
  /// `interp`: "linear" / "log_df" / "cubic" / "monotone_convex".
  #[staticmethod]
  #[pyo3(signature = (maturities, zero_rates, interp="linear"))]
  fn from_zero_rates<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    zero_rates: numpy::PyReadonlyArray1<'py, f64>,
    interp: &str,
  ) -> PyResult<Self> {
    use crate::curves::types::InterpolationMethod;
    let im = match interp.to_ascii_lowercase().as_str() {
      "linear" | "linear_zr" => InterpolationMethod::LinearOnZeroRates,
      "log_df" | "loglinear_df" => InterpolationMethod::LogLinearOnDiscountFactors,
      "cubic" | "cubic_zr" => InterpolationMethod::CubicSplineOnZeroRates,
      "monotone_convex" | "mc" => InterpolationMethod::MonotoneConvex,
      o => {
        return Err(PyValueError::new_err(format!(
          "interp must be linear/log_df/cubic/monotone_convex, got '{o}'"
        )));
      }
    };
    let mat = maturities.as_array().to_owned();
    let zr = zero_rates.as_array().to_owned();
    Ok(Self {
      inner: crate::curves::discount_curve::DiscountCurve::from_zero_rates(&mat, &zr, im),
    })
  }

  fn discount_factor(&self, t: f64) -> f64 {
    self.inner.discount_factor(t)
  }
  fn zero_rate(&self, t: f64) -> f64 {
    self.inner.zero_rate(t)
  }
  fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
    self.inner.forward_rate(t1, t2)
  }
  fn par_rate(&self, maturity: f64, frequency: u32) -> f64 {
    self.inner.par_rate(maturity, frequency)
  }

  /// Vectorised zero rates on a maturity array.
  fn zero_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let mat = maturities.as_array().to_owned();
    self.inner.zero_rates(&mat).into_pyarray(py)
  }
}

#[pyclass(name = "NelsonSiegel", unsendable)]
pub struct PyNelsonSiegel {
  inner: crate::curves::nelson_siegel::NelsonSiegel<f64>,
}

#[pymethods]
impl PyNelsonSiegel {
  #[new]
  fn new(beta0: f64, beta1: f64, beta2: f64, lambda: f64) -> Self {
    Self {
      inner: crate::curves::nelson_siegel::NelsonSiegel::new(beta0, beta1, beta2, lambda),
    }
  }

  /// Fit Nelson-Siegel parameters to market zero rates (requires openblas feature).
  #[cfg(feature = "openblas")]
  #[staticmethod]
  fn fit_curve<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    market_rates: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    let mat = maturities.as_array().to_owned();
    let mr = market_rates.as_array().to_owned();
    Self {
      inner: <crate::curves::nelson_siegel::NelsonSiegel<f64>>::fit(&mat, &mr),
    }
  }

  fn zero_rate(&self, tau: f64) -> f64 {
    self.inner.zero_rate(tau)
  }
  fn forward_rate(&self, tau: f64) -> f64 {
    self.inner.forward_rate(tau)
  }
  fn discount_factor(&self, tau: f64) -> f64 {
    self.inner.discount_factor(tau)
  }
}

#[pyclass(name = "ZeroCouponInflationCurve", unsendable)]
pub struct PyZeroCouponInflationCurve {
  inner: crate::inflation::curve::ZeroCouponInflationCurve<f64>,
}

#[pymethods]
impl PyZeroCouponInflationCurve {
  /// Build a zero-coupon inflation curve from `(pillars, breakevens)`.
  #[new]
  fn new<'py>(
    pillars: numpy::PyReadonlyArray1<'py, f64>,
    breakevens: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::inflation::curve::ZeroCouponInflationCurve::new(
        pillars.as_array().to_owned(),
        breakevens.as_array().to_owned(),
      ),
    }
  }

  /// Forward CPI index ratio $I(0, T)/I(0, 0) = (1 + b(T))^T$.
  fn forward_index_ratio(&self, t: f64) -> f64 {
    use crate::inflation::curve::InflationCurve;
    self.inner.forward_index_ratio(t)
  }
}

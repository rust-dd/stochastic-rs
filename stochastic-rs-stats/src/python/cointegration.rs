#[cfg(feature = "openblas")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "openblas")]
use pyo3::prelude::*;

#[cfg(feature = "openblas")]
#[pyclass(name = "EngleGranger", unsendable)]
pub struct PyEngleGranger {
  inner: crate::econometrics::cointegration::EngleGrangerResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyEngleGranger {
  /// Engle-Granger 2-step cointegration test for `y_t = α + β x_t + ε_t`.
  #[new]
  fn new<'py>(y: PyReadonlyArray1<'py, f64>, x: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::econometrics::cointegration::engle_granger_test(y.as_array(), x.as_array()),
    }
  }

  #[getter]
  fn alpha(&self) -> f64 {
    self.inner.alpha
  }
  #[getter]
  fn beta(&self) -> f64 {
    self.inner.beta
  }
  #[getter]
  fn adf_statistic(&self) -> f64 {
    self.inner.adf_statistic
  }
  #[getter]
  fn critical_values(&self) -> (f64, f64, f64) {
    self.inner.critical_values
  }
  #[getter]
  fn reject_no_cointegration(&self) -> bool {
    self.inner.reject_no_cointegration
  }
  fn residuals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.residuals.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "Johansen", unsendable)]
pub struct PyJohansen {
  inner: crate::econometrics::cointegration::JohansenResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyJohansen {
  /// Johansen trace test on an `(t, k)` matrix.
  #[new]
  #[pyo3(signature = (series, lags=1))]
  fn new<'py>(series: numpy::PyReadonlyArray2<'py, f64>, lags: usize) -> Self {
    Self {
      inner: crate::econometrics::cointegration::johansen_test(series.as_array(), lags),
    }
  }

  fn eigenvalues<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.eigenvalues.clone().into_pyarray(py)
  }
  fn trace_statistics<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_statistics.clone().into_pyarray(py)
  }
  fn trace_critical_5pct<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_critical_5pct.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "Granger", unsendable)]
pub struct PyGranger {
  inner: crate::econometrics::granger::GrangerResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyGranger {
  /// Granger causality of `x` → `y` with `lags` lags at significance `alpha`.
  #[new]
  #[pyo3(signature = (y, x, lags, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    lags: usize,
    alpha: f64,
  ) -> Self {
    Self {
      inner: crate::econometrics::granger::granger_causality(
        y.as_array(),
        x.as_array(),
        lags,
        alpha,
      ),
    }
  }

  #[getter]
  fn f_statistic(&self) -> f64 {
    self.inner.f_statistic
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn lags(&self) -> usize {
    self.inner.lags
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
  #[getter]
  fn reject_no_causality(&self) -> bool {
    self.inner.reject_no_causality
  }
}

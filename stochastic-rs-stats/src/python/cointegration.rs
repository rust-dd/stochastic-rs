use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "EngleGranger", unsendable)]
pub struct PyEngleGranger {
  inner: crate::econometrics::cointegration::EngleGrangerResult,
}

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

#[pyclass(name = "Johansen", unsendable)]
pub struct PyJohansen {
  inner: crate::econometrics::cointegration::JohansenResult,
}

#[pymethods]
impl PyJohansen {
  /// Johansen trace and maximum-eigenvalue tests on a `(t, k)` matrix with
  /// VAR order `lags`.
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

  /// Eigenvectors as columns, normalised `V' S11 V = I`.
  fn eigenvectors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.eigenvectors.clone().into_pyarray(py)
  }

  fn trace_statistics<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_statistics.clone().into_pyarray(py)
  }

  fn max_eig_statistics<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.max_eig_statistics.clone().into_pyarray(py)
  }

  fn trace_critical_5pct<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_critical_5pct.clone().into_pyarray(py)
  }

  fn max_eig_critical_5pct<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.max_eig_critical_5pct.clone().into_pyarray(py)
  }

  #[getter]
  fn rank_trace(&self) -> usize {
    self.inner.rank_trace
  }

  #[getter]
  fn rank_max_eig(&self) -> usize {
    self.inner.rank_max_eig
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
}

#[pyclass(name = "Vecm", unsendable)]
pub struct PyVecm {
  inner: crate::econometrics::cointegration::Vecm,
}

#[pymethods]
impl PyVecm {
  /// Maximum-likelihood VECM of a `(t, k)` matrix at cointegrating rank
  /// `rank` with VAR order `lags` (unrestricted constant).
  #[new]
  #[pyo3(signature = (series, lags=1, rank=1))]
  fn new<'py>(series: numpy::PyReadonlyArray2<'py, f64>, lags: usize, rank: usize) -> Self {
    Self {
      inner: crate::econometrics::cointegration::vecm_fit(series.as_array(), lags, rank),
    }
  }

  /// Cointegrating vectors as columns (`k × rank`), `beta' S11 beta = I`.
  fn beta<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.beta.clone().into_pyarray(py)
  }

  /// Adjustment coefficients (`k × rank`).
  fn alpha<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.alpha.clone().into_pyarray(py)
  }

  /// `alpha @ beta.T` (`k × k`).
  fn pi<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.pi.clone().into_pyarray(py)
  }

  /// Short-run matrices `Gamma_1 … Gamma_{lags-1}`, each `k × k`.
  fn gamma<'py>(&self, py: Python<'py>) -> Vec<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
    use numpy::IntoPyArray;
    self
      .inner
      .gamma
      .iter()
      .map(|g| g.clone().into_pyarray(py))
      .collect()
  }

  fn intercept<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.intercept.clone().into_pyarray(py)
  }

  /// Residuals (`nobs × k`).
  fn residuals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.residuals.clone().into_pyarray(py)
  }

  /// ML residual covariance (`k × k`).
  fn sigma<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.sigma.clone().into_pyarray(py)
  }

  fn eigenvalues<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.eigenvalues.clone().into_pyarray(py)
  }

  #[getter]
  fn rank(&self) -> usize {
    self.inner.rank
  }

  #[getter]
  fn lags(&self) -> usize {
    self.inner.lags
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
}

#[pyclass(name = "Granger", unsendable)]
pub struct PyGranger {
  inner: crate::econometrics::granger::GrangerResult,
}

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

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "HillEstimator", unsendable)]
pub struct PyHillEstimator {
  inner: crate::evt::HillResult,
}

#[pymethods]
impl PyHillEstimator {
  /// Hill (1975) tail-index estimate from the `k` largest positive entries.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, k: usize) -> Self {
    Self {
      inner: crate::evt::hill_estimator(data.as_array(), k),
    }
  }

  #[getter]
  fn xi(&self) -> f64 {
    self.inner.xi
  }

  #[getter]
  fn alpha(&self) -> f64 {
    self.inner.alpha
  }

  #[getter]
  fn std_error(&self) -> f64 {
    self.inner.std_error
  }

  #[getter]
  fn k(&self) -> usize {
    self.inner.k
  }

  #[getter]
  fn threshold(&self) -> f64 {
    self.inner.threshold
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
}

#[pyclass(name = "GpdFit", unsendable)]
pub struct PyGpdFit {
  inner: crate::evt::GpdFit,
}

#[pymethods]
impl PyGpdFit {
  /// GPD maximum-likelihood fit to non-negative threshold excesses.
  #[new]
  fn new<'py>(exceedances: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::evt::gpd_fit(exceedances.as_array()),
    }
  }

  /// Standard errors of `[sigma, xi]`.
  fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.std_errors.clone().into_pyarray(py)
  }

  fn covariance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.covariance.clone().into_pyarray(py)
  }

  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }

  #[getter]
  fn xi(&self) -> f64 {
    self.inner.xi
  }

  #[getter]
  fn log_likelihood(&self) -> f64 {
    self.inner.log_likelihood
  }

  #[getter]
  fn aic(&self) -> f64 {
    self.inner.aic
  }

  #[getter]
  fn bic(&self) -> f64 {
    self.inner.bic
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }

  #[getter]
  fn iterations(&self) -> usize {
    self.inner.iterations
  }

  #[getter]
  fn converged(&self) -> bool {
    self.inner.converged
  }
}

#[pyclass(name = "PotFit", unsendable)]
pub struct PyPotFit {
  inner: crate::evt::PotFit,
}

#[pymethods]
impl PyPotFit {
  /// Peaks-over-threshold tail model of `data` (losses) above `threshold`.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, threshold: f64) -> Self {
    Self {
      inner: crate::evt::pot_fit(data.as_array(), threshold),
    }
  }

  /// Tail quantile (Value-at-Risk) at level `p`.
  fn quantile(&self, p: f64) -> f64 {
    self.inner.quantile(p)
  }

  /// Expected shortfall at level `p`.
  fn expected_shortfall(&self, p: f64) -> f64 {
    self.inner.expected_shortfall(p)
  }

  /// Standard errors of the GPD `[sigma, xi]`.
  fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.gpd.std_errors.clone().into_pyarray(py)
  }

  #[getter]
  fn threshold(&self) -> f64 {
    self.inner.threshold
  }

  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.gpd.sigma
  }

  #[getter]
  fn xi(&self) -> f64 {
    self.inner.gpd.xi
  }

  #[getter]
  fn log_likelihood(&self) -> f64 {
    self.inner.gpd.log_likelihood
  }

  #[getter]
  fn n_exceedances(&self) -> usize {
    self.inner.n_exceedances
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }

  #[getter]
  fn exceedance_rate(&self) -> f64 {
    self.inner.exceedance_rate
  }

  #[getter]
  fn converged(&self) -> bool {
    self.inner.gpd.converged
  }
}

#[pyclass(name = "GevFit", unsendable)]
pub struct PyGevFit {
  inner: crate::evt::GevFit,
}

#[pymethods]
impl PyGevFit {
  /// GEV maximum-likelihood fit to block maxima.
  #[new]
  fn new<'py>(maxima: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::evt::gev_fit(maxima.as_array()),
    }
  }

  /// Return level exceeded once every `period` blocks on average.
  fn return_level(&self, period: f64) -> f64 {
    self.inner.return_level(period)
  }

  /// Standard errors of `[mu, sigma, xi]`.
  fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.std_errors.clone().into_pyarray(py)
  }

  fn covariance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.covariance.clone().into_pyarray(py)
  }

  #[getter]
  fn mu(&self) -> f64 {
    self.inner.mu
  }

  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }

  #[getter]
  fn xi(&self) -> f64 {
    self.inner.xi
  }

  #[getter]
  fn log_likelihood(&self) -> f64 {
    self.inner.log_likelihood
  }

  #[getter]
  fn aic(&self) -> f64 {
    self.inner.aic
  }

  #[getter]
  fn bic(&self) -> f64 {
    self.inner.bic
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }

  #[getter]
  fn iterations(&self) -> usize {
    self.inner.iterations
  }

  #[getter]
  fn converged(&self) -> bool {
    self.inner.converged
  }
}

/// Maxima of consecutive blocks of `block_size` observations (a trailing
/// partial block is dropped).
#[pyfunction]
pub fn block_maxima<'py>(
  py: Python<'py>,
  data: PyReadonlyArray1<'py, f64>,
  block_size: usize,
) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
  use numpy::IntoPyArray;
  crate::evt::block_maxima(data.as_array(), block_size).into_pyarray(py)
}

/// Mean excess over each threshold (the mean-residual-life plot); NaN
/// where nothing exceeds the threshold.
#[pyfunction]
pub fn mean_excess<'py>(
  py: Python<'py>,
  data: PyReadonlyArray1<'py, f64>,
  thresholds: PyReadonlyArray1<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
  use numpy::IntoPyArray;
  crate::evt::mean_excess(data.as_array(), thresholds.as_array()).into_pyarray(py)
}

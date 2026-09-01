use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::garch::GarchKind;
use crate::garch::GarchSpec;
use crate::garch::MeanSpec;

#[pyclass(name = "GarchFit", unsendable)]
pub struct PyGarchFit {
  inner: crate::garch::GarchFit,
}

#[pymethods]
impl PyGarchFit {
  /// Gaussian QMLE of a GARCH-family model. `kind` is one of "garch",
  /// "gjr" or "egarch"; `mean` is "constant" (estimated jointly) or "zero".
  #[new]
  #[pyo3(signature = (returns, kind="garch", p=1, q=1, mean="constant"))]
  fn new<'py>(
    returns: PyReadonlyArray1<'py, f64>,
    kind: &str,
    p: usize,
    q: usize,
    mean: &str,
  ) -> PyResult<Self> {
    let kind = match kind.to_ascii_lowercase().as_str() {
      "garch" => GarchKind::Garch,
      "gjr" | "gjr-garch" | "gjrgarch" => GarchKind::GjrGarch,
      "egarch" => GarchKind::Egarch,
      other => {
        return Err(PyValueError::new_err(format!(
          "kind must be one of garch/gjr/egarch, got '{other}'"
        )));
      }
    };
    let mean = match mean.to_ascii_lowercase().as_str() {
      "constant" => MeanSpec::Constant,
      "zero" => MeanSpec::Zero,
      other => {
        return Err(PyValueError::new_err(format!(
          "mean must be one of constant/zero, got '{other}'"
        )));
      }
    };
    let spec = GarchSpec { kind, p, q, mean };
    Ok(Self {
      inner: crate::garch::garch_fit(returns.as_array(), spec),
    })
  }

  /// Parameter names in the order of `params()`.
  fn param_names(&self) -> Vec<String> {
    self.inner.spec.param_names()
  }
  /// All parameters, `[mu, omega, alpha…, gamma…, beta…]`.
  fn params<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.params.clone().into_pyarray(py)
  }
  fn alpha<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.alpha.clone().into_pyarray(py)
  }
  /// Asymmetry coefficients; empty for plain GARCH.
  fn gamma<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.gamma.clone().into_pyarray(py)
  }
  fn beta<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.beta.clone().into_pyarray(py)
  }
  /// Inverse-Hessian standard errors, aligned with `params()`.
  fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.std_errors.clone().into_pyarray(py)
  }
  /// Bollerslev-Wooldridge robust standard errors, aligned with `params()`.
  fn robust_std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.robust_std_errors.clone().into_pyarray(py)
  }
  fn covariance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.covariance.clone().into_pyarray(py)
  }
  fn robust_covariance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.robust_covariance.clone().into_pyarray(py)
  }
  fn conditional_variance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.conditional_variance.clone().into_pyarray(py)
  }
  fn residuals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.residuals.clone().into_pyarray(py)
  }
  fn standardized_residuals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.standardized_residuals.clone().into_pyarray(py)
  }

  #[getter]
  fn kind(&self) -> &'static str {
    match self.inner.spec.kind {
      GarchKind::Garch => "garch",
      GarchKind::GjrGarch => "gjr",
      GarchKind::Egarch => "egarch",
    }
  }
  #[getter]
  fn mean(&self) -> &'static str {
    match self.inner.spec.mean {
      MeanSpec::Constant => "constant",
      MeanSpec::Zero => "zero",
    }
  }
  #[getter]
  fn p(&self) -> usize {
    self.inner.spec.p
  }
  #[getter]
  fn q(&self) -> usize {
    self.inner.spec.q
  }
  #[getter]
  fn mu(&self) -> f64 {
    self.inner.mu
  }
  #[getter]
  fn omega(&self) -> f64 {
    self.inner.omega
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
  fn persistence(&self) -> f64 {
    self.inner.persistence
  }
  #[getter]
  fn backcast(&self) -> f64 {
    self.inner.backcast
  }
  #[getter]
  fn iterations(&self) -> usize {
    self.inner.iterations
  }
  #[getter]
  fn converged(&self) -> bool {
    self.inner.converged
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
}

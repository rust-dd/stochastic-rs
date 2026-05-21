use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "GaussianKDE", unsendable)]
pub struct PyGaussianKDE {
  inner: crate::gaussian_kde::GaussianKDE,
}

#[pymethods]
impl PyGaussianKDE {
  /// Construct a Gaussian KDE with explicit bandwidth.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, bandwidth: f64) -> Self {
    Self {
      inner: crate::gaussian_kde::GaussianKDE::new(data.as_array().to_owned(), bandwidth),
    }
  }

  /// Construct a Gaussian KDE with Silverman's rule-of-thumb bandwidth.
  #[staticmethod]
  fn silverman<'py>(data: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::gaussian_kde::GaussianKDE::with_silverman_bandwidth(data.as_array().to_owned()),
    }
  }

  fn evaluate(&self, x: f64) -> f64 {
    self.inner.evaluate(x)
  }

  fn evaluate_array<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let out = self.inner.evaluate_array(&x.as_array().to_owned());
    out.into_pyarray(py)
  }
}

#[pyclass(name = "TailIndex", unsendable)]
pub struct PyTailIndex {
  xi: f64,
  alpha: f64,
}

#[pymethods]
impl PyTailIndex {
  /// Hill-style tail-exponent estimator (Mancini 2008). Provide pre-computed
  /// `mean` and `var` of the centred returns.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, mean: f64, var: f64) -> Self {
    let view = data.as_array();
    let xi = crate::tail_index::estimate_tail_exponent(view, mean, var);
    let alpha = crate::tail_index::tail_exponent_to_cgmy_alpha(xi);
    Self { xi, alpha }
  }

  #[getter]
  fn tail_exponent(&self) -> f64 {
    self.xi
  }
  /// CGMY α parameter implied by the tail exponent (`Y = α + 1`).
  #[getter]
  fn cgmy_alpha(&self) -> f64 {
    self.alpha
  }
}

#[pyclass(name = "Leverage", unsendable)]
pub struct PyLeverage {
  rho: f64,
}

#[pymethods]
impl PyLeverage {
  /// Estimate leverage correlation $\rho$ between price and volatility from
  /// a closing-price series.
  #[new]
  fn new<'py>(closes: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      rho: crate::leverage::estimate_leverage_rho(closes.as_array()),
    }
  }

  #[getter]
  fn rho(&self) -> f64 {
    self.rho
  }
}

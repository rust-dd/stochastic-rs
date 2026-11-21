use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use stochastic_rs_distributions::python::value_error_on_panic;

use crate::risk::var::PnlOrLoss;
use crate::risk::var::VarMethod;

fn parse_method_and_orientation(
  method: &str,
  orientation: &str,
) -> PyResult<(VarMethod, PnlOrLoss)> {
  let m = match method.to_ascii_lowercase().as_str() {
    "gaussian" => VarMethod::Gaussian,
    "historical" => VarMethod::Historical,
    "monte_carlo" | "mc" => VarMethod::MonteCarlo,
    o => {
      return Err(PyValueError::new_err(format!(
        "method must be one of gaussian/historical/monte_carlo, got '{o}'"
      )));
    }
  };
  let pol = match orientation.to_ascii_lowercase().as_str() {
    "pnl" => PnlOrLoss::Pnl,
    "loss" => PnlOrLoss::Loss,
    o => {
      return Err(PyValueError::new_err(format!(
        "orientation must be 'pnl' or 'loss', got '{o}'"
      )));
    }
  };
  Ok((m, pol))
}

/// Value-at-Risk with Gaussian / historical / Monte-Carlo methods.
#[pyclass(name = "VaR", unsendable)]
pub struct PyVaR {
  value: f64,
  method: String,
}

#[pymethods]
impl PyVaR {
  /// `method`: one of "gaussian" / "historical" / "monte_carlo".
  /// `orientation`: "pnl" (default — losses are `-x`) or "loss".
  #[new]
  #[pyo3(signature = (samples, confidence=0.99, method="historical", orientation="pnl"))]
  fn new<'py>(
    samples: numpy::PyReadonlyArray1<'py, f64>,
    confidence: f64,
    method: &str,
    orientation: &str,
  ) -> PyResult<Self> {
    let (m, pol) = parse_method_and_orientation(method, orientation)?;
    let value = value_error_on_panic(|| {
      crate::risk::var::value_at_risk(samples.as_array(), confidence, pol, m)
    })?
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Self {
      value,
      method: format!("{m:?}"),
    })
  }

  #[getter]
  fn value(&self) -> f64 {
    self.value
  }

  #[getter]
  fn method(&self) -> String {
    self.method.clone()
  }
}

#[pyclass(name = "ExpectedShortfall", unsendable)]
pub struct PyExpectedShortfall {
  value: f64,
}

#[pymethods]
impl PyExpectedShortfall {
  #[new]
  #[pyo3(signature = (samples, confidence=0.99, method="historical", orientation="pnl"))]
  fn new<'py>(
    samples: numpy::PyReadonlyArray1<'py, f64>,
    confidence: f64,
    method: &str,
    orientation: &str,
  ) -> PyResult<Self> {
    let (m, pol) = parse_method_and_orientation(method, orientation)?;
    let value = value_error_on_panic(|| {
      crate::risk::expected_shortfall::expected_shortfall(samples.as_array(), confidence, pol, m)
    })?
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Self { value })
  }

  #[getter]
  fn value(&self) -> f64 {
    self.value
  }
}

#[pyclass(name = "DrawdownStats", unsendable)]
pub struct PyDrawdownStats {
  inner: crate::risk::drawdown::DrawdownStats<f64>,
}

#[pymethods]
impl PyDrawdownStats {
  #[new]
  fn new<'py>(equity: numpy::PyReadonlyArray1<'py, f64>) -> pyo3::PyResult<Self> {
    crate::risk::drawdown::DrawdownStats::from_equity(equity.as_array())
      .map(|inner| Self { inner })
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  #[getter]
  fn max(&self) -> f64 {
    self.inner.max
  }

  #[getter]
  fn max_index(&self) -> usize {
    self.inner.max_index
  }

  #[getter]
  fn longest_duration(&self) -> usize {
    self.inner.longest_duration
  }

  #[getter]
  fn average(&self) -> f64 {
    self.inner.average
  }

  fn series<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.series.clone().into_pyarray(py)
  }
}

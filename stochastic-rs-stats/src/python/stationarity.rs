use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// ADF/KPSS bindings ship in every wheel (the underlying
/// `stationarity` module pulls in `ndarray-linalg`).
#[pyclass(name = "ADFTest", unsendable)]
pub struct PyADFTest {
  inner: crate::stationarity::adf::AdfTest,
}

#[pymethods]
impl PyADFTest {
  /// `deterministic`: "n" (none), "c" (constant), or "ct" (constant + trend).
  /// `lag_selection`: "aic" (default), "bic", or a non-negative integer literal
  /// for a fixed lag order.
  #[new]
  #[pyo3(signature = (y, deterministic="c", lag_selection="aic", max_lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    deterministic: &str,
    lag_selection: &str,
    max_lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::stationarity::LagSelection;
      use crate::stationarity::Trend;
      let trend = match deterministic.to_ascii_lowercase().as_str() {
        "n" | "none" => Trend::None,
        "c" | "constant" => Trend::Constant,
        "ct" | "trend" | "constant+trend" => Trend::ConstantTrend,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "deterministic must be 'n', 'c' or 'ct', got '{o}'"
          )));
        }
      };
      let sel = match lag_selection.to_ascii_lowercase().as_str() {
        "aic" => LagSelection::Aic,
        "bic" => LagSelection::Bic,
        s => match s.parse::<usize>() {
          Ok(p) => LagSelection::Fixed(p),
          Err(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
              "lag_selection must be 'aic', 'bic', or a non-negative integer, got '{s}'"
            )));
          }
        },
      };
      let cfg = crate::stationarity::adf::AdfConfig {
        trend,
        lag_selection: sel,
        max_lags,
        alpha,
      };
      let view = y.as_array();
      Ok(Self {
        inner: crate::stationarity::adf::adf_test(view, cfg),
      })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }

  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }

  #[getter]
  fn reject_unit_root(&self) -> bool {
    self.inner.rejects_null
  }

  /// Returns `(1%, 5%, 10%)` critical values.
  #[getter]
  fn critical_values(&self) -> (f64, f64, f64) {
    let cv = self.inner.critical_values.as_ref().unwrap();
    (
      cv.at(0.01).unwrap(),
      cv.at(0.05).unwrap(),
      cv.at(0.10).unwrap(),
    )
  }
}

#[pyclass(name = "KPSSTest", unsendable)]
pub struct PyKPSSTest {
  inner: crate::stationarity::kpss::KpssTest,
}

#[pymethods]
impl PyKPSSTest {
  #[new]
  #[pyo3(signature = (y, trend="level", lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let t = match trend.to_ascii_lowercase().as_str() {
        "level" | "c" => crate::stationarity::Trend::Constant,
        "trend" | "ct" => crate::stationarity::Trend::ConstantTrend,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "trend must be 'level' or 'trend', got '{o}'"
          )));
        }
      };
      let cfg = crate::stationarity::kpss::KpssConfig {
        trend: t,
        lags,
        alpha,
      };
      let view = y.as_array();
      Ok(Self {
        inner: crate::stationarity::kpss::kpss_test(view, cfg),
      })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }

  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }

  #[getter]
  fn reject_stationarity(&self) -> bool {
    self.inner.rejects_null
  }
}

#[pyclass(name = "PhillipsPerronTest", unsendable)]
pub struct PyPhillipsPerronTest {
  inner: crate::stationarity::phillips_perron::PhillipsPerronTest,
}

#[pymethods]
impl PyPhillipsPerronTest {
  /// `deterministic`: "n" / "c" / "ct". `test_type`: "tau" or "rho".
  #[new]
  #[pyo3(signature = (y, deterministic="c", test_type="tau", lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    deterministic: &str,
    test_type: &str,
    lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::stationarity::Trend;
      use crate::stationarity::phillips_perron::PpTestType;
      let trend = match deterministic.to_ascii_lowercase().as_str() {
        "n" | "none" => Trend::None,
        "c" | "constant" => Trend::Constant,
        "ct" | "trend" => Trend::ConstantTrend,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "deterministic must be 'n'/'c'/'ct', got '{o}'"
          )));
        }
      };
      let tt = match test_type.to_ascii_lowercase().as_str() {
        "tau" => PpTestType::Tau,
        "rho" => PpTestType::Rho,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "test_type must be 'tau' or 'rho', got '{o}'"
          )));
        }
      };
      let cfg = crate::stationarity::phillips_perron::PhillipsPerronConfig {
        trend,
        test_type: tt,
        lags,
        alpha,
      };
      Ok(Self {
        inner: crate::stationarity::phillips_perron::phillips_perron_test(y.as_array(), cfg),
      })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }

  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }

  #[getter]
  fn reject_unit_root(&self) -> Option<bool> {
    self.inner.rejects_null
  }
}

#[pyclass(name = "ERSTest", unsendable)]
pub struct PyERSTest {
  inner: crate::stationarity::ers_dfgls::ErsTest,
}

#[pymethods]
impl PyERSTest {
  /// Elliott-Rothenberg-Stock DF-GLS unit-root test.
  /// `trend`: "c" (constant) or "ct" (constant + trend).
  #[new]
  #[pyo3(signature = (y, trend="c", lag_selection="aic", max_lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    lag_selection: &str,
    max_lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::stationarity::LagSelection;
      use crate::stationarity::Trend;
      let tr = match trend.to_ascii_lowercase().as_str() {
        "c" | "constant" => Trend::Constant,
        "ct" | "trend" => Trend::ConstantTrend,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "trend must be 'c' or 'ct', got '{o}'"
          )));
        }
      };
      let sel = match lag_selection.to_ascii_lowercase().as_str() {
        "aic" => LagSelection::Aic,
        "bic" => LagSelection::Bic,
        s => match s.parse::<usize>() {
          Ok(p) => LagSelection::Fixed(p),
          Err(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
              "lag_selection must be 'aic'/'bic'/integer, got '{s}'"
            )));
          }
        },
      };
      let cfg = crate::stationarity::ers_dfgls::ErsConfig {
        trend: tr,
        lag_selection: sel,
        max_lags,
        alpha,
      };
      Ok(Self {
        inner: crate::stationarity::ers_dfgls::ers_dfgls_test(y.as_array(), cfg),
      })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }

  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }

  #[getter]
  fn reject_unit_root(&self) -> bool {
    self.inner.rejects_null
  }
}

#[pyclass(name = "LeybourneMcCabeTest", unsendable)]
pub struct PyLeybourneMcCabeTest {
  inner: crate::stationarity::leybourne_mccabe::LeybourneMcCabeTest,
}

#[pymethods]
impl PyLeybourneMcCabeTest {
  /// Leybourne-McCabe stationarity test (parametric bootstrap p-value).
  #[new]
  #[pyo3(signature = (y, trend="level", ar_lags=1, bootstrap_samples=400, bootstrap_seed=1234, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    ar_lags: usize,
    bootstrap_samples: usize,
    bootstrap_seed: u64,
    alpha: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::stationarity::Trend;
      let tr = match trend.to_ascii_lowercase().as_str() {
        "level" | "c" => Trend::Constant,
        "trend" | "ct" => Trend::ConstantTrend,
        o => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "trend must be 'level' or 'trend', got '{o}'"
          )));
        }
      };
      let cfg = crate::stationarity::leybourne_mccabe::LeybourneMcCabeConfig {
        trend: tr,
        ar_lags,
        bootstrap_samples,
        bootstrap_seed,
        alpha,
      };
      Ok(Self {
        inner: crate::stationarity::leybourne_mccabe::leybourne_mccabe_test(y.as_array(), cfg),
      })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }

  #[getter]
  fn ar_lags(&self) -> usize {
    self.inner.ar_lags
  }

  #[getter]
  fn p_value(&self) -> Option<f64> {
    self.inner.p_value
  }

  #[getter]
  fn reject_stationarity(&self) -> bool {
    self.inner.rejects_null
  }
}

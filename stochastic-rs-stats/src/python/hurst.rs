use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "FukasawaHurst", unsendable)]
pub struct PyFukasawaHurst {
  inner: crate::hurst::whittle::FukasawaResult,
}

#[pymethods]
impl PyFukasawaHurst {
  /// Estimate Fukasawa Hurst directly from a price series.
  #[staticmethod]
  fn from_prices<'py>(closes: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::hurst::whittle::estimate_from_prices(closes.as_array()),
    }
  }

  /// Estimate from log realised variance series, with `m` intra-bucket samples
  /// and `delta` sampling step.
  #[staticmethod]
  fn from_log_rv<'py>(log_rv: PyReadonlyArray1<'py, f64>, m: usize, delta: f64) -> Self {
    Self {
      inner: crate::hurst::whittle::estimate(log_rv.as_array(), m, delta),
    }
  }

  #[getter]
  fn hurst(&self) -> f64 {
    self.inner.hurst
  }
  #[getter]
  fn eta(&self) -> f64 {
    self.inner.eta
  }
  #[getter]
  fn neg_log_lik(&self) -> f64 {
    self.inner.neg_log_lik
  }
  #[getter]
  fn n_obs(&self) -> usize {
    self.inner.n_obs
  }
}

#[pyclass(name = "FouEstimate", unsendable)]
pub struct PyFouEstimate {
  inner: crate::fou_estimator::FouEstimateResult,
}

#[pymethods]
impl PyFouEstimate {
  /// V1 estimator (Daubechies-filter-based).
  #[staticmethod]
  #[pyo3(signature = (path, delta=None, hurst_override=None))]
  fn v1<'py>(
    path: PyReadonlyArray1<'py, f64>,
    delta: Option<f64>,
    hurst_override: Option<f64>,
  ) -> Self {
    Self {
      inner: crate::fou_estimator::estimate_fou_v1(
        path.as_array(),
        crate::fou_estimator::FilterType::Daubechies,
        delta,
        hurst_override,
      ),
    }
  }

  /// V2 estimator (moments-based, no linear filters).
  #[staticmethod]
  #[pyo3(signature = (path, delta=None, hurst_override=None))]
  fn v2<'py>(
    path: PyReadonlyArray1<'py, f64>,
    delta: Option<f64>,
    hurst_override: Option<f64>,
  ) -> Self {
    let n = path.as_array().len();
    Self {
      inner: crate::fou_estimator::estimate_fou_v2(path.as_array(), delta, n, hurst_override),
    }
  }

  #[getter]
  fn hurst(&self) -> f64 {
    self.inner.hurst
  }
  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }
  #[getter]
  fn mu(&self) -> f64 {
    self.inner.mu
  }
  #[getter]
  fn theta(&self) -> f64 {
    self.inner.theta
  }
}

#[pyclass(name = "HurstResult", unsendable)]
pub struct PyHurstResult {
  inner: crate::hurst::HurstResult<f64>,
}

#[pymethods]
impl PyHurstResult {
  #[getter]
  fn hurst(&self) -> f64 {
    self.inner.hurst
  }
  #[getter]
  fn std_err(&self) -> Option<f64> {
    self.inner.std_err
  }
  #[getter]
  fn n_obs(&self) -> usize {
    self.inner.n_obs
  }
  #[getter]
  fn slope(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::LogLogRegression { slope, .. } => Some(*slope),
      _ => None,
    }
  }
  #[getter]
  fn intercept(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::LogLogRegression { intercept, .. } => Some(*intercept),
      _ => None,
    }
  }
  #[getter]
  fn r_squared(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::LogLogRegression { r_squared, .. } => Some(*r_squared),
      _ => None,
    }
  }
  #[getter]
  fn neg_log_lik(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::Whittle { neg_log_lik, .. } => Some(*neg_log_lik),
      _ => None,
    }
  }
  #[getter]
  fn eta(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::Whittle { eta, .. } => Some(*eta),
      _ => None,
    }
  }
  #[getter]
  fn fractal_dim(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::hurst::HurstDiagnostic::FractalDim { d } => Some(*d),
      _ => None,
    }
  }
}

#[pyclass(name = "RescaledRange", unsendable)]
pub struct PyRescaledRange {
  inner: crate::hurst::RescaledRange,
}

#[pymethods]
impl PyRescaledRange {
  #[new]
  #[pyo3(signature = (anis_lloyd=true, take_differences=true, min_window=16, max_window=None, n_windows=30))]
  fn new(
    anis_lloyd: bool,
    take_differences: bool,
    min_window: usize,
    max_window: Option<usize>,
    n_windows: usize,
  ) -> Self {
    Self {
      inner: crate::hurst::RescaledRange {
        anis_lloyd,
        take_differences,
        min_window,
        max_window,
        n_windows,
      },
    }
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Dfa", unsendable)]
pub struct PyDfa {
  inner: crate::hurst::Dfa,
}

#[pymethods]
impl PyDfa {
  #[new]
  #[pyo3(signature = (order=1, min_window=8, max_window=None, n_windows=24, overlap_pct=0.0, assume_integrated=false))]
  fn new(
    order: usize,
    min_window: usize,
    max_window: Option<usize>,
    n_windows: usize,
    overlap_pct: f64,
    assume_integrated: bool,
  ) -> Self {
    Self {
      inner: crate::hurst::Dfa {
        order,
        min_window,
        max_window,
        n_windows,
        overlap_pct,
        assume_integrated,
      },
    }
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Gph", unsendable)]
pub struct PyGph {
  inner: crate::hurst::Gph,
}

#[pymethods]
impl PyGph {
  #[new]
  #[pyo3(signature = (bandwidth_alpha=0.6, take_differences=true))]
  fn new(bandwidth_alpha: f64, take_differences: bool) -> Self {
    Self {
      inner: crate::hurst::Gph {
        bandwidth_alpha,
        take_differences,
      },
    }
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Wavelet", unsendable)]
pub struct PyWavelet {
  inner: crate::hurst::Wavelet,
}

#[pymethods]
impl PyWavelet {
  #[new]
  #[pyo3(signature = (take_differences=true, j_min=2, j_max=None))]
  fn new(take_differences: bool, j_min: usize, j_max: Option<usize>) -> Self {
    Self {
      inner: crate::hurst::Wavelet {
        wavelet: crate::hurst::WaveletKind::Daubechies4,
        take_differences,
        j_min,
        j_max,
      },
    }
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "FdResult", unsendable)]
pub struct PyFdResult {
  inner: crate::fractal_dim::FdResult<f64>,
}

#[pymethods]
impl PyFdResult {
  #[getter]
  fn d(&self) -> f64 {
    self.inner.d
  }
  #[getter]
  fn n_obs(&self) -> usize {
    self.inner.n_obs
  }
  #[getter]
  fn slope(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::fractal_dim::FdDiagnostic::LogLogRegression { slope, .. } => Some(*slope),
      _ => None,
    }
  }
  #[getter]
  fn intercept(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::fractal_dim::FdDiagnostic::LogLogRegression { intercept, .. } => Some(*intercept),
      _ => None,
    }
  }
  #[getter]
  fn r_squared(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::fractal_dim::FdDiagnostic::LogLogRegression { r_squared, .. } => Some(*r_squared),
      _ => None,
    }
  }
  #[getter]
  fn v_short(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::fractal_dim::FdDiagnostic::VariogramRatio { v_short, .. } => Some(*v_short),
      _ => None,
    }
  }
  #[getter]
  fn v_long(&self) -> Option<f64> {
    match &self.inner.diagnostic {
      crate::fractal_dim::FdDiagnostic::VariogramRatio { v_long, .. } => Some(*v_long),
      _ => None,
    }
  }
}

#[pyclass(name = "Higuchi", unsendable)]
pub struct PyHiguchi {
  inner: crate::fractal_dim::Higuchi,
}

#[pymethods]
impl PyHiguchi {
  #[new]
  #[pyo3(signature = (kmax=32))]
  fn new(kmax: usize) -> Self {
    Self {
      inner: crate::fractal_dim::Higuchi { kmax },
    }
  }

  /// Return the fractal dimension `D` via the
  /// [`FractalDimEstimator`](crate::fractal_dim::FractalDimEstimator) trait.
  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyFdResult> {
    use crate::fractal_dim::FractalDimEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyFdResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  /// Return the Hurst exponent `H = 2 - D` (clamped to `(0, 1)`) via
  /// the [`HurstEstimator`](crate::hurst::HurstEstimator) trait impl.
  fn estimate_hurst<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    HurstEstimator::<f64>::estimate(&self.inner, x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Variogram", unsendable)]
pub struct PyVariogram {
  inner: crate::fractal_dim::Variogram,
}

#[pymethods]
impl PyVariogram {
  #[new]
  #[pyo3(signature = (p=2.0))]
  fn new(p: f64) -> Self {
    Self {
      inner: crate::fractal_dim::Variogram { p },
    }
  }

  /// Return the fractal dimension `D` via the
  /// [`FractalDimEstimator`](crate::fractal_dim::FractalDimEstimator) trait.
  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyFdResult> {
    use crate::fractal_dim::FractalDimEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyFdResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }

  /// Return the Hurst exponent `H = 2 - D` (clamped to `(0, 1)`) via
  /// the [`HurstEstimator`](crate::hurst::HurstEstimator) trait impl.
  fn estimate_hurst<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    HurstEstimator::<f64>::estimate(&self.inner, x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Whittle", unsendable)]
pub struct PyWhittle {
  inner: crate::hurst::Whittle,
}

#[pymethods]
impl PyWhittle {
  #[new]
  #[pyo3(signature = (m=1, delta=1.0/250.0, psi=1e-5, k_trunc=500, j_max=20))]
  fn new(m: usize, delta: f64, psi: f64, k_trunc: usize, j_max: usize) -> Self {
    Self {
      inner: crate::hurst::Whittle {
        m,
        delta,
        psi,
        k_trunc,
        j_max,
      },
    }
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

#[pyclass(name = "Variations", unsendable)]
pub struct PyVariations {
  inner: crate::hurst::Variations,
}

#[pymethods]
impl PyVariations {
  /// `kind` is one of `"daubechies"`, `"central_diff"`, or
  /// `"power_variation"`.  For `"power_variation"` pass `k` and `p`
  /// (defaults `k=2`, `p=2.0`).
  #[new]
  #[pyo3(signature = (kind="central_diff", k=2, p=2.0))]
  fn new(kind: &str, k: usize, p: f64) -> PyResult<Self> {
    let kind = match kind {
      "daubechies" => crate::hurst::VariationKind::Daubechies,
      "central_diff" => crate::hurst::VariationKind::CentralDiff,
      "power_variation" => crate::hurst::VariationKind::PowerVariation { k, p },
      other => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "unknown variation kind `{other}` (expected daubechies/central_diff/power_variation)"
        )));
      }
    };
    Ok(Self {
      inner: crate::hurst::Variations { kind },
    })
  }

  fn estimate<'py>(&self, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyHurstResult> {
    use crate::hurst::HurstEstimator;
    self
      .inner
      .estimate(x.as_array())
      .map(|inner| PyHurstResult { inner })
      .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
  }
}

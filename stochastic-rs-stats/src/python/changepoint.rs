use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "Cusum", unsendable)]
pub struct PyCusum {
  inner: crate::econometrics::changepoint::CusumResult,
}

#[pymethods]
impl PyCusum {
  /// CUSUM control chart with reference value `k` (half the smallest shift in
  /// SD units) and threshold `h`.
  #[new]
  fn new<'py>(series: PyReadonlyArray1<'py, f64>, k: f64, h: f64) -> Self {
    Self {
      inner: crate::econometrics::changepoint::cusum(series.as_array(), k, h),
    }
  }

  fn upper<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.upper.clone().into_pyarray(py)
  }
  fn lower<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.lower.clone().into_pyarray(py)
  }
  #[getter]
  fn alarms(&self) -> Vec<usize> {
    self.inner.alarms.clone()
  }
}

#[pyclass(name = "Pelt", unsendable)]
pub struct PyPelt {
  inner: crate::econometrics::changepoint::PeltResult,
}

#[pymethods]
impl PyPelt {
  /// PELT changepoint detection with mean-shift cost and `penalty` per
  /// changepoint. `min_size` enforces a minimum segment length.
  #[new]
  #[pyo3(signature = (series, penalty, min_size=1))]
  fn new<'py>(series: PyReadonlyArray1<'py, f64>, penalty: f64, min_size: usize) -> Self {
    Self {
      inner: crate::econometrics::changepoint::pelt(series.as_array(), penalty, min_size),
    }
  }

  #[getter]
  fn changepoints(&self) -> Vec<usize> {
    self.inner.changepoints.clone()
  }
  #[getter]
  fn cost(&self) -> f64 {
    self.inner.cost
  }
}

/// Spectral / periodogram bindings — Welch / Bartlett / Parzen periodograms via FFT.
#[pyclass(name = "PeriodogramFFT", unsendable)]
pub struct PyPeriodogramFFT {
  inner: crate::spectral::PeriodogramResult,
}

#[pymethods]
impl PyPeriodogramFFT {
  /// FFT periodogram with default config (mean-detrend, Hann window, one-sided density).
  #[new]
  #[pyo3(signature = (signal, sampling_rate=1.0))]
  fn new<'py>(signal: PyReadonlyArray1<'py, f64>, sampling_rate: f64) -> Self {
    let cfg = crate::spectral::PeriodogramConfig {
      sampling_rate,
      ..crate::spectral::PeriodogramConfig::default()
    };
    Self {
      inner: crate::spectral::periodogram_fft(signal.as_array(), cfg),
    }
  }

  fn frequencies(&self) -> Vec<f64> {
    self.inner.frequencies.clone()
  }
  fn spectrum(&self) -> Vec<f64> {
    self.inner.spectrum.clone()
  }
  #[getter]
  fn resolution_hz(&self) -> f64 {
    self.inner.resolution_hz
  }
  #[getter]
  fn nfft(&self) -> usize {
    self.inner.nfft
  }
}

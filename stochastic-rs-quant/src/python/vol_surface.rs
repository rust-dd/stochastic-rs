use pyo3::prelude::*;

use super::pricing_fourier::PyBatesFourier;
use super::pricing_fourier::PyHestonFourier;

#[pyclass(name = "SviRawParams", unsendable)]
pub struct PySviRawParams {
  inner: crate::vol_surface::svi::SviRawParams<f64>,
}

#[pymethods]
impl PySviRawParams {
  #[new]
  fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
    Self {
      inner: crate::vol_surface::svi::SviRawParams::new(a, b, rho, m, sigma),
    }
  }

  fn total_variance(&self, k: f64) -> f64 {
    self.inner.total_variance(k)
  }
  fn implied_vol(&self, k: f64, t: f64) -> f64 {
    self.inner.implied_vol(k, t)
  }
  fn min_variance(&self) -> f64 {
    self.inner.min_variance()
  }
  fn is_admissible(&self) -> bool {
    self.inner.is_admissible()
  }
}

#[pyclass(name = "SsviParams", unsendable)]
pub struct PySsviParams {
  inner: crate::vol_surface::ssvi::SsviParams<f64>,
}

#[pymethods]
impl PySsviParams {
  #[new]
  fn new(rho: f64, eta: f64, gamma: f64) -> Self {
    Self {
      inner: crate::vol_surface::ssvi::SsviParams::new(rho, eta, gamma),
    }
  }

  fn total_variance(&self, k: f64, theta: f64) -> f64 {
    self.inner.total_variance(k, theta)
  }
  fn implied_vol(&self, k: f64, theta: f64, t: f64) -> f64 {
    self.inner.implied_vol(k, theta, t)
  }
  fn satisfies_no_butterfly_condition(&self) -> bool {
    self.inner.satisfies_no_butterfly_condition()
  }
}

/// Implied vol surface — built from a `(N_T, N_K)` price grid via FFT
/// inversion.
#[pyclass(name = "ImpliedVolSurface", unsendable)]
pub struct PyImpliedVolSurface {
  inner: crate::vol_surface::implied::ImpliedVolSurface,
}

#[pymethods]
impl PyImpliedVolSurface {
  /// Build from a Heston Fourier model + grid via Carr-Madan FFT.
  #[staticmethod]
  fn from_heston(
    model: &PyHestonFourier,
    s: f64,
    r: f64,
    q: f64,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
  ) -> Self {
    Self {
      inner: crate::vol_surface::model_surface::fourier_model_surface_fft(
        &model.inner,
        s,
        r,
        q,
        &strikes,
        &maturities,
      ),
    }
  }

  /// Build from a Bates Fourier model + grid via Carr-Madan FFT.
  #[staticmethod]
  fn from_bates(
    model: &PyBatesFourier,
    s: f64,
    r: f64,
    q: f64,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
  ) -> Self {
    Self {
      inner: crate::vol_surface::model_surface::fourier_model_surface_fft(
        &model.inner,
        s,
        r,
        q,
        &strikes,
        &maturities,
      ),
    }
  }

  fn strikes(&self) -> Vec<f64> {
    self.inner.strikes.clone()
  }
  fn maturities(&self) -> Vec<f64> {
    self.inner.maturities.clone()
  }

  /// Returns the IV grid as a flattened `(N_T, N_K)` row-major numpy array.
  fn ivs<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.ivs.clone().into_pyarray(py)
  }
}

/// Standalone SVI single-slice calibrator (least-squares fit of a single
/// `SviRawParams` to a `(log_moneyness, total_variance)` slice).
#[pyclass(name = "SviCalibrator", unsendable)]
pub struct PySviCalibrator {
  fitted: crate::vol_surface::svi::SviRawParams<f64>,
}

#[pymethods]
impl PySviCalibrator {
  #[new]
  fn new(log_moneyness: Vec<f64>, total_variance: Vec<f64>) -> Self {
    Self {
      fitted: crate::vol_surface::svi::calibrate_svi(&log_moneyness, &total_variance, None),
    }
  }

  /// `(a, b, rho, m, sigma)`.
  fn params(&self) -> (f64, f64, f64, f64, f64) {
    let p = &self.fitted;
    (p.a, p.b, p.rho, p.m, p.sigma)
  }

  fn implied_vol(&self, k: f64, t: f64) -> f64 {
    self.fitted.implied_vol(k, t)
  }

  fn total_variance(&self, k: f64) -> f64 {
    self.fitted.total_variance(k)
  }
}

/// Standalone SSVI joint multi-slice calibrator.
#[pyclass(name = "SsviCalibrator", unsendable)]
pub struct PySsviCalibrator {
  fitted: crate::vol_surface::ssvi::SsviParams<f64>,
}

#[pymethods]
impl PySsviCalibrator {
  /// `slices`: list of `(log_moneyness, total_variance, theta_atm)` triplets.
  #[new]
  fn new(slices: Vec<(Vec<f64>, Vec<f64>, f64)>) -> Self {
    let inner_slices: Vec<crate::vol_surface::ssvi::SsviSlice<f64>> = slices
      .into_iter()
      .map(
        |(log_moneyness, total_variance, theta)| crate::vol_surface::ssvi::SsviSlice {
          log_moneyness,
          total_variance,
          theta,
        },
      )
      .collect();
    Self {
      fitted: crate::vol_surface::ssvi::calibrate_ssvi(&inner_slices, None),
    }
  }

  /// `(rho, eta, gamma)`.
  fn params(&self) -> (f64, f64, f64) {
    let p = &self.fitted;
    (p.rho, p.eta, p.gamma)
  }

  fn implied_vol(&self, k: f64, theta: f64, t: f64) -> f64 {
    self.fitted.implied_vol(k, theta, t)
  }
}

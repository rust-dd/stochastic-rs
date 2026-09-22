use ndarray::Array1;
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::PyMcEstimate;
use crate::calibration::heston::HestonParams;
use crate::calibration::heston_slv::HestonSlvCalibrationResult;
use crate::calibration::heston_slv::HestonSlvCalibrator;
use crate::pricing::slv::FokkerPlanckMethod;
use crate::pricing::slv::HestonSlvParams;
use crate::pricing::slv::HestonSlvPricer;
use crate::pricing::slv::LeverageSurface;
use crate::pricing::slv::ParticleMethod;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

fn params_tuple(p: &HestonSlvParams) -> (f64, f64, f64, f64, f64, f64) {
  (p.kappa, p.theta, p.sigma, p.rho, p.v0, p.eta)
}

#[pyclass(name = "LeverageSurface", from_py_object)]
#[derive(Clone)]
pub struct PyLeverageSurface {
  pub inner: LeverageSurface,
}

#[pymethods]
impl PyLeverageSurface {
  /// `values[j, i] = L(spots[i], times[j])`, both axes strictly ascending.
  #[new]
  fn new(
    spots: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray2<'_, f64>,
  ) -> PyResult<Self> {
    let spots = spots.as_array().to_owned();
    let times = times.as_array().to_owned();
    let values = values.as_array().to_owned();
    if spots.is_empty() || times.is_empty() {
      return Err(PyValueError::new_err("spots and times must not be empty"));
    }
    if spots.windows(2).into_iter().any(|w| w[0] >= w[1])
      || times.windows(2).into_iter().any(|w| w[0] >= w[1])
    {
      return Err(PyValueError::new_err(
        "spots and times must be strictly ascending",
      ));
    }
    if values.dim() != (times.len(), spots.len()) {
      return Err(PyValueError::new_err(format!(
        "values must have shape (times, spots) = ({}, {}), got {:?}",
        times.len(),
        spots.len(),
        values.dim()
      )));
    }
    Ok(Self {
      inner: LeverageSurface::new(spots, times, values),
    })
  }

  /// `L(s, t)` by bilinear interpolation, the nearest edge held flat outside
  /// the grid.
  fn interpolate(&self, s: f64, t: f64) -> f64 {
    self.inner.interpolate(s, t)
  }

  /// Whether `(s, tau)` lies inside the calibrated box.
  fn covers(&self, s: f64, tau: f64) -> bool {
    self.inner.covers(s, tau)
  }

  fn spot_range(&self) -> (f64, f64) {
    self.inner.spot_range()
  }

  fn horizon(&self) -> f64 {
    self.inner.horizon()
  }

  #[getter]
  fn spots<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.spots().clone().into_pyarray(py)
  }

  #[getter]
  fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.times().clone().into_pyarray(py)
  }

  /// Shape `(times, spots)`.
  #[getter]
  fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    self.inner.values().clone().into_pyarray(py)
  }
}

#[pyclass(name = "HestonSlvPricer")]
pub struct PyHestonSlvPricer {
  inner: HestonSlvPricer,
}

#[pymethods]
impl PyHestonSlvPricer {
  /// Monte Carlo pricer of the Heston SLV model under `leverage`. Pass the
  /// `r` and `q` the surface was calibrated at to anchor the pricer — any
  /// other query rate then raises — or neither for a hand-built surface with
  /// no rate provenance.
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, v0, eta, leverage, r=None, q=None, n_paths=100_000, steps_per_year=200, seed=42))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    eta: f64,
    leverage: PyLeverageSurface,
    r: Option<f64>,
    q: Option<f64>,
    n_paths: usize,
    steps_per_year: usize,
    seed: u64,
  ) -> PyResult<Self> {
    if v0 < 0.0 || theta < 0.0 || sigma < 0.0 || kappa < 0.0 {
      return Err(PyValueError::new_err(
        "kappa, theta, sigma and v0 must be non-negative",
      ));
    }
    if !(0.0..=1.0).contains(&eta) {
      return Err(PyValueError::new_err("eta must lie in [0, 1]"));
    }
    if !(-1.0..=1.0).contains(&rho) {
      return Err(PyValueError::new_err("rho must lie in [-1, 1]"));
    }
    if n_paths == 0 || steps_per_year == 0 {
      return Err(PyValueError::new_err(
        "n_paths and steps_per_year must be positive",
      ));
    }
    let params = HestonSlvParams {
      kappa,
      theta,
      sigma,
      rho,
      v0,
      eta,
    };
    let inner = match (r, q) {
      (Some(r), Some(q)) => HestonSlvPricer::new(params, leverage.inner, r, q),
      (None, None) => HestonSlvPricer::unanchored(params, leverage.inner),
      _ => {
        return Err(PyValueError::new_err(
          "pass both r and q to anchor the pricer to the calibration rates, or neither",
        ));
      }
    };
    Ok(Self {
      inner: inner
        .with_paths(n_paths)
        .with_steps_per_year(steps_per_year)
        .with_seed(seed),
    })
  }

  /// Raises when the pricer is anchored and `(r, q)` is not the calibration
  /// pair. `NaN` when `(s, tau)` lies outside the leverage surface.
  #[pyo3(signature = (s, k, r, q, tau))]
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> PyResult<f64> {
    self.check_rates(r, q)?;
    Ok(self.inner.price_call(s, k, r, q, tau))
  }

  #[pyo3(signature = (s, k, r, q, tau))]
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> PyResult<f64> {
    self.check_rates(r, q)?;
    Ok(self.inner.price_put(s, k, r, q, tau))
  }

  /// The call price with its Monte Carlo error bar.
  #[pyo3(signature = (s, k, r, q, tau))]
  fn price_call_estimate(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> PyResult<PyMcEstimate> {
    self.check_rates(r, q)?;
    Ok(self.inner.price_call_estimate(s, k, r, q, tau).into())
  }

  /// `(kappa, theta, sigma, rho, v0, eta)`.
  #[getter]
  fn params(&self) -> (f64, f64, f64, f64, f64, f64) {
    params_tuple(&self.inner.params)
  }

  #[getter]
  fn leverage(&self) -> PyLeverageSurface {
    PyLeverageSurface {
      inner: self.inner.leverage.clone(),
    }
  }

  /// The `(r, q)` the surface was calibrated at, `None` when unanchored.
  #[getter]
  fn calibration_rates(&self) -> Option<(f64, f64)> {
    self.inner.calibration_rates
  }
}

impl PyHestonSlvPricer {
  fn check_rates(&self, r: f64, q: f64) -> PyResult<()> {
    match self.inner.calibration_rates {
      Some((r0, q0)) if (r - r0).abs() > 1e-12 || (q - q0).abs() > 1e-12 => {
        Err(PyValueError::new_err(format!(
          "the leverage surface was calibrated at r={r0}, q={q0} but the query is at r={r}, q={q}; \
         recalibrate at the query rates"
        )))
      }
      _ => Ok(()),
    }
  }
}

#[pyclass(name = "HestonSlvCalibrationResult")]
pub struct PyHestonSlvCalibrationResult {
  inner: HestonSlvCalibrationResult,
}

#[pymethods]
impl PyHestonSlvCalibrationResult {
  /// `(kappa, theta, sigma, rho, v0, eta)`.
  fn params(&self) -> (f64, f64, f64, f64, f64, f64) {
    params_tuple(&self.inner.fit.params)
  }

  fn leverage(&self) -> PyLeverageSurface {
    PyLeverageSurface {
      inner: self.inner.fit.leverage.clone(),
    }
  }

  /// The Heston fit behind the parameters as
  /// `(v0, kappa, theta, sigma, rho, converged, rmse)`, `None` when they
  /// were pinned.
  fn heston(&self) -> Option<(f64, f64, f64, f64, f64, bool, f64)> {
    self.inner.heston.as_ref().map(|fit| {
      let p = &fit.params;
      (
        p.v0,
        p.kappa,
        p.theta,
        p.sigma,
        p.rho,
        fit.converged,
        fit.rmse(),
      )
    })
  }

  #[getter]
  fn rmse(&self) -> f64 {
    self.inner.rmse()
  }

  #[getter]
  fn max_error(&self) -> f64 {
    self.inner.max_error
  }

  #[getter]
  fn converged(&self) -> bool {
    self.inner.converged
  }

  /// The rates the leverage is anchored to.
  #[getter]
  fn rates(&self) -> (f64, f64) {
    (self.inner.r, self.inner.q)
  }

  /// The Monte Carlo pricer of the calibrated model, anchored to the
  /// calibration rates.
  #[pyo3(signature = (n_paths=100_000, steps_per_year=200, seed=42))]
  fn to_model(&self, n_paths: usize, steps_per_year: usize, seed: u64) -> PyHestonSlvPricer {
    PyHestonSlvPricer {
      inner: self
        .inner
        .to_model(self.inner.r, self.inner.q)
        .with_paths(n_paths)
        .with_steps_per_year(steps_per_year)
        .with_seed(seed),
    }
  }
}

#[pyclass(name = "HestonSlvCalibrator")]
pub struct PyHestonSlvCalibrator {
  inner: HestonSlvCalibrator,
}

#[pymethods]
impl PyHestonSlvCalibrator {
  /// `calls[j, i]` is the present call price at `strikes[i]`, `maturities[j]`.
  /// `heston` pins `(v0, kappa, theta, sigma, rho)` — the order
  /// `HestonCalibrator.calibrate` returns — and `heston_initial_guess` seeds
  /// the fit instead; `local_vol` supplies the local volatility on the same
  /// grid instead of a Dupire read of the calls. `method` is `"particle"`
  /// (the Guyon–Henry-Labordère cloud, tuned by `n_particles`, `seed` and
  /// the bandwidth constants) or `"fokker_planck"` (the finite-volume
  /// forward Kolmogorov equation after Wyns & Du Toit, tuned by
  /// `log_spot_nodes`, `variance_nodes` and `inner_iterations`);
  /// `steps_per_year` serves both.
  #[new]
  #[pyo3(signature = (s, r, q, strikes, maturities, calls, eta=1.0, heston=None, heston_initial_guess=None, local_vol=None, dupire_eps=1e-6, method="particle", n_particles=100_000, steps_per_year=200, seed=42, bandwidth_factor=1.5, bandwidth_t_min=0.25, log_spot_nodes=201, variance_nodes=100, inner_iterations=2))]
  fn new(
    s: f64,
    r: f64,
    q: f64,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    calls: PyReadonlyArray2<'_, f64>,
    eta: f64,
    heston: Option<(f64, f64, f64, f64, f64)>,
    heston_initial_guess: Option<(f64, f64, f64, f64, f64)>,
    local_vol: Option<PyReadonlyArray2<'_, f64>>,
    dupire_eps: f64,
    method: &str,
    n_particles: usize,
    steps_per_year: usize,
    seed: u64,
    bandwidth_factor: f64,
    bandwidth_t_min: f64,
    log_spot_nodes: usize,
    variance_nodes: usize,
    inner_iterations: usize,
  ) -> PyResult<Self> {
    let heston_params = |(v0, kappa, theta, sigma, rho): (f64, f64, f64, f64, f64)| HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    };
    let calls = calls.as_array().to_owned();
    let mut inner = HestonSlvCalibrator::new(s, r, q, strikes, maturities, calls)
      .with_mixing(eta)
      .with_dupire_eps(dupire_eps);
    inner = match method.to_ascii_lowercase().as_str() {
      "particle" => inner.with_particle_method(
        ParticleMethod::default()
          .with_particles(n_particles)
          .with_steps_per_year(steps_per_year)
          .with_seed(seed)
          .with_bandwidth_factor(bandwidth_factor)
          .with_bandwidth_t_min(bandwidth_t_min),
      ),
      "fokker_planck" | "fokker-planck" | "pde" => inner.with_fokker_planck(
        FokkerPlanckMethod::default()
          .with_nodes(log_spot_nodes, variance_nodes)
          .with_steps_per_year(steps_per_year)
          .with_inner_iterations(inner_iterations),
      ),
      other => {
        return Err(PyValueError::new_err(format!(
          "method must be 'particle' or 'fokker_planck', got '{other}'"
        )));
      }
    };
    if let Some(pinned) = heston {
      inner = inner.with_heston_params(heston_params(pinned));
    }
    if let Some(guess) = heston_initial_guess {
      inner = inner.with_heston_initial_guess(heston_params(guess));
    }
    if let Some(lv) = local_vol {
      let lv = lv.as_array().to_owned();
      inner = inner.with_local_vol(lv);
    }
    Ok(Self { inner })
  }

  fn calibrate(&self) -> PyResult<PyHestonSlvCalibrationResult> {
    let inner = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Heston SLV calibration failed: {e}")))?;
    Ok(PyHestonSlvCalibrationResult { inner })
  }

  /// Calibrate and return the pricer of the calibrated model.
  #[pyo3(signature = (n_paths=100_000, steps_per_year=200, seed=42))]
  fn calibrate_to_model(
    &self,
    n_paths: usize,
    steps_per_year: usize,
    seed: u64,
  ) -> PyResult<PyHestonSlvPricer> {
    Ok(self.calibrate()?.to_model(n_paths, steps_per_year, seed))
  }

  /// The strikes of the input grid.
  #[getter]
  fn strikes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(self.inner.strikes.clone()).into_pyarray(py)
  }

  /// The maturities of the input grid.
  #[getter]
  fn maturities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(self.inner.maturities.clone()).into_pyarray(py)
  }
}

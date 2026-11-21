use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::calibration_basic::PyMarketSlice;
use super::calibration_basic::quotes;
use super::calibration_basic::quotes_as;
use super::parse_option_type;

#[pyclass(name = "SVJCalibrator", unsendable)]
pub struct PySvjCalibrator {
  inner: crate::calibration::svj::SvjCalibrator,
}

#[pymethods]
impl PySvjCalibrator {
  /// Bates / SVJ joint multi-maturity calibration (Heston + Merton jumps).
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let quotes = quotes_as(slices, parse_option_type(option_type)?);
      Ok(Self {
        inner: crate::calibration::svj::SvjCalibrator::new(s, r, q.unwrap_or(0.0), quotes),
      })
    })?
  }

  /// Returns `(v0, kappa, theta, sigma_v, rho, lambda, mu_j, sigma_j, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SVJ calibration failed: {e}")))?;
    let p = fit.params;
    Ok((
      p.v0,
      p.kappa,
      p.theta,
      p.sigma,
      p.rho,
      p.lambda,
      p.mu_j,
      p.sigma_j,
      fit.converged,
      fit.rmse(),
    ))
  }
}

#[pyclass(name = "DoubleHestonCalibrator", unsendable)]
pub struct PyDoubleHestonCalibrator {
  inner: crate::calibration::double_heston::DoubleHestonCalibrator,
}

#[pymethods]
impl PyDoubleHestonCalibrator {
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::calibration::double_heston::DoubleHestonCalibrator;
      let quotes = quotes_as(slices, parse_option_type(option_type)?);
      Ok(Self {
        inner: DoubleHestonCalibrator::new(s, r, q.unwrap_or(0.0), quotes),
      })
    })?
  }

  /// Returns `(v1_0, kappa1, theta1, sigma1, rho1, v2_0, kappa2, theta2, sigma2, rho2, converged, loss_rmse)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Double-Heston calibration failed: {e}")))?;
    let (a, b) = (fit.params.first, fit.params.second);
    Ok((
      a.v0,
      a.kappa,
      a.theta,
      a.sigma,
      a.rho,
      b.v0,
      b.kappa,
      b.theta,
      b.sigma,
      b.rho,
      fit.converged,
      fit.rmse(),
    ))
  }
}

#[pyclass(name = "LevyCalibrator", unsendable)]
pub struct PyLevyCalibrator {
  inner: crate::calibration::levy::LevyCalibrator,
}

#[pymethods]
impl PyLevyCalibrator {
  /// `model`: one of "vg", "nig", "cgmy", "merton_jd", "kou".
  #[new]
  #[pyo3(signature = (slices, s, r, q, model))]
  fn new(slices: Vec<PyMarketSlice>, s: f64, r: f64, q: f64, model: &str) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::calibration::levy::LevyModelType;
      let mt = match model.to_ascii_lowercase().as_str() {
        "vg" | "variance_gamma" => LevyModelType::VarianceGamma,
        "nig" => LevyModelType::Nig,
        "cgmy" => LevyModelType::Cgmy,
        "merton_jd" | "merton" | "mjd" => LevyModelType::MertonJd,
        "kou" => LevyModelType::Kou,
        o => {
          return Err(PyValueError::new_err(format!(
            "model must be one of vg/nig/cgmy/merton_jd/kou, got '{o}'"
          )));
        }
      };
      Ok(Self {
        inner: crate::calibration::levy::LevyCalibrator::new(mt, s, r, q, quotes(slices)),
      })
    })?
  }

  /// Returns `(params_vec, converged, loss_rmse, iterations)`.
  fn calibrate(&self) -> PyResult<(Vec<f64>, bool, f64, usize)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Lévy calibration failed: {e}")))?;
    let iterations = fit.iterations.unwrap_or_default();
    Ok((
      fit.params.values.clone(),
      fit.converged,
      fit.rmse(),
      iterations,
    ))
  }
}

#[pyclass(name = "HKDECalibrator", unsendable)]
pub struct PyHkdeCalibrator {
  inner: crate::calibration::hkde::HkdeCalibrator,
}

#[pymethods]
impl PyHkdeCalibrator {
  /// Heston + Kou double-exponential jump joint calibration over multiple maturities.
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let quotes = quotes_as(slices, parse_option_type(option_type)?);
      Ok(Self {
        inner: crate::calibration::hkde::HkdeCalibrator::new(s, r, q.unwrap_or(0.0), quotes),
      })
    })?
  }

  /// Returns `(v0, kappa, theta, sigma_v, rho, lambda, p_up, eta1, eta2, converged, loss_rmse)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("HKDE calibration failed: {e}")))?;
    let p = fit.params;
    Ok((
      p.v0,
      p.kappa,
      p.theta,
      p.sigma,
      p.rho,
      p.lambda,
      p.p_up,
      p.eta1,
      p.eta2,
      fit.converged,
      fit.rmse(),
    ))
  }
}

/// Rough Bergomi calibrator (Wasserstein-1 path matching, projected Adam).
#[pyclass(name = "RBergomiCalibrator", unsendable)]
pub struct PyRBergomiCalibrator {
  inner: crate::calibration::rbergomi::RBergomiCalibrator,
}

#[pymethods]
impl PyRBergomiCalibrator {
  /// `slices`: list of `(maturity, terminal_samples)` pairs.
  /// Initial guess uses Hurst=0.1, rho=-0.7, eta=2.0, xi0=Constant(0.04).
  #[new]
  #[pyo3(signature = (s0, r, slices, hurst=0.1, rho=-0.7, eta=2.0, xi0=0.04, max_iters=60, paths=1024))]
  fn new(
    s0: f64,
    r: f64,
    slices: Vec<(f64, Vec<f64>)>,
    hurst: f64,
    rho: f64,
    eta: f64,
    xi0: f64,
    max_iters: usize,
    paths: usize,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      use crate::calibration::rbergomi::*;
      let inner_slices: Vec<RBergomiMarketSlice> = slices
        .into_iter()
        .map(|(tau, terminal_samples)| RBergomiMarketSlice {
          tau,
          terminal_samples,
        })
        .collect();
      let params = RBergomiParams {
        hurst,
        rho,
        eta,
        xi0: RBergomiXi0::Constant(xi0),
      };
      let config = RBergomiCalibrationConfig {
        paths,
        ..RBergomiCalibrationConfig::default()
      };
      let options =
        crate::calibration::CalibrationOptions::default().with_max_iterations(max_iters);
      let inner =
        RBergomiCalibrator::new(s0, r, 0.0, params, inner_slices, config).with_options(options);
      Ok(Self { inner })
    })?
  }

  /// Returns `(hurst, rho, eta, xi0_const, final_loss, iterations, converged)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, usize, bool)> {
    use crate::calibration::rbergomi::RBergomiXi0;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("rBergomi calibration failed: {e}")))?;
    let p = &fit.params;
    let xi0_const = match &p.xi0 {
      RBergomiXi0::Constant(c) => *c,
      _ => 0.0,
    };
    Ok((
      p.hurst,
      p.rho,
      p.eta,
      xi0_const,
      fit.loss.get(crate::LossMetric::Mse),
      fit.iterations.unwrap_or_default(),
      fit.converged,
    ))
  }
}

#[pyclass(name = "CgmysvCalibrator", unsendable)]
pub struct PyCgmysvCalibrator {
  inner: crate::calibration::cgmysv::CgmysvCalibrator,
}

#[pymethods]
impl PyCgmysvCalibrator {
  /// CGMYSV (Carr-Geman-Madan-Yor + stochastic volatility) calibrator
  /// via Lewis Fourier pricing + Levenberg-Marquardt.
  #[new]
  fn new(slices: Vec<PyMarketSlice>, s: f64, r: f64, q: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::calibration::cgmysv::CgmysvCalibrator::new(s, r, q, quotes(slices)),
    })
  }

  /// Returns `(alpha, lambda_plus, lambda_minus, kappa, eta, zeta, rho, v0, converged, loss_rmse, iterations)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, bool, f64, usize)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("CGMYSV calibration failed: {e}")))?;
    let p = &fit.params;
    Ok((
      p.alpha,
      p.lambda_plus,
      p.lambda_minus,
      p.kappa,
      p.eta,
      p.zeta,
      p.rho,
      p.v0,
      fit.converged,
      fit.rmse(),
      fit.iterations.unwrap_or_default(),
    ))
  }
}

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::calibration_basic::PyMarketSlice;
use super::parse_option_type;

#[pyclass(name = "SVJCalibrator", unsendable)]
pub struct PySVJCalibrator {
  inner: crate::calibration::svj::SVJCalibrator,
}

#[pymethods]
impl PySVJCalibrator {
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
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Ok(Self {
      inner: crate::calibration::svj::SVJCalibrator::from_slices(
        None,
        &inner_slices,
        s,
        r,
        q,
        ot,
        false,
      ),
    })
  }

  /// Returns `(v0, kappa, theta, sigma_v, rho, lambda, mu_j, sigma_j, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SVJ calibration failed: {e}")))?;
    Ok((
      res.v0,
      res.kappa,
      res.theta,
      res.sigma_v,
      res.rho,
      res.lambda,
      res.mu_j,
      res.sigma_j,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
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
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Ok(Self {
      inner: crate::calibration::double_heston::DoubleHestonCalibrator::from_slices(
        None,
        &inner_slices,
        s,
        r,
        q,
        ot,
        false,
      ),
    })
  }

  /// Returns `(v1_0, kappa1, theta1, sigma1, rho1, v2_0, kappa2, theta2, sigma2, rho2, converged, loss_rmse)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Double-Heston calibration failed: {e}")))?;
    Ok((
      res.v1_0,
      res.kappa1,
      res.theta1,
      res.sigma1,
      res.rho1,
      res.v2_0,
      res.kappa2,
      res.theta2,
      res.sigma2,
      res.rho2,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
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
    use crate::calibration::levy::LevyModelType;
    let mt = match model.to_ascii_lowercase().as_str() {
      "vg" | "variance_gamma" => LevyModelType::VarianceGamma,
      "nig" => LevyModelType::Nig,
      "cgmy" => LevyModelType::Cgmy,
      "merton_jd" | "merton" | "mjd" => LevyModelType::MertonJD,
      "kou" => LevyModelType::Kou,
      o => {
        return Err(PyValueError::new_err(format!(
          "model must be one of vg/nig/cgmy/merton_jd/kou, got '{o}'"
        )));
      }
    };
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Ok(Self {
      inner: crate::calibration::levy::LevyCalibrator::new(mt, s, r, q, inner_slices),
    })
  }

  /// Returns `(params_vec, converged, loss_rmse, iterations)`.
  fn calibrate(&self) -> PyResult<(Vec<f64>, bool, f64, usize)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Lévy calibration failed: {e}")))?;
    Ok((
      res.params,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
      res.iterations,
    ))
  }
}

#[pyclass(name = "HKDECalibrator", unsendable)]
pub struct PyHKDECalibrator {
  inner: crate::calibration::hkde::HKDECalibrator,
}

#[pymethods]
impl PyHKDECalibrator {
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
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Ok(Self {
      inner: crate::calibration::hkde::HKDECalibrator::from_slices(
        None,
        &inner_slices,
        s,
        r,
        q,
        ot,
        false,
      ),
    })
  }

  /// Returns `(v0, kappa, theta, sigma_v, rho, lambda, p_up, eta1, eta2, converged, loss_rmse)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
    let res = self.inner.calibrate(None);
    Ok((
      res.v0,
      res.kappa,
      res.theta,
      res.sigma_v,
      res.rho,
      res.lambda,
      res.p_up,
      res.eta1,
      res.eta2,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
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
    use crate::calibration::rbergomi::*;
    let inner_slices: Vec<RBergomiMarketSlice> = slices
      .into_iter()
      .map(|(maturity, terminal_samples)| RBergomiMarketSlice {
        maturity,
        terminal_samples,
      })
      .collect();
    let params = RBergomiParams {
      hurst,
      rho,
      eta,
      xi0: RBergomiXi0::Constant(xi0),
    };
    let mut cfg = RBergomiCalibrationConfig {
      max_iters,
      paths,
      ..RBergomiCalibrationConfig::default()
    };
    cfg.paths = paths;
    let inner = RBergomiCalibrator::new(s0, r, params, inner_slices, cfg, false).map_err(|e| {
      PyValueError::new_err(format!("RBergomi calibrator construction failed: {e}"))
    })?;
    Ok(Self { inner })
  }

  /// Returns `(hurst, rho, eta, xi0_const, final_loss, iterations, converged)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, usize, bool)> {
    use crate::calibration::rbergomi::RBergomiXi0;
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("rBergomi calibration failed: {e}")))?;
    let p = &res.calibrated_params;
    let xi0_const = match &p.xi0 {
      RBergomiXi0::Constant(c) => *c,
      _ => 0.0,
    };
    Ok((
      p.hurst,
      p.rho,
      p.eta,
      xi0_const,
      res.final_loss,
      res.iterations,
      res.converged,
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
  fn new(slices: Vec<PyMarketSlice>, s: f64, r: f64, q: f64) -> Self {
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    Self {
      inner: crate::calibration::cgmysv::CgmysvCalibrator::new(s, r, q, inner_slices),
    }
  }

  /// Returns `(alpha, lambda_plus, lambda_minus, kappa, eta, zeta, rho, v0, converged, loss_rmse, iterations)`.
  #[allow(clippy::type_complexity)]
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, bool, f64, usize)> {
    let res = self.inner.calibrate(None);
    let p = &res.params;
    Ok((
      p.alpha,
      p.lambda_plus,
      p.lambda_minus,
      p.kappa,
      p.eta,
      p.zeta,
      p.rho,
      p.v0,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
      res.iterations,
    ))
  }
}

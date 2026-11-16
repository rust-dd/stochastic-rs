use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::parse_option_type;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::levy::MarketSlice;

#[pyclass(name = "MarketSlice", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyMarketSlice {
  pub inner: Vec<OptionQuote>,
}

#[pymethods]
impl PyMarketSlice {
  #[new]
  fn new(strikes: Vec<f64>, prices: Vec<f64>, is_call: Vec<bool>, t: f64) -> pyo3::PyResult<Self> {
    if strikes.len() != prices.len() || strikes.len() != is_call.len() {
      return Err(PyValueError::new_err(format!(
        "strikes, prices and is_call must have equal length, got {}, {} and {}",
        strikes.len(),
        prices.len(),
        is_call.len()
      )));
    }
    let side = |call: bool| {
      if call {
        OptionType::Call
      } else {
        OptionType::Put
      }
    };
    let inner = strikes
      .into_iter()
      .zip(prices)
      .zip(is_call)
      .map(|((k, price), call)| OptionQuote::new(k, t, price, side(call)))
      .collect();
    Ok(Self { inner })
  }
}

/// Every slice's quotes in slice order, priced as `option_type`, the class's own argument.
pub(super) fn quotes_as(slices: Vec<PyMarketSlice>, option_type: OptionType) -> Vec<OptionQuote> {
  let quotes = slices.into_iter().flat_map(|slice| slice.inner);
  quotes
    .map(|quote| OptionQuote {
      option_type,
      ..quote
    })
    .collect()
}

/// One single-quote `MarketSlice` per quote, for the calibrators that flatten slices themselves.
pub(super) fn market_slices(slices: Vec<PyMarketSlice>) -> Vec<MarketSlice> {
  let quotes = slices.into_iter().flat_map(|slice| slice.inner);
  quotes
    .map(|quote| MarketSlice {
      strikes: vec![quote.strike],
      prices: vec![quote.price],
      is_call: vec![quote.option_type == OptionType::Call],
      tau: quote.tau,
    })
    .collect()
}

#[pyclass(name = "BSMCalibrator", unsendable)]
pub struct PyBsmCalibrator {
  inner: crate::calibration::bsm::BsmCalibrator,
}

#[pymethods]
impl PyBsmCalibrator {
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None, sigma_init=0.2))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
    sigma_init: f64,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let quotes = quotes_as(slices, parse_option_type(option_type)?);
      let start = crate::calibration::bsm::BsmParams { sigma: sigma_init };
      Ok(Self {
        inner: crate::calibration::bsm::BsmCalibrator::new(s, r, q.unwrap_or(0.0), quotes)
          .with_start(start),
      })
    })?
  }

  /// Run calibration. Returns `(sigma, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, bool, f64)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("BSM calibration failed: {e}")))?;
    Ok((fit.params.sigma, fit.converged, fit.rmse()))
  }
}

#[pyclass(name = "HestonCalibrator", unsendable)]
pub struct PyHestonCalibrator {
  inner: crate::calibration::heston::HestonCalibrator,
}

#[pymethods]
impl PyHestonCalibrator {
  /// `regularization`: optional `(anchor, weights)` Tikhonov pull on `(v0, kappa, theta, sigma, rho)`; `jacobian_method` is
  /// `"cui"` or `"fd"`; `residual_weights` holds one positive weight per quote in slice order.
  #[new]
  #[pyo3(signature = (slices, s, r, option_type="call", q=None, regularization=None, jacobian_method="cui", residual_weights=None, feller_projection=false))]
  fn new(
    slices: Vec<PyMarketSlice>,
    s: f64,
    r: f64,
    option_type: &str,
    q: Option<f64>,
    regularization: Option<(Vec<f64>, Vec<f64>)>,
    jacobian_method: &str,
    residual_weights: Option<Vec<f64>>,
    feller_projection: bool,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let quotes = quotes_as(slices, parse_option_type(option_type)?);
      let mut options = crate::calibration::CalibrationOptions::default();
      if let Some((anchor, weights)) = regularization {
        options =
          options.with_regularization(crate::calibration::Regularization::new(anchor, weights));
      }
      let mut inner =
        crate::calibration::heston::HestonCalibrator::new(s, r, q.unwrap_or(0.0), quotes)
          .with_options(options)
          .with_jacobian_method(parse_heston_jacobian_method(jacobian_method)?)
          .with_feller_projection(feller_projection);
      if let Some(weights) = residual_weights {
        inner = inner.with_residual_weights(weights);
      }
      Ok(Self { inner })
    })?
  }

  /// `initial_guess` is `(v0, kappa, theta, sigma, rho)`. Returns `(v0, kappa, theta, sigma, rho, converged, loss_rmse)`,
  /// with the stop reason (`"converged"`, `"evaluation budget exhausted"`, …) appended when `full_output` is true.
  #[pyo3(signature = (initial_guess=None, full_output=false))]
  fn calibrate(
    &self,
    py: Python<'_>,
    initial_guess: Option<[f64; 5]>,
    full_output: bool,
  ) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;

    use crate::pricing::heston::HestonParams;
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let initial_guess = initial_guess.map(|[v0, kappa, theta, sigma, rho]| HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    });
    let res = self
      .inner
      .calibrate(initial_guess)
      .map_err(|e| PyValueError::new_err(format!("Heston calibration failed: {e}")))?;
    let p = &res.params;
    let rmse = res.rmse();
    if full_output {
      (
        p.v0,
        p.kappa,
        p.theta,
        p.sigma,
        p.rho,
        res.converged,
        rmse,
        res.message(),
      )
        .into_py_any(py)
    } else {
      (p.v0, p.kappa, p.theta, p.sigma, p.rho, res.converged, rmse).into_py_any(py)
    }
  }
}

fn parse_heston_jacobian_method(
  s: &str,
) -> PyResult<crate::calibration::heston::HestonJacobianMethod> {
  use crate::calibration::heston::HestonJacobianMethod;
  match s.to_ascii_lowercase().as_str() {
    "cui" => Ok(HestonJacobianMethod::CuiAnalytic),
    "fd" => Ok(HestonJacobianMethod::NumericFiniteDiff),
    other => Err(PyValueError::new_err(format!(
      "jacobian_method must be 'cui' or 'fd', got '{other}'"
    ))),
  }
}

#[pyclass(name = "SabrCalibrator", unsendable)]
pub struct PySabrCalibrator {
  inner: crate::calibration::sabr::SabrCalibrator,
}

#[pymethods]
impl PySabrCalibrator {
  /// Single-maturity SABR calibration (β fixed at 1.0 by default).
  /// `regularization`: optional `(anchor, weights)` Tikhonov pull on `(alpha, nu, rho)`.
  #[new]
  #[pyo3(signature = (strikes, prices, s, r, tau, option_type="call", q=None, regularization=None))]
  fn new(
    strikes: Vec<f64>,
    prices: Vec<f64>,
    s: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
    regularization: Option<(Vec<f64>, Vec<f64>)>,
  ) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      if strikes.len() != prices.len() {
        return Err(PyValueError::new_err(
          "strikes and prices must have equal length",
        ));
      }
      let ot = parse_option_type(option_type)?;
      let quotes = strikes
        .iter()
        .zip(&prices)
        .map(|(&k, &price)| crate::OptionQuote::new(k, tau, price, ot))
        .collect();
      let mut options = crate::calibration::CalibrationOptions::default();
      if let Some((anchor, weights)) = regularization {
        options = options.with_regularization(crate::calibration::Regularization::new(
          hagan_order(anchor),
          hagan_order(weights),
        ));
      }
      let inner = crate::calibration::sabr::SabrCalibrator::new(s, r, q.unwrap_or(0.0), quotes)
        .with_options(options);
      Ok(Self { inner })
    })?
  }

  /// Returns `(alpha, beta, nu, rho, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool, f64)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR calibration failed: {e}")))?;
    let p = fit.params;
    Ok((p.alpha, p.beta, p.nu, p.rho, fit.converged, fit.rmse()))
  }
}

/// Python's `(alpha, nu, rho)` in the Rust anchor's Hagan order `(alpha, rho, nu)`; any other
/// length reaches the calibrator unchanged, which reports the count.
fn hagan_order(v: Vec<f64>) -> Vec<f64> {
  if let [alpha, nu, rho] = v[..] {
    vec![alpha, rho, nu]
  } else {
    v
  }
}

#[pyclass(name = "SabrCapletCalibrator", unsendable)]
pub struct PySabrCapletCalibrator {
  inner: crate::calibration::sabr_caplet::SabrCapletCalibrator,
}

#[pymethods]
impl PySabrCapletCalibrator {
  /// SABR caplet smile calibrator: fits `(α, ν, ρ)` for one expiry at a fixed β, from `ν = 0.4`,
  /// `ρ = −0.2` and `α = σ F^{1−β}` with `σ` the vol at the strike nearest the forward.
  #[new]
  fn new(
    forward: f64,
    expiry: f64,
    beta: f64,
    strikes: Vec<f64>,
    market_vols: Vec<f64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let start = caplet_start(forward, beta, &strikes, &market_vols)?;
      Ok(Self {
        inner: crate::calibration::sabr_caplet::SabrCapletCalibrator::new(
          forward,
          expiry,
          strikes,
          market_vols,
          start,
        ),
      })
    })?
  }

  /// Returns `(alpha, beta, nu, rho, rmse, converged)`; `converged` means the simplex cost spread
  /// fell below 1e-10, and reaching 600 iterations or a nonfinite cost gives `false`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, bool)> {
    use crate::traits::CalibrationResult;
    use crate::traits::Calibrator;
    let fit = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR caplet calibration failed: {e}")))?;
    let p = fit.params.sabr;
    Ok((p.alpha, p.beta, p.nu, p.rho, fit.rmse(), fit.converged))
  }
}

fn caplet_start(
  forward: f64,
  beta: f64,
  strikes: &[f64],
  vols: &[f64],
) -> PyResult<crate::calibration::sabr_caplet::SabrCapletParams> {
  let atm = strikes
    .iter()
    .zip(vols)
    .min_by(|a, b| (a.0 - forward).abs().total_cmp(&(b.0 - forward).abs()))
    .map_or(0.2, |(_, &vol)| vol);
  let alpha = atm * forward.powf(1.0 - beta);
  if !(alpha.is_finite() && alpha > 0.0) {
    return Err(PyValueError::new_err(format!(
      "forward {forward} and the vol nearest it give no positive starting alpha, got {alpha}"
    )));
  }
  Ok(crate::calibration::sabr_caplet::SabrCapletParams {
    sabr: crate::pricing::sabr::SabrParams {
      alpha,
      beta,
      rho: -0.2,
      nu: 0.4,
    },
    shift: 0.0,
  })
}

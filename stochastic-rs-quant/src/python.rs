//! PyO3 wrappers for `stochastic-rs-quant`.
//!
//! Pricing engines (analytic + Fourier), calibrators, vol-surface parameterisations,
//! and bond-pricing closed forms exposed as `#[pyclass]` types. Registered by the
//! `stochastic-rs-py` cdylib.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::OptionType;
use crate::traits::ModelPricer;
use crate::traits::PricerExt;

fn parse_option_type(s: &str) -> PyResult<OptionType> {
  match s.to_ascii_lowercase().as_str() {
    "c" | "call" => Ok(OptionType::Call),
    "p" | "put" => Ok(OptionType::Put),
    other => Err(PyValueError::new_err(format!(
      "option_type must be 'call' or 'put', got '{other}'"
    ))),
  }
}

#[pyclass(name = "BSMPricer", unsendable)]
pub struct PyBSMPricer {
  inner: crate::pricing::bsm::BSMPricer,
}

#[pymethods]
impl PyBSMPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, option_type="call", q=None))]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let inner = crate::pricing::bsm::BSMPricer::new(
      s,
      v,
      k,
      r,
      None,
      None,
      q,
      Some(tau),
      None,
      None,
      ot,
      crate::pricing::bsm::BSMCoc::default(),
    );
    Ok(Self { inner })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
  fn delta(&self) -> f64 {
    self.inner.delta()
  }
  fn gamma(&self) -> f64 {
    self.inner.gamma()
  }
  fn vega(&self) -> f64 {
    self.inner.vega()
  }
  fn theta(&self) -> f64 {
    self.inner.theta()
  }
  fn rho(&self) -> f64 {
    self.inner.rho()
  }
  fn vanna(&self) -> f64 {
    self.inner.vanna()
  }
  fn charm(&self) -> f64 {
    self.inner.charm()
  }
  fn implied_volatility(&self, c_price: f64, option_type: &str) -> PyResult<f64> {
    let ot = parse_option_type(option_type)?;
    Ok(self.inner.implied_volatility(c_price, ot))
  }
}

#[pyclass(name = "HestonPricer", unsendable)]
pub struct PyHestonPricer {
  inner: crate::pricing::heston::HestonPricer,
}

#[pymethods]
impl PyHestonPricer {
  #[new]
  #[pyo3(signature = (s, v0, k, r, kappa, theta, sigma, rho, tau, q=None, lambda_=None))]
  fn new(
    s: f64,
    v0: f64,
    k: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    tau: f64,
    q: Option<f64>,
    lambda_: Option<f64>,
  ) -> Self {
    let inner = crate::pricing::heston::HestonPricer::new(
      s,
      v0,
      k,
      r,
      q,
      rho,
      kappa,
      theta,
      sigma,
      lambda_,
      Some(tau),
      None,
      None,
    );
    Self { inner }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "SabrPricer", unsendable)]
pub struct PySabrPricer {
  inner: crate::pricing::sabr::SabrPricer,
}

#[pymethods]
impl PySabrPricer {
  #[new]
  #[pyo3(signature = (s, k, r, alpha, beta, nu, rho, tau, q=None))]
  fn new(
    s: f64,
    k: f64,
    r: f64,
    alpha: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    tau: f64,
    q: Option<f64>,
  ) -> Self {
    let inner = crate::pricing::sabr::SabrPricer::new(
      s,
      k,
      r,
      q,
      alpha,
      beta,
      nu,
      rho,
      Some(tau),
      None,
      None,
    );
    Self { inner }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "Merton1976Pricer", unsendable)]
pub struct PyMerton1976Pricer {
  inner: crate::pricing::merton_jump::Merton1976Pricer,
}

#[pymethods]
impl PyMerton1976Pricer {
  /// `m` is the Poisson-series truncation iteration limit (default 50).
  #[new]
  #[pyo3(signature = (s, v, k, r, lambda_, gamma, tau, option_type="call", q=None, m=50))]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    lambda_: f64,
    gamma: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
    m: usize,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    let mut builder =
      crate::pricing::merton_jump::Merton1976Pricer::builder(s, v, k, r, lambda_, gamma, m)
        .tau(tau)
        .option_type(ot);
    if let Some(qv) = q {
      builder = builder.q(qv);
    }
    Ok(Self {
      inner: builder.build(),
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "BSMFourier", unsendable)]
pub struct PyBSMFourier {
  pub inner: crate::pricing::fourier::BSMFourier,
}

#[pymethods]
impl PyBSMFourier {
  #[new]
  fn new(sigma: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::BSMFourier { sigma, r, q },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "HestonFourier", unsendable)]
pub struct PyHestonFourier {
  pub inner: crate::pricing::fourier::HestonFourier,
}

#[pymethods]
impl PyHestonFourier {
  #[new]
  fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::HestonFourier {
        v0,
        kappa,
        theta,
        sigma,
        rho,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "VarianceGammaFourier", unsendable)]
pub struct PyVarianceGammaFourier {
  pub inner: crate::pricing::fourier::VarianceGammaFourier,
}

#[pymethods]
impl PyVarianceGammaFourier {
  #[new]
  fn new(sigma: f64, theta: f64, nu: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::VarianceGammaFourier {
        sigma,
        theta,
        nu,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "CGMYFourier", unsendable)]
pub struct PyCGMYFourier {
  pub inner: crate::pricing::fourier::CGMYFourier,
}

#[pymethods]
impl PyCGMYFourier {
  #[new]
  fn new(c: f64, g: f64, m: f64, y: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::CGMYFourier { c, g, m, y, r, q },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "MertonJDFourier", unsendable)]
pub struct PyMertonJDFourier {
  pub inner: crate::pricing::fourier::MertonJDFourier,
}

#[pymethods]
impl PyMertonJDFourier {
  #[new]
  fn new(sigma: f64, lambda: f64, mu_j: f64, sigma_j: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::MertonJDFourier {
        sigma,
        lambda,
        mu_j,
        sigma_j,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "KouFourier", unsendable)]
pub struct PyKouFourier {
  pub inner: crate::pricing::fourier::KouFourier,
}

#[pymethods]
impl PyKouFourier {
  #[new]
  fn new(sigma: f64, lambda: f64, p_up: f64, eta1: f64, eta2: f64, r: f64, q: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::KouFourier {
        sigma,
        lambda,
        p_up,
        eta1,
        eta2,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "BatesFourier", unsendable)]
pub struct PyBatesFourier {
  pub inner: crate::pricing::fourier::BatesFourier,
}

#[pymethods]
impl PyBatesFourier {
  #[new]
  fn new(
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    lambda: f64,
    mu_j: f64,
    sigma_j: f64,
    r: f64,
    q: f64,
  ) -> Self {
    Self {
      inner: crate::pricing::fourier::BatesFourier {
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        lambda,
        mu_j,
        sigma_j,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "CarrMadanPricer", unsendable)]
pub struct PyCarrMadanPricer {
  inner: crate::pricing::fourier::CarrMadanPricer,
}

#[pymethods]
impl PyCarrMadanPricer {
  #[new]
  #[pyo3(signature = (n=4096, alpha=0.75))]
  fn new(n: usize, alpha: f64) -> Self {
    Self {
      inner: crate::pricing::fourier::CarrMadanPricer::new(n, alpha),
    }
  }

  fn price_heston_call(&self, model: &PyHestonFourier, s: f64, k: f64, r: f64, tau: f64) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, tau)
  }
  fn price_bates_call(&self, model: &PyBatesFourier, s: f64, k: f64, r: f64, tau: f64) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, tau)
  }
  fn price_kou_call(&self, model: &PyKouFourier, s: f64, k: f64, r: f64, tau: f64) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, tau)
  }
}

#[pyclass(name = "VasicekBond", unsendable)]
pub struct PyVasicekBond {
  inner: crate::bonds::vasicek::Vasicek,
}

#[pymethods]
impl PyVasicekBond {
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> Self {
    Self {
      inner: crate::bonds::vasicek::Vasicek {
        r_t,
        theta,
        mu,
        sigma,
        tau,
        eval: None,
        expiration: None,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}

#[pyclass(name = "CIRBond", unsendable)]
pub struct PyCIRBond {
  inner: crate::bonds::cir::Cir,
}

#[pymethods]
impl PyCIRBond {
  #[new]
  fn new(r_t: f64, theta: f64, mu: f64, sigma: f64, tau: f64) -> Self {
    Self {
      inner: crate::bonds::cir::Cir {
        r_t,
        theta,
        mu,
        sigma,
        tau,
        eval: None,
        expiration: None,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
}

#[pyclass(name = "MarketSlice", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyMarketSlice {
  pub inner: crate::calibration::levy::MarketSlice,
}

#[pymethods]
impl PyMarketSlice {
  #[new]
  fn new(strikes: Vec<f64>, prices: Vec<f64>, is_call: Vec<bool>, t: f64) -> Self {
    Self {
      inner: crate::calibration::levy::MarketSlice {
        strikes,
        prices,
        is_call,
        t,
      },
    }
  }
}

#[pyclass(name = "BSMCalibrator", unsendable)]
pub struct PyBSMCalibrator {
  inner: crate::calibration::bsm::BSMCalibrator,
}

#[pymethods]
impl PyBSMCalibrator {
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
    let ot = parse_option_type(option_type)?;
    let inner_slices: Vec<crate::calibration::levy::MarketSlice> =
      slices.into_iter().map(|s| s.inner).collect();
    let params = crate::calibration::bsm::BSMParams { v: sigma_init };
    Ok(Self {
      inner: crate::calibration::bsm::BSMCalibrator::from_slices(
        params,
        &inner_slices,
        s,
        r,
        None,
        None,
        q,
        ot,
      ),
    })
  }

  /// Run calibration. Returns `(sigma, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("BSM calibration failed: {e}")))?;
    Ok((
      res.v,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

#[pyclass(name = "HestonCalibrator", unsendable)]
pub struct PyHestonCalibrator {
  inner: crate::calibration::heston::HestonCalibrator,
}

#[pymethods]
impl PyHestonCalibrator {
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
      inner: crate::calibration::heston::HestonCalibrator::from_slices(
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

  /// Returns `(v0, kappa, theta, sigma, rho, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("Heston calibration failed: {e}")))?;
    let p = &res.params;
    Ok((
      p.v0,
      p.kappa,
      p.theta,
      p.sigma,
      p.rho,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

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

#[pyclass(name = "AsianPricer", unsendable)]
pub struct PyAsianPricer {
  inner: crate::pricing::asian::AsianPricer,
}

#[pymethods]
impl PyAsianPricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, q=None))]
  fn new(s: f64, v: f64, k: f64, r: f64, tau: f64, q: Option<f64>) -> Self {
    Self {
      inner: crate::pricing::asian::AsianPricer::new(s, v, k, r, q, Some(tau), None, None),
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "BarrierPricer", unsendable)]
pub struct PyBarrierPricer {
  inner: crate::pricing::barrier::BarrierPricer,
}

#[pymethods]
impl PyBarrierPricer {
  /// `barrier_type`: one of "up_in", "up_out", "down_in", "down_out".
  #[new]
  #[pyo3(signature = (s, k, h, r, q, sigma, t, barrier_type, option_type="call", rebate=0.0))]
  fn new(
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier_type: &str,
    option_type: &str,
    rebate: f64,
  ) -> PyResult<Self> {
    use crate::pricing::barrier::BarrierType;
    let bt = match barrier_type.to_ascii_lowercase().as_str() {
      "up_in" | "ui" | "upandin" => BarrierType::UpAndIn,
      "up_out" | "uo" | "upandout" => BarrierType::UpAndOut,
      "down_in" | "di" | "downandin" => BarrierType::DownAndIn,
      "down_out" | "do" | "downandout" => BarrierType::DownAndOut,
      o => {
        return Err(PyValueError::new_err(format!(
          "barrier_type must be one of up_in/up_out/down_in/down_out, got '{o}'"
        )));
      }
    };
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::barrier::BarrierPricer {
        s,
        k,
        h,
        r,
        q,
        sigma,
        t,
        rebate,
        barrier_type: bt,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "CashOrNothingPricer", unsendable)]
pub struct PyCashOrNothingPricer {
  inner: crate::pricing::digital::CashOrNothingPricer,
}

#[pymethods]
impl PyCashOrNothingPricer {
  /// `b` is the cost-of-carry (typically `r - q`).
  #[new]
  #[pyo3(signature = (s, k, cash, r, b, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k: f64,
    cash: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::CashOrNothingPricer {
        s,
        k,
        cash,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
  fn delta(&self) -> f64 {
    self.inner.delta()
  }
  fn gamma(&self) -> f64 {
    self.inner.gamma()
  }
  fn vega(&self) -> f64 {
    self.inner.vega()
  }
}

#[pyclass(name = "AssetOrNothingPricer", unsendable)]
pub struct PyAssetOrNothingPricer {
  inner: crate::pricing::digital::AssetOrNothingPricer,
}

#[pymethods]
impl PyAssetOrNothingPricer {
  #[new]
  #[pyo3(signature = (s, k, r, b, sigma, t, option_type="call"))]
  fn new(s: f64, k: f64, r: f64, b: f64, sigma: f64, t: f64, option_type: &str) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::AssetOrNothingPricer {
        s,
        k,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "FloatingLookbackPricer", unsendable)]
pub struct PyFloatingLookbackPricer {
  inner: crate::pricing::lookback::FloatingLookbackPricer,
}

#[pymethods]
impl PyFloatingLookbackPricer {
  #[new]
  #[pyo3(signature = (s, r, q, sigma, t, option_type="call", s_min=None, s_max=None))]
  fn new(
    s: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
    s_min: Option<f64>,
    s_max: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::lookback::FloatingLookbackPricer {
        s,
        s_min,
        s_max,
        r,
        q,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "BjerksundStensland2002Pricer", unsendable)]
pub struct PyBjerksundStensland2002Pricer {
  inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer,
}

#[pymethods]
impl PyBjerksundStensland2002Pricer {
  #[new]
  #[pyo3(signature = (s, v, k, r, tau, option_type="call", q=None))]
  fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::bjerksund_stensland::BjerksundStensland2002Pricer::new(
        s,
        v,
        k,
        r,
        q,
        Some(tau),
        None,
        None,
        ot,
      ),
    })
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
  }
  fn call_put(&self) -> (f64, f64) {
    self.inner.calculate_call_put()
  }
}

#[pyclass(name = "VarianceSwapPricer", unsendable)]
pub struct PyVarianceSwapPricer {
  inner: crate::pricing::variance_swap::VarianceSwapPricer,
}

#[pymethods]
impl PyVarianceSwapPricer {
  #[new]
  fn new(s: f64, r: f64, q: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::variance_swap::VarianceSwapPricer { s, r, q, t },
    }
  }

  fn forward(&self) -> f64 {
    self.inner.forward()
  }
  /// BSM fair-strike (`σ²`).
  fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    self.inner.fair_strike_bsm(sigma)
  }
  /// Heston closed-form fair-variance strike (Brockhaus-Long 2000).
  fn fair_strike_heston(&self, v0: f64, kappa: f64, theta: f64) -> f64 {
    self.inner.fair_strike_heston(v0, kappa, theta)
  }
  /// Demeterfi-Derman-Kamal-Zou static-replication fair strike.
  fn fair_strike_replication(&self, strikes: Vec<f64>, otm_prices: Vec<f64>) -> f64 {
    self.inner.fair_strike_replication(&strikes, &otm_prices)
  }
}

#[pyclass(name = "DoubleHestonFourier", unsendable)]
pub struct PyDoubleHestonFourier {
  pub inner: crate::pricing::fourier::DoubleHestonFourier,
}

#[pymethods]
impl PyDoubleHestonFourier {
  #[new]
  fn new(
    v1_0: f64,
    kappa1: f64,
    theta1: f64,
    sigma1: f64,
    rho1: f64,
    v2_0: f64,
    kappa2: f64,
    theta2: f64,
    sigma2: f64,
    rho2: f64,
    r: f64,
    q: f64,
  ) -> Self {
    Self {
      inner: crate::pricing::fourier::DoubleHestonFourier {
        v1_0,
        kappa1,
        theta1,
        sigma1,
        rho1,
        v2_0,
        kappa2,
        theta2,
        sigma2,
        rho2,
        r,
        q,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "HKDEFourier", unsendable)]
pub struct PyHKDEFourier {
  pub inner: crate::pricing::fourier::HKDEFourier,
}

#[pymethods]
impl PyHKDEFourier {
  #[new]
  fn new(
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    r: f64,
    q: f64,
    lam: f64,
    p_up: f64,
    eta1: f64,
    eta2: f64,
  ) -> Self {
    Self {
      inner: crate::pricing::fourier::HKDEFourier {
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        r,
        q,
        lam,
        p_up,
        eta1,
        eta2,
      },
    }
  }
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "SabrCalibrator", unsendable)]
pub struct PySabrCalibrator {
  inner: crate::calibration::sabr::SabrCalibrator,
}

#[pymethods]
impl PySabrCalibrator {
  /// Single-maturity SABR calibration (β fixed at 1.0 by default).
  #[new]
  #[pyo3(signature = (strikes, prices, s, r, tau, option_type="call", q=None))]
  fn new(
    strikes: Vec<f64>,
    prices: Vec<f64>,
    s: f64,
    r: f64,
    tau: f64,
    option_type: &str,
    q: Option<f64>,
  ) -> PyResult<Self> {
    use nalgebra::DVector;
    if strikes.len() != prices.len() {
      return Err(PyValueError::new_err(
        "strikes and prices must have equal length",
      ));
    }
    let n = strikes.len();
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::calibration::sabr::SabrCalibrator::new(
        None,
        DVector::from_vec(prices),
        DVector::from_vec(vec![s; n]),
        DVector::from_vec(strikes),
        r,
        q,
        tau,
        ot,
        false,
      ),
    })
  }

  /// Returns `(alpha, beta, nu, rho, converged, loss_rmse)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, bool, f64)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR calibration failed: {e}")))?;
    Ok((
      res.alpha,
      res.beta,
      res.nu,
      res.rho,
      res.converged,
      res.loss.get(crate::types::LossMetric::Rmse),
    ))
  }
}

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

#[pyclass(name = "CompoundPricer", unsendable)]
pub struct PyCompoundPricer {
  inner: crate::pricing::compound::CompoundPricer,
}

#[pymethods]
impl PyCompoundPricer {
  /// `compound_type`: one of "call_on_call" / "call_on_put" / "put_on_call" / "put_on_put".
  #[new]
  fn new(
    s: f64,
    k1: f64,
    k2: f64,
    t1: f64,
    t2: f64,
    r: f64,
    q: f64,
    sigma: f64,
    compound_type: &str,
  ) -> PyResult<Self> {
    use crate::pricing::compound::CompoundType;
    let ct = match compound_type.to_ascii_lowercase().as_str() {
      "call_on_call" | "coc" => CompoundType::CallOnCall,
      "call_on_put" | "cop" => CompoundType::CallOnPut,
      "put_on_call" | "poc" => CompoundType::PutOnCall,
      "put_on_put" | "pop" => CompoundType::PutOnPut,
      o => {
        return Err(PyValueError::new_err(format!(
          "compound_type must be one of call_on_call/call_on_put/put_on_call/put_on_put, got '{o}'"
        )));
      }
    };
    Ok(Self {
      inner: crate::pricing::compound::CompoundPricer {
        s,
        k1,
        k2,
        t1,
        t2,
        r,
        q,
        sigma,
        compound_type: ct,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "SimpleChooserPricer", unsendable)]
pub struct PySimpleChooserPricer {
  inner: crate::pricing::chooser::SimpleChooserPricer,
}

#[pymethods]
impl PySimpleChooserPricer {
  #[new]
  fn new(s: f64, k: f64, r: f64, q: f64, sigma: f64, t1: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::chooser::SimpleChooserPricer {
        s,
        k,
        r,
        q,
        sigma,
        t1,
        t,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "CliquetPricer", unsendable)]
pub struct PyCliquetPricer {
  inner: crate::pricing::cliquet::CliquetPricer,
}

#[pymethods]
impl PyCliquetPricer {
  #[new]
  #[pyo3(signature = (s, notional, m, t, r, q, sigma, local_floor=None, local_cap=None))]
  fn new(
    s: f64,
    notional: f64,
    m: usize,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
    local_floor: Option<f64>,
    local_cap: Option<f64>,
  ) -> Self {
    Self {
      inner: crate::pricing::cliquet::CliquetPricer {
        s,
        notional,
        m,
        t,
        r,
        q,
        sigma,
        local_floor,
        local_cap,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "GapPricer", unsendable)]
pub struct PyGapPricer {
  inner: crate::pricing::digital::GapPricer,
}

#[pymethods]
impl PyGapPricer {
  #[new]
  #[pyo3(signature = (s, k1, k2, r, b, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k1: f64,
    k2: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::digital::GapPricer {
        s,
        k1,
        k2,
        r,
        b,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "SuperSharePricer", unsendable)]
pub struct PySuperSharePricer {
  inner: crate::pricing::digital::SuperSharePricer,
}

#[pymethods]
impl PySuperSharePricer {
  #[new]
  fn new(s: f64, x_low: f64, x_high: f64, r: f64, b: f64, sigma: f64, t: f64) -> Self {
    Self {
      inner: crate::pricing::digital::SuperSharePricer {
        s,
        x_low,
        x_high,
        r,
        b,
        sigma,
        t,
      },
    }
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "FixedLookbackPricer", unsendable)]
pub struct PyFixedLookbackPricer {
  inner: crate::pricing::lookback::FixedLookbackPricer,
}

#[pymethods]
impl PyFixedLookbackPricer {
  #[new]
  #[pyo3(signature = (s, k, r, q, sigma, t, option_type="call", s_min=None, s_max=None))]
  fn new(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
    s_min: Option<f64>,
    s_max: Option<f64>,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::lookback::FixedLookbackPricer {
        s,
        k,
        s_min,
        s_max,
        r,
        q,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "DoubleBarrierPricer", unsendable)]
pub struct PyDoubleBarrierPricer {
  inner: crate::pricing::barrier::DoubleBarrierPricer,
}

#[pymethods]
impl PyDoubleBarrierPricer {
  #[new]
  #[pyo3(signature = (s, k, h_upper, h_lower, r, q, sigma, t, option_type="call"))]
  fn new(
    s: f64,
    k: f64,
    h_upper: f64,
    h_lower: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
    let ot = parse_option_type(option_type)?;
    Ok(Self {
      inner: crate::pricing::barrier::DoubleBarrierPricer {
        s,
        k,
        h_upper,
        h_lower,
        r,
        q,
        sigma,
        t,
        option_type: ot,
      },
    })
  }

  fn price(&self) -> f64 {
    self.inner.price()
  }
}

#[pyclass(name = "MCBarrierPricer", unsendable)]
pub struct PyMCBarrierPricer {
  inner: crate::pricing::barrier::MCBarrierPricer,
}

#[pymethods]
impl PyMCBarrierPricer {
  #[new]
  #[pyo3(signature = (n_paths=10000, n_steps=252))]
  fn new(n_paths: usize, n_steps: usize) -> Self {
    Self {
      inner: crate::pricing::barrier::MCBarrierPricer { n_paths, n_steps },
    }
  }

  /// `barrier_type`: one of "up_in" / "up_out" / "down_in" / "down_out".
  #[pyo3(signature = (s, k, h, r, sigma, t, barrier_type, option_type="call"))]
  fn price(
    &self,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier_type: &str,
    option_type: &str,
  ) -> PyResult<f64> {
    use crate::pricing::barrier::BarrierType;
    let bt = match barrier_type.to_ascii_lowercase().as_str() {
      "up_in" | "ui" => BarrierType::UpAndIn,
      "up_out" | "uo" => BarrierType::UpAndOut,
      "down_in" | "di" => BarrierType::DownAndIn,
      "down_out" | "do" => BarrierType::DownAndOut,
      o => {
        return Err(PyValueError::new_err(format!(
          "barrier_type must be one of up_in/up_out/down_in/down_out, got '{o}'"
        )));
      }
    };
    let ot = parse_option_type(option_type)?;
    Ok(self.inner.price(s, k, h, r, sigma, t, bt, ot))
  }
}

#[pyclass(name = "KirkSpreadPricer", unsendable)]
pub struct PyKirkSpreadPricer {
  inner: crate::pricing::kirk::KirkSpreadPricer,
}

#[pymethods]
impl PyKirkSpreadPricer {
  #[new]
  fn new(f1: f64, f2: f64, x: f64, r: f64, v1: f64, v2: f64, corr: f64, tau: f64) -> Self {
    Self {
      inner: crate::pricing::kirk::KirkSpreadPricer::new(
        f1,
        f2,
        x,
        r,
        v1,
        v2,
        corr,
        Some(tau),
        None,
        None,
      ),
    }
  }

  fn price(&self) -> f64 {
    self.inner.calculate_price()
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
  ) -> Self {
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
    Self {
      inner: RBergomiCalibrator::new(s0, r, params, inner_slices, cfg, false),
    }
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
    use crate::risk::var::PnlOrLoss;
    use crate::risk::var::VarMethod;
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
    let value = crate::risk::var::value_at_risk(samples.as_array(), confidence, pol, m);
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
    use crate::risk::var::PnlOrLoss;
    use crate::risk::var::VarMethod;
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
    let value =
      crate::risk::expected_shortfall::expected_shortfall(samples.as_array(), confidence, pol, m);
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
  fn new<'py>(equity: numpy::PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::risk::drawdown::DrawdownStats::from_equity(equity.as_array()),
    }
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

#[pyclass(name = "AlmgrenChrissPlan", unsendable)]
pub struct PyAlmgrenChrissPlan {
  inner: crate::microstructure::almgren_chriss::AlmgrenChrissPlan<f64>,
}

#[pymethods]
impl PyAlmgrenChrissPlan {
  /// Compute the Almgren-Chriss optimal-execution schedule.
  /// `direction`: "sell" (default) or "buy".
  #[new]
  #[pyo3(signature = (
    total_shares, horizon, n_intervals, volatility, gamma, eta, lambda_,
    epsilon=0.0, direction="sell"
  ))]
  fn new(
    total_shares: f64,
    horizon: f64,
    n_intervals: usize,
    volatility: f64,
    gamma: f64,
    eta: f64,
    lambda_: f64,
    epsilon: f64,
    direction: &str,
  ) -> PyResult<Self> {
    use crate::microstructure::almgren_chriss::AlmgrenChrissParams;
    use crate::microstructure::almgren_chriss::ExecutionDirection;
    use crate::microstructure::almgren_chriss::optimal_execution;
    let dir = match direction.to_ascii_lowercase().as_str() {
      "sell" => ExecutionDirection::Sell,
      "buy" => ExecutionDirection::Buy,
      o => {
        return Err(PyValueError::new_err(format!(
          "direction must be 'sell' or 'buy', got '{o}'"
        )));
      }
    };
    let params = AlmgrenChrissParams {
      total_shares,
      direction: dir,
      horizon,
      n_intervals,
      volatility,
      gamma,
      eta,
      epsilon,
      lambda: lambda_,
    };
    Ok(Self {
      inner: optimal_execution(&params),
    })
  }

  fn inventory<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.inventory.clone().into_pyarray(py)
  }
  fn trades<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trades.clone().into_pyarray(py)
  }
  fn rates<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.rates.clone().into_pyarray(py)
  }
  #[getter]
  fn kappa(&self) -> f64 {
    self.inner.kappa
  }
  #[getter]
  fn expected_cost(&self) -> f64 {
    self.inner.expected_cost
  }
  #[getter]
  fn variance(&self) -> f64 {
    self.inner.variance
  }
  fn risk_adjusted_cost(&self, lambda: f64) -> f64 {
    self.inner.risk_adjusted_cost(lambda)
  }
}

#[pyclass(name = "KyleEquilibrium", unsendable)]
pub struct PyKyleEquilibrium {
  inner: crate::microstructure::kyle::KyleEquilibrium<f64>,
}

#[pymethods]
impl PyKyleEquilibrium {
  /// Single-period Kyle (1985) equilibrium.
  #[new]
  fn new(prior_variance: f64, noise_variance: f64) -> Self {
    Self {
      inner: crate::microstructure::kyle::single_period_kyle(prior_variance, noise_variance),
    }
  }

  #[getter]
  fn beta(&self) -> f64 {
    self.inner.beta
  }
  /// Kyle's lambda (price-impact coefficient).
  #[getter]
  fn lambda(&self) -> f64 {
    self.inner.lambda
  }
  #[getter]
  fn posterior_variance(&self) -> f64 {
    self.inner.posterior_variance
  }
  #[getter]
  fn expected_profit(&self) -> f64 {
    self.inner.expected_profit
  }
}

#[pyfunction]
#[pyo3(signature = (prior_variance, noise_variance_per_round, n_periods))]
pub fn multi_period_kyle(
  prior_variance: f64,
  noise_variance_per_round: f64,
  n_periods: usize,
) -> Vec<(f64, f64, f64, f64)> {
  let v = crate::microstructure::kyle::multi_period_kyle(
    prior_variance,
    noise_variance_per_round,
    n_periods,
  );
  v.into_iter()
    .map(|e| (e.beta, e.lambda, e.posterior_variance, e.expected_profit))
    .collect()
}

#[pyfunction]
pub fn roll_spread<'py>(prices: numpy::PyReadonlyArray1<'py, f64>) -> f64 {
  crate::microstructure::spread::roll_spread(prices.as_array())
}

#[pyfunction]
pub fn effective_spread<'py>(
  trade_price: numpy::PyReadonlyArray1<'py, f64>,
  mid: numpy::PyReadonlyArray1<'py, f64>,
) -> f64 {
  crate::microstructure::spread::effective_spread(trade_price.as_array(), mid.as_array())
}

#[pyfunction]
pub fn corwin_schultz_spread<'py>(
  high: numpy::PyReadonlyArray1<'py, f64>,
  low: numpy::PyReadonlyArray1<'py, f64>,
) -> f64 {
  crate::microstructure::spread::corwin_schultz_spread(high.as_array(), low.as_array())
}

#[pyfunction]
#[pyo3(signature = (signed_volumes, kernel="powerlaw", g0=1.0, beta=0.5))]
pub fn propagator_price_impact<'py>(
  signed_volumes: numpy::PyReadonlyArray1<'py, f64>,
  kernel: &str,
  g0: f64,
  beta: f64,
) -> PyResult<f64> {
  use crate::microstructure::impact::ImpactKernel;
  let k = match kernel.to_ascii_lowercase().as_str() {
    "powerlaw" | "power_law" => ImpactKernel::<f64>::PowerLaw,
    "exponential" | "exp" => ImpactKernel::<f64>::Exponential,
    o => {
      return Err(PyValueError::new_err(format!(
        "kernel must be 'powerlaw' or 'exponential', got '{o}'"
      )));
    }
  };
  Ok(crate::microstructure::impact::propagator_price_impact(
    signed_volumes.as_array(),
    k,
    g0,
    beta,
  ))
}

/// Limit order book — supports add / cancel / market-order execution.
#[pyclass(name = "OrderBook", unsendable)]
pub struct PyOrderBook {
  inner: crate::order_book::OrderBook,
}

fn parse_side(s: &str) -> PyResult<crate::order_book::Side> {
  match s.to_ascii_lowercase().as_str() {
    "buy" | "b" | "bid" => Ok(crate::order_book::Side::Buy),
    "sell" | "s" | "ask" | "offer" => Ok(crate::order_book::Side::Sell),
    o => Err(PyValueError::new_err(format!(
      "side must be 'buy' or 'sell', got '{o}'"
    ))),
  }
}

#[pymethods]
impl PyOrderBook {
  #[new]
  fn new() -> Self {
    Self {
      inner: crate::order_book::OrderBook::new(),
    }
  }

  /// Add a limit order. Returns `(order_id, [(price, size, taker_id, maker_id)] of immediate fills)`.
  fn add_order(
    &mut self,
    side: &str,
    price: f64,
    size: f64,
  ) -> PyResult<(u64, Vec<(f64, f64, u64, u64)>)> {
    let s = parse_side(side)?;
    let (id, trades) = self.inner.add_order(s, price, size);
    Ok((
      id,
      trades
        .into_iter()
        .map(|t| (t.price, t.size, t.taker_id, t.maker_id))
        .collect(),
    ))
  }

  /// Execute a market order. Returns `(taker_id, [trades], remaining_unfilled_size)`.
  fn execute_order(
    &mut self,
    side: &str,
    size: f64,
  ) -> PyResult<(u64, Vec<(f64, f64, u64, u64)>, f64)> {
    let s = parse_side(side)?;
    let (id, trades, remaining) = self.inner.execute_order(s, size);
    Ok((
      id,
      trades
        .into_iter()
        .map(|t| (t.price, t.size, t.taker_id, t.maker_id))
        .collect(),
      remaining,
    ))
  }

  fn cancel_order(&mut self, id: u64) -> bool {
    self.inner.cancel_order(id)
  }

  fn best_bid(&self) -> Option<(f64, f64)> {
    self.inner.best_bid()
  }
  fn best_ask(&self) -> Option<(f64, f64)> {
    self.inner.best_ask()
  }
  fn mid(&self) -> Option<f64> {
    self.inner.mid()
  }
  fn spread(&self) -> Option<f64> {
    self.inner.spread()
  }
  /// Returns `(bids, asks)` where each side is `[(price, total_size)]`.
  fn depth(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    self.inner.depth()
  }
}

#[pyclass(name = "DiscountCurve", unsendable)]
pub struct PyDiscountCurve {
  inner: crate::curves::discount_curve::DiscountCurve<f64>,
}

#[pymethods]
impl PyDiscountCurve {
  /// Build from `(maturities, zero_rates)` arrays under continuous compounding.
  /// `interp`: "linear" / "log_df" / "cubic" / "monotone_convex".
  #[staticmethod]
  #[pyo3(signature = (maturities, zero_rates, interp="linear"))]
  fn from_zero_rates<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    zero_rates: numpy::PyReadonlyArray1<'py, f64>,
    interp: &str,
  ) -> PyResult<Self> {
    use crate::curves::types::InterpolationMethod;
    let im = match interp.to_ascii_lowercase().as_str() {
      "linear" | "linear_zr" => InterpolationMethod::LinearOnZeroRates,
      "log_df" | "loglinear_df" => InterpolationMethod::LogLinearOnDiscountFactors,
      "cubic" | "cubic_zr" => InterpolationMethod::CubicSplineOnZeroRates,
      "monotone_convex" | "mc" => InterpolationMethod::MonotoneConvex,
      o => {
        return Err(PyValueError::new_err(format!(
          "interp must be linear/log_df/cubic/monotone_convex, got '{o}'"
        )));
      }
    };
    let mat = maturities.as_array().to_owned();
    let zr = zero_rates.as_array().to_owned();
    Ok(Self {
      inner: crate::curves::discount_curve::DiscountCurve::from_zero_rates(&mat, &zr, im),
    })
  }

  fn discount_factor(&self, t: f64) -> f64 {
    self.inner.discount_factor(t)
  }
  fn zero_rate(&self, t: f64) -> f64 {
    self.inner.zero_rate(t)
  }
  fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
    self.inner.forward_rate(t1, t2)
  }
  fn par_rate(&self, maturity: f64, frequency: u32) -> f64 {
    self.inner.par_rate(maturity, frequency)
  }

  /// Vectorised zero rates on a maturity array.
  fn zero_rates<'py>(
    &self,
    py: Python<'py>,
    maturities: numpy::PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let mat = maturities.as_array().to_owned();
    self.inner.zero_rates(&mat).into_pyarray(py)
  }
}

#[pyclass(name = "NelsonSiegel", unsendable)]
pub struct PyNelsonSiegel {
  inner: crate::curves::nelson_siegel::NelsonSiegel<f64>,
}

#[pymethods]
impl PyNelsonSiegel {
  #[new]
  fn new(beta0: f64, beta1: f64, beta2: f64, lambda: f64) -> Self {
    Self {
      inner: crate::curves::nelson_siegel::NelsonSiegel::new(beta0, beta1, beta2, lambda),
    }
  }

  /// Fit Nelson-Siegel parameters to market zero rates (requires openblas feature).
  #[cfg(feature = "openblas")]
  #[staticmethod]
  fn fit_curve<'py>(
    maturities: numpy::PyReadonlyArray1<'py, f64>,
    market_rates: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    let mat = maturities.as_array().to_owned();
    let mr = market_rates.as_array().to_owned();
    Self {
      inner: <crate::curves::nelson_siegel::NelsonSiegel<f64>>::fit(&mat, &mr),
    }
  }

  fn zero_rate(&self, tau: f64) -> f64 {
    self.inner.zero_rate(tau)
  }
  fn forward_rate(&self, tau: f64) -> f64 {
    self.inner.forward_rate(tau)
  }
  fn discount_factor(&self, tau: f64) -> f64 {
    self.inner.discount_factor(tau)
  }
}

#[pyclass(name = "SabrCapletCalibrator", unsendable)]
pub struct PySabrCapletCalibrator {
  inner: crate::calibration::sabr_caplet::SabrCapletCalibrator,
}

#[pymethods]
impl PySabrCapletCalibrator {
  /// SABR caplet smile calibrator — fits `(α, ν, ρ)` for a single expiry,
  /// β held fixed.
  #[new]
  fn new(forward: f64, expiry: f64, beta: f64, strikes: Vec<f64>, market_vols: Vec<f64>) -> Self {
    Self {
      inner: crate::calibration::sabr_caplet::SabrCapletCalibrator::new(
        forward,
        expiry,
        beta,
        strikes,
        market_vols,
      ),
    }
  }

  /// Returns `(alpha, beta, nu, rho, rmse, converged)`.
  fn calibrate(&self) -> PyResult<(f64, f64, f64, f64, f64, bool)> {
    use crate::traits::Calibrator;
    let res = self
      .inner
      .calibrate(None)
      .map_err(|e| PyValueError::new_err(format!("SABR caplet calibration failed: {e}")))?;
    Ok((
      res.alpha,
      res.beta,
      res.nu,
      res.rho,
      res.rmse,
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

#[pyclass(name = "ZeroCouponInflationCurve", unsendable)]
pub struct PyZeroCouponInflationCurve {
  inner: crate::inflation::curve::ZeroCouponInflationCurve<f64>,
}

#[pymethods]
impl PyZeroCouponInflationCurve {
  /// Build a zero-coupon inflation curve from `(pillars, breakevens)`.
  #[new]
  fn new<'py>(
    pillars: numpy::PyReadonlyArray1<'py, f64>,
    breakevens: numpy::PyReadonlyArray1<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::inflation::curve::ZeroCouponInflationCurve::new(
        pillars.as_array().to_owned(),
        breakevens.as_array().to_owned(),
      ),
    }
  }

  /// Forward CPI index ratio $I(0, T)/I(0, 0) = (1 + b(T))^T$.
  fn forward_index_ratio(&self, t: f64) -> f64 {
    use crate::inflation::curve::InflationCurve;
    self.inner.forward_index_ratio(t)
  }
}

#[pyfunction]
pub fn ledoit_wolf_shrinkage<'py>(
  py: Python<'py>,
  returns: numpy::PyReadonlyArray2<'py, f64>,
) -> (pyo3::Bound<'py, numpy::PyArray2<f64>>, f64) {
  use numpy::IntoPyArray;
  let res = crate::factors::shrinkage::ledoit_wolf_shrinkage(returns.as_array());
  (res.covariance.into_pyarray(py), res.alpha)
}

#[pyfunction]
pub fn sample_covariance<'py>(
  py: Python<'py>,
  returns: numpy::PyReadonlyArray2<'py, f64>,
) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
  use numpy::IntoPyArray;
  let res = crate::factors::shrinkage::sample_covariance(returns.as_array());
  res.into_pyarray(py)
}

#[cfg(feature = "openblas")]
#[pyclass(name = "PCA", unsendable)]
pub struct PyPCA {
  inner: crate::factors::pca::PcaResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyPCA {
  /// PCA on a `(T, N)` returns matrix; `k=0` keeps all factors.
  #[new]
  #[pyo3(signature = (returns, k=0))]
  fn new<'py>(returns: numpy::PyReadonlyArray2<'py, f64>, k: usize) -> Self {
    Self {
      inner: crate::factors::pca::pca_decompose(returns.as_array(), k),
    }
  }

  fn singular_values<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.singular_values.clone().into_pyarray(py)
  }
  fn eigenvalues<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.eigenvalues.clone().into_pyarray(py)
  }
  fn explained_variance_ratio<'py>(
    &self,
    py: Python<'py>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.explained_variance_ratio.clone().into_pyarray(py)
  }
  fn loadings<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.loadings.clone().into_pyarray(py)
  }
  fn scores<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.scores.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "FamaMacBeth", unsendable)]
pub struct PyFamaMacBeth {
  inner: crate::factors::fama_macbeth::FamaMacBethResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyFamaMacBeth {
  /// Two-pass Fama-MacBeth cross-sectional regression on `(T, N)` returns
  /// and `(T, K)` factors.
  #[new]
  fn new<'py>(
    returns: numpy::PyReadonlyArray2<'py, f64>,
    factors: numpy::PyReadonlyArray2<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::factors::fama_macbeth::fama_macbeth(returns.as_array(), factors.as_array()),
    }
  }

  fn gamma<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.gamma.clone().into_pyarray(py)
  }
  fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.std_errors.clone().into_pyarray(py)
  }
  fn t_statistics<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.t_statistics.clone().into_pyarray(py)
  }
  fn betas<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
    use numpy::IntoPyArray;
    self.inner.betas.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "PairsStrategy", unsendable)]
pub struct PyPairsStrategy {
  inner: crate::factors::pairs::PairsStrategy,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyPairsStrategy {
  /// Build a pairs-trading strategy from cointegration regression of `y` on `x`.
  /// `entry_z`: enter when `|z| ≥ entry_z`. `exit_z`: close when `|z| ≤ exit_z`.
  #[new]
  #[pyo3(signature = (y, x, entry_z=2.0, exit_z=0.5))]
  fn new<'py>(
    y: numpy::PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray1<'py, f64>,
    entry_z: f64,
    exit_z: f64,
  ) -> Self {
    Self {
      inner: crate::factors::pairs::pairs_signals(y.as_array(), x.as_array(), entry_z, exit_z),
    }
  }

  #[getter]
  fn alpha(&self) -> f64 {
    self.inner.alpha
  }
  #[getter]
  fn beta(&self) -> f64 {
    self.inner.beta
  }
  #[getter]
  fn spread_mean(&self) -> f64 {
    self.inner.spread_mean
  }
  #[getter]
  fn spread_std(&self) -> f64 {
    self.inner.spread_std
  }

  fn spread<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.spread.clone().into_pyarray(py)
  }
  fn z_score<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.z_score.clone().into_pyarray(py)
  }
  /// Returns signals as an integer array: -1 = ShortSpread, 0 = Flat, +1 = LongSpread.
  fn signals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<i64>> {
    use numpy::IntoPyArray;
    let arr: ndarray::Array1<i64> = self
      .inner
      .signals
      .iter()
      .map(|s| match s {
        crate::factors::pairs::PairsSignal::LongSpread => 1i64,
        crate::factors::pairs::PairsSignal::ShortSpread => -1i64,
        crate::factors::pairs::PairsSignal::Flat => 0i64,
      })
      .collect();
    arr.into_pyarray(py)
  }
}

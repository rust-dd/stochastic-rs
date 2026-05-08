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
    let mut builder = crate::pricing::merton_jump::Merton1976Pricer::builder(
      s, v, k, r, lambda_, gamma, m,
    )
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

#[pyclass(name = "MarketSlice", unsendable)]
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
  fn new(
    s: f64,
    k: f64,
    r: f64,
    b: f64,
    sigma: f64,
    t: f64,
    option_type: &str,
  ) -> PyResult<Self> {
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
  fn calibrate(
    &self,
  ) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, f64)> {
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

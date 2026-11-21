use num_complex::Complex64;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::parse_option_type;
use crate::pricing::fourier::Cumulants;
use crate::pricing::fourier::FourierModelExt;
use crate::pricing::fourier::HkdeParams;
use crate::pricing::fourier::SvjParams;
use crate::pricing::heston::HestonParams;
use crate::traits::ModelPricer;

#[pyclass(name = "BSMFourier", unsendable)]
pub struct PyBsmFourier {
  pub inner: crate::pricing::fourier::BsmFourier,
}

#[pymethods]
impl PyBsmFourier {
  #[new]
  fn new(sigma: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::BsmFourier { sigma },
    })
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
  fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::HestonFourier::new(HestonParams {
        v0,
        kappa,
        theta,
        sigma,
        rho,
      }),
    })
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
  fn new(sigma: f64, theta: f64, nu: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::VarianceGammaFourier { sigma, theta, nu },
    })
  }

  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "CGMYFourier", unsendable)]
pub struct PyCgmyFourier {
  pub inner: crate::pricing::fourier::CgmyFourier,
}

#[pymethods]
impl PyCgmyFourier {
  #[new]
  fn new(c: f64, g: f64, m: f64, y: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::CgmyFourier { c, g, m, y },
    })
  }

  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "MertonJDFourier", unsendable)]
pub struct PyMertonJdFourier {
  pub inner: crate::pricing::fourier::MertonJdFourier,
}

#[pymethods]
impl PyMertonJdFourier {
  #[new]
  fn new(sigma: f64, lambda: f64, mu_j: f64, sigma_j: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::MertonJdFourier {
        sigma,
        lambda,
        mu_j,
        sigma_j,
      },
    })
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
  fn new(sigma: f64, lambda: f64, p_up: f64, eta1: f64, eta2: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::KouFourier {
        sigma,
        lambda,
        p_up,
        eta1,
        eta2,
      },
    })
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::BatesFourier::new(SvjParams {
        v0,
        kappa,
        theta,
        sigma: sigma_v,
        rho,
        lambda,
        mu_j,
        sigma_j,
      }),
    })
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
  fn new(n: usize, alpha: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::CarrMadanPricer::new(n, alpha),
    })
  }

  /// Heston call price; `nan` for a strike outside the FFT grid (widen it with a larger `n`, or
  /// test the strike with `strike_in_grid_heston`).
  fn price_heston_call(
    &self,
    model: &PyHestonFourier,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, q, tau)
  }

  /// Price a Bates call. See `price_heston_call` for the NaN convention.
  fn price_bates_call(
    &self,
    model: &PyBatesFourier,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, q, tau)
  }

  /// Price a Kou call. See `price_heston_call` for the NaN convention.
  fn price_kou_call(&self, model: &PyKouFourier, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.inner.price_call(&model.inner, s, k, r, q, tau)
  }

  /// Returns true iff strike `k` is within the FFT log-strike grid for the
  /// supplied Heston model and market state. Use to detect / widen the grid
  /// before calling `price_heston_call` rather than handling NaN downstream.
  fn strike_in_grid_heston(
    &self,
    model: &PyHestonFourier,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> bool {
    self.inner.strike_in_grid(&model.inner, s, k, r, q, tau)
  }
}

/// NIG (Normal Inverse Gaussian) Lévy model for Fourier pricing: tail heaviness `alpha > 0`,
/// skewness `|beta| < alpha` and scale `delta > 0`.
#[pyclass(name = "NigFourier", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyNigFourier {
  pub inner: crate::pricing::fourier::NigFourier,
}

#[pymethods]
impl PyNigFourier {
  #[new]
  fn new(alpha: f64, beta: f64, delta: f64) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      if alpha <= 0.0 {
        return Err(PyValueError::new_err("alpha must be > 0"));
      }
      if beta.abs() >= alpha {
        return Err(PyValueError::new_err("|beta| must be < alpha"));
      }
      if delta <= 0.0 {
        return Err(PyValueError::new_err("delta must be > 0"));
      }
      Ok(Self {
        inner: crate::pricing::fourier::NigFourier { alpha, beta, delta },
      })
    })?
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::DoubleHestonFourier::new(
        HestonParams {
          v0: v1_0,
          kappa: kappa1,
          theta: theta1,
          sigma: sigma1,
          rho: rho1,
        },
        HestonParams {
          v0: v2_0,
          kappa: kappa2,
          theta: theta2,
          sigma: sigma2,
          rho: rho2,
        },
      ),
    })
  }

  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

#[pyclass(name = "HKDEFourier", unsendable)]
pub struct PyHkdeFourier {
  pub inner: crate::pricing::fourier::HkdeFourier,
}

#[pymethods]
impl PyHkdeFourier {
  #[new]
  fn new(
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    lam: f64,
    p_up: f64,
    eta1: f64,
    eta2: f64,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: crate::pricing::fourier::HkdeFourier::new(HkdeParams {
        v0,
        kappa,
        theta,
        sigma: sigma_v,
        rho,
        lambda: lam,
        p_up,
        eta1,
        eta2,
      }),
    })
  }

  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    ModelPricer::price_call(&self.inner, s, k, r, q, tau)
  }
}

/// The Rust model inside one of this module's Fourier model classes, as the model
/// [`PyCosEngine`] prices.
struct DynFourierModel(Box<dyn FourierModelExt>);

impl FourierModelExt for DynFourierModel {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    self.0.chf(t, xi)
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    self.0.cumulants(t)
  }
}

fn fourier_model(model: &Bound<'_, PyAny>) -> PyResult<DynFourierModel> {
  macro_rules! model_of {
    ($($wrapper:ty),+ $(,)?) => {{
      $(
        if let Ok(wrapper) = model.extract::<PyRef<'_, $wrapper>>() {
          return Ok(DynFourierModel(Box::new(wrapper.inner.clone())));
        }
      )+
      [$(<$wrapper as pyo3::PyClass>::NAME),+]
    }};
  }
  let names = model_of!(
    PyBsmFourier,
    PyHestonFourier,
    PyDoubleHestonFourier,
    PyBatesFourier,
    PyHkdeFourier,
    PyVarianceGammaFourier,
    PyCgmyFourier,
    PyMertonJdFourier,
    PyKouFourier,
    PyNigFourier,
  );
  Err(PyTypeError::new_err(format!(
    "model must be one of the Fourier models ({}), got {}",
    names.join(", "),
    model.get_type().name()?
  )))
}

/// Fang-Oosterlee (2008) COS pricer; `n=None`, the default, adds terms until the chf has decayed.
/// `l` is their L: the range spans `l` cumulant standard deviations either side, 10 by default.
#[pyclass(name = "CosEngine", unsendable)]
pub struct PyCosEngine {
  inner: crate::pricing::fourier::CosEngine,
}

#[pymethods]
impl PyCosEngine {
  #[new]
  #[pyo3(signature = (n=None, l=10.0))]
  fn new(n: Option<usize>, l: f64) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| Self {
      inner: match n {
        Some(n) => crate::pricing::fourier::CosEngine::new(n, l),
        None => crate::pricing::fourier::CosEngine::adaptive(l),
      },
    })
  }

  /// European call or put: the payoff at `tau = 0`; `nan` past expiry and, before it, for a rate
  /// not finite or an `s` or `k` that is not finite and positive.
  #[pyo3(signature = (model, s, k, r, q, tau, option_type="call"))]
  fn price(
    &self,
    model: &Bound<'_, PyAny>,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: &str,
  ) -> PyResult<f64> {
    let model = fourier_model(model)?;
    let option_type = parse_option_type(option_type)?;
    Ok(self.inner.price(&model, s, k, r, q, tau, option_type))
  }

  /// [`price`](Self::price) at each strike of an array, on one pass over the characteristic
  /// function.
  #[pyo3(signature = (model, s, strikes, r, q, tau, option_type="call"))]
  fn price_strikes<'py>(
    &self,
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    s: f64,
    strikes: numpy::PyReadonlyArray1<'py, f64>,
    r: f64,
    q: f64,
    tau: f64,
    option_type: &str,
  ) -> PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::IntoPyArray;
    let model = fourier_model(model)?;
    let option_type = parse_option_type(option_type)?;
    Ok(
      self
        .inner
        .price_strikes(&model, s, strikes.as_array(), r, q, tau, option_type)
        .into_pyarray(py),
    )
  }
}

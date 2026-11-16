use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
      inner: crate::pricing::fourier::HestonFourier {
        v0,
        kappa,
        theta,
        sigma,
        rho,
      },
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
      inner: crate::pricing::fourier::BatesFourier {
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
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
      },
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
      inner: crate::pricing::fourier::HkdeFourier {
        v0,
        kappa,
        theta,
        sigma_v,
        rho,
        lam,
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

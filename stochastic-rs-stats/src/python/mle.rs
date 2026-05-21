use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "HestonMLE", unsendable)]
pub struct PyHestonMLE {
  inner: crate::heston_mle::HestonMleResult,
}

#[pymethods]
impl PyHestonMLE {
  /// NMLE (Wang et al., 2018) closed-form Heston estimator.
  #[staticmethod]
  fn nmle<'py>(s: PyReadonlyArray1<'py, f64>, v: PyReadonlyArray1<'py, f64>, r: f64) -> Self {
    Self {
      inner: crate::heston_mle::nmle_heston(s.as_array(), v.as_array(), r),
    }
  }

  /// PMLE (penalised) Heston estimator.
  #[staticmethod]
  fn pmle<'py>(s: PyReadonlyArray1<'py, f64>, v: PyReadonlyArray1<'py, f64>, r: f64) -> Self {
    Self {
      inner: crate::heston_mle::pmle_heston(s.as_array(), v.as_array(), r),
    }
  }

  #[getter]
  fn v0(&self) -> f64 {
    self.inner.v0
  }
  #[getter]
  fn kappa(&self) -> f64 {
    self.inner.kappa
  }
  #[getter]
  fn theta(&self) -> f64 {
    self.inner.theta
  }
  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }
  #[getter]
  fn rho(&self) -> f64 {
    self.inner.rho
  }
}

#[pyclass(name = "HestonNMLECEKF", unsendable)]
pub struct PyHestonNMLECEKF {
  inner: crate::heston_nml_cekf::HestonNMLECEKFResult,
}

#[pymethods]
impl PyHestonNMLECEKF {
  /// NMLE-CEKF (Wang et al. 2018) Heston estimator from spot path only.
  #[new]
  #[pyo3(signature = (s, r=0.0, delta=None, max_iters=12))]
  fn new<'py>(s: PyReadonlyArray1<'py, f64>, r: f64, delta: Option<f64>, max_iters: usize) -> Self {
    let default_cfg = crate::heston_nml_cekf::HestonNMLECEKFConfig::default();
    let cfg = crate::heston_nml_cekf::HestonNMLECEKFConfig {
      r,
      delta: delta.unwrap_or(default_cfg.delta),
      max_iters,
      ..default_cfg
    };
    let arr = s.as_array();
    Self {
      inner: crate::heston_nml_cekf::nmle_cekf_heston(arr, cfg),
    }
  }

  #[getter]
  fn v0(&self) -> f64 {
    self.inner.params.v0
  }
  #[getter]
  fn kappa(&self) -> f64 {
    self.inner.params.kappa
  }
  #[getter]
  fn theta(&self) -> f64 {
    self.inner.params.theta
  }
  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.params.sigma
  }
  #[getter]
  fn rho(&self) -> f64 {
    self.inner.params.rho
  }
  #[getter]
  fn iterations(&self) -> usize {
    self.inner.iterations
  }
  #[getter]
  fn converged(&self) -> bool {
    self.inner.converged
  }
}

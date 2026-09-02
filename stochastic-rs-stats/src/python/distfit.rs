use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

macro_rules! fit_class {
  ($py:ident, $name:literal, $inner:path, $fit:path, $doc:literal, [$(($getter:ident, $field:ident)),+]) => {
    #[pyclass(name = $name, unsendable)]
    pub struct $py {
      inner: $inner,
    }

    #[pymethods]
    impl $py {
      #[doc = $doc]
      #[new]
      fn new<'py>(data: PyReadonlyArray1<'py, f64>) -> Self {
        Self { inner: $fit(data.as_array()) }
      }

      /// Standard errors in the parameter order of the getters.
      fn std_errors<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
        use numpy::IntoPyArray;
        self.inner.std_errors.clone().into_pyarray(py)
      }

      fn covariance<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray2<f64>> {
        use numpy::IntoPyArray;
        self.inner.covariance.clone().into_pyarray(py)
      }
      $(
        #[getter]
        fn $getter(&self) -> f64 {
          self.inner.$field
        }
      )+
      #[getter]
      fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
      }

      #[getter]
      fn aic(&self) -> f64 {
        self.inner.aic
      }

      #[getter]
      fn bic(&self) -> f64 {
        self.inner.bic
      }

      #[getter]
      fn nobs(&self) -> usize {
        self.inner.nobs
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
  };
}

fit_class!(
  PyJohnsonSuFit,
  "JohnsonSuFit",
  crate::distfit::JohnsonSuFit,
  crate::distfit::johnson_su_fit,
  "Johnson SU maximum-likelihood fit; parameters `gamma`, `delta`, `xi`, `lambda_`.",
  [(gamma, gamma), (delta, delta), (xi, xi), (lambda_, lambda)]
);

fit_class!(
  PySkewTFit,
  "SkewTFit",
  crate::distfit::SkewTFit,
  crate::distfit::skew_t_fit,
  "Hansen skew-t maximum-likelihood fit with location and scale; parameters `mu`, `sigma`, `eta`, `lambda_`.",
  [(mu, mu), (sigma, sigma), (eta, eta), (lambda_, lambda)]
);

fit_class!(
  PyVarianceGammaFit,
  "VarianceGammaFit",
  crate::distfit::VarianceGammaFit,
  crate::distfit::variance_gamma_fit,
  "Variance-gamma maximum-likelihood fit; parameters `sigma`, `nu`, `theta`, `mu`.",
  [(sigma, sigma), (nu, nu), (theta, theta), (mu, mu)]
);

#[pyclass(name = "GpdPwm", unsendable)]
pub struct PyGpdPwm {
  inner: crate::evt::GpdPwm,
}

#[pymethods]
impl PyGpdPwm {
  /// Hosking-Wallis probability-weighted-moment GPD estimate on excesses.
  #[new]
  fn new<'py>(exceedances: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::evt::gpd_pwm(exceedances.as_array()),
    }
  }

  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }

  #[getter]
  fn xi(&self) -> f64 {
    self.inner.xi
  }

  #[getter]
  fn a0(&self) -> f64 {
    self.inner.a0
  }

  #[getter]
  fn a1(&self) -> f64 {
    self.inner.a1
  }

  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
}

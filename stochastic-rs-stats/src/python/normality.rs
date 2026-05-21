use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "JarqueBera", unsendable)]
pub struct PyJarqueBera {
  inner: crate::normality::jarque_bera::JarqueBeraResult,
}

#[pymethods]
impl PyJarqueBera {
  #[new]
  #[pyo3(signature = (sample, alpha=0.05))]
  fn new<'py>(sample: PyReadonlyArray1<'py, f64>, alpha: f64) -> Self {
    let cfg = crate::normality::jarque_bera::JarqueBeraConfig { alpha };
    let view = sample.as_array();
    let inner = crate::normality::jarque_bera::jarque_bera_test(view, cfg);
    Self { inner }
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn skewness(&self) -> f64 {
    self.inner.skewness
  }
  #[getter]
  fn excess_kurtosis(&self) -> f64 {
    self.inner.excess_kurtosis
  }
  #[getter]
  fn reject_normality(&self) -> bool {
    self.inner.reject_normality
  }
}

#[pyclass(name = "AndersonDarling", unsendable)]
pub struct PyAndersonDarling {
  inner: crate::normality::anderson_darling::AndersonDarlingResult,
}

#[pymethods]
impl PyAndersonDarling {
  #[new]
  #[pyo3(signature = (sample, alpha=0.05))]
  fn new<'py>(sample: PyReadonlyArray1<'py, f64>, alpha: f64) -> Self {
    let cfg = crate::normality::anderson_darling::AndersonDarlingConfig { alpha };
    let view = sample.as_array();
    let inner = crate::normality::anderson_darling::anderson_darling_normal_test(view, cfg);
    Self { inner }
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn adjusted_statistic(&self) -> f64 {
    self.inner.adjusted_statistic
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn reject_normality(&self) -> bool {
    self.inner.reject_normality
  }
}

#[pyclass(name = "ShapiroFrancia", unsendable)]
pub struct PyShapiroFrancia {
  inner: crate::normality::shapiro_francia::ShapiroFranciaResult,
}

#[pymethods]
impl PyShapiroFrancia {
  #[new]
  #[pyo3(signature = (sample, alpha=0.05, bootstrap_samples=512, bootstrap_seed=42))]
  fn new<'py>(
    sample: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    bootstrap_samples: usize,
    bootstrap_seed: u64,
  ) -> Self {
    let cfg = crate::normality::shapiro_francia::ShapiroFranciaConfig {
      alpha,
      bootstrap_samples,
      bootstrap_seed,
    };
    let view = sample.as_array();
    let inner = crate::normality::shapiro_francia::shapiro_francia_test(view, cfg);
    Self { inner }
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn reject_normality(&self) -> bool {
    self.inner.reject_normality
  }
}

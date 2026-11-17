use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pyclass(name = "PCA", unsendable)]
pub struct PyPca {
  inner: crate::factors::pca::PcaResult,
}

#[pymethods]
impl PyPca {
  /// PCA on a `(T, N)` returns matrix; `k=0` keeps all factors.
  #[new]
  #[pyo3(signature = (returns, k=0))]
  fn new<'py>(returns: numpy::PyReadonlyArray2<'py, f64>, k: usize) -> pyo3::PyResult<Self> {
    Ok(Self {
      inner: crate::factors::pca::try_pca_decompose(returns.as_array(), k)
        .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
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

#[pyclass(name = "FamaMacBeth", unsendable)]
pub struct PyFamaMacBeth {
  inner: crate::factors::fama_macbeth::FamaMacBethResult,
}

#[pymethods]
impl PyFamaMacBeth {
  /// Two-pass Fama-MacBeth cross-sectional regression on `(T, N)` returns
  /// and `(T, K)` factors.
  #[new]
  fn new<'py>(
    returns: numpy::PyReadonlyArray2<'py, f64>,
    factors: numpy::PyReadonlyArray2<'py, f64>,
  ) -> pyo3::PyResult<Self> {
    Ok(Self {
      inner: crate::factors::fama_macbeth::try_fama_macbeth(returns.as_array(), factors.as_array())
        .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
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

#[pyclass(name = "PairsStrategy", unsendable)]
pub struct PyPairsStrategy {
  inner: crate::factors::pairs::PairsStrategy,
}

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
  ) -> pyo3::PyResult<Self> {
    Ok(Self {
      inner: crate::factors::pairs::try_pairs_signals(y.as_array(), x.as_array(), entry_z, exit_z)
        .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
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

/// Tail mean of the losses `−returns` over their worst `alpha` share (`0.05`: the worst 5 %, not a
/// confidence level); `ValueError` for an empty or non-finite sample or `alpha` outside `(0, 0.5)`.
#[pyfunction]
#[pyo3(signature = (returns, alpha))]
pub fn empirical_cvar<'py>(
  returns: numpy::PyReadonlyArray1<'py, f64>,
  alpha: f64,
) -> PyResult<f64> {
  let mut returns_vec: Vec<f64> = returns.as_array().to_vec();
  stochastic_rs_distributions::python::value_error_on_panic(|| {
    crate::portfolio::optimizers::empirical_cvar(&mut returns_vec, alpha)
  })?
  .map_err(|e| PyValueError::new_err(e.to_string()))
}

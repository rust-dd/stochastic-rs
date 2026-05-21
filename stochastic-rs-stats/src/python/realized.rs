use ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyclass(name = "RealizedMoments", unsendable)]
pub struct PyRealizedMoments {
  rv: f64,
  rvol: f64,
  skew: f64,
  kurt: f64,
  rq: f64,
}

#[pymethods]
impl PyRealizedMoments {
  /// Compute realised variance, volatility, skewness, kurtosis and quarticity
  /// from a log-return series. `annualisation` is multiplied into the realised
  /// volatility result (e.g. 252.0 for daily-to-annual).
  #[new]
  #[pyo3(signature = (returns, annualisation=1.0))]
  fn new<'py>(returns: PyReadonlyArray1<'py, f64>, annualisation: f64) -> Self {
    let view: ArrayView1<f64> = returns.as_array();
    Self {
      rv: crate::realized::variance::realized_variance(view),
      rvol: crate::realized::variance::realized_volatility(view, annualisation),
      skew: crate::realized::variance::realized_skewness(view),
      kurt: crate::realized::variance::realized_kurtosis(view),
      rq: crate::realized::variance::realized_quarticity(view),
    }
  }

  #[getter]
  fn variance(&self) -> f64 {
    self.rv
  }
  #[getter]
  fn volatility(&self) -> f64 {
    self.rvol
  }
  #[getter]
  fn skewness(&self) -> f64 {
    self.skew
  }
  #[getter]
  fn kurtosis(&self) -> f64 {
    self.kurt
  }
  #[getter]
  fn quarticity(&self) -> f64 {
    self.rq
  }
}

#[pyclass(name = "BipowerVariation", unsendable)]
pub struct PyBipowerVariation {
  bv: f64,
  minrv: f64,
  medrv: f64,
  tpq: f64,
}

#[pymethods]
impl PyBipowerVariation {
  /// Compute jump-robust bipower variation, minRV, medRV and tripower quarticity
  /// from a log-return series.
  #[new]
  fn new<'py>(returns: PyReadonlyArray1<'py, f64>) -> Self {
    let view: ArrayView1<f64> = returns.as_array();
    Self {
      bv: crate::realized::bipower::bipower_variation(view),
      minrv: crate::realized::bipower::minrv(view),
      medrv: crate::realized::bipower::medrv(view),
      tpq: crate::realized::bipower::tripower_quarticity(view),
    }
  }

  #[getter]
  fn bipower(&self) -> f64 {
    self.bv
  }
  #[getter]
  fn minrv(&self) -> f64 {
    self.minrv
  }
  #[getter]
  fn medrv(&self) -> f64 {
    self.medrv
  }
  #[getter]
  fn tripower_quarticity(&self) -> f64 {
    self.tpq
  }
}

#[pyclass(name = "BNSJumpTest", unsendable)]
pub struct PyBNSJumpTest {
  inner: crate::realized::bipower::BnsJumpTest,
}

#[pymethods]
impl PyBNSJumpTest {
  /// Barndorff-Nielsen / Shephard jump test on a log-return series.
  #[new]
  #[pyo3(signature = (returns, alpha=0.05))]
  fn new<'py>(returns: PyReadonlyArray1<'py, f64>, alpha: f64) -> Self {
    Self {
      inner: crate::realized::bipower::bns_jump_test(returns.as_array(), alpha),
    }
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
  fn reject_no_jump(&self) -> bool {
    self.inner.reject_no_jump
  }
}

#[pyclass(name = "RealizedKernel", unsendable)]
pub struct PyRealizedKernel {
  rk: f64,
  bandwidth: usize,
}

#[pymethods]
impl PyRealizedKernel {
  /// Realised kernel (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).
  /// `kernel`: one of "parzen" (default), "bartlett", "tukey_hanning",
  /// "tukey_hanning2", "cubic", "quadratic_spectral". `bandwidth` (None →
  /// Parzen automatic via `parzen_default_bandwidth`).
  #[new]
  #[pyo3(signature = (returns, kernel="parzen", bandwidth=None))]
  fn new<'py>(
    returns: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    bandwidth: Option<usize>,
  ) -> PyResult<Self> {
    use crate::realized::kernel::KernelType;
    let kt = match kernel.to_ascii_lowercase().as_str() {
      "parzen" => KernelType::Parzen,
      "bartlett" => KernelType::Bartlett,
      "tukey_hanning" | "th" => KernelType::TukeyHanning,
      "tukey_hanning2" | "th2" => KernelType::TukeyHanning2,
      "cubic" => KernelType::Cubic,
      "quadratic_spectral" | "qs" => KernelType::QuadraticSpectral,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "kernel must be one of parzen/bartlett/tukey_hanning/tukey_hanning2/cubic/quadratic_spectral, got '{o}'"
        )));
      }
    };
    let n = returns.as_array().len();
    let h = bandwidth.unwrap_or_else(|| crate::realized::kernel::parzen_default_bandwidth(n, 1.0));
    Ok(Self {
      rk: crate::realized::kernel::realized_kernel(returns.as_array(), kt, h),
      bandwidth: h,
    })
  }

  #[getter]
  fn realised(&self) -> f64 {
    self.rk
  }
  #[getter]
  fn bandwidth(&self) -> usize {
    self.bandwidth
  }
}

#[pyclass(name = "TwoScaleRV", unsendable)]
pub struct PyTwoScaleRV {
  rv: f64,
}

#[pymethods]
impl PyTwoScaleRV {
  /// Two-scale realised variance (Zhang-Mykland-Aït-Sahalia 2005).
  /// `prices`: log-price series; `k`: subsample step.
  #[new]
  fn new<'py>(prices: PyReadonlyArray1<'py, f64>, k: usize) -> Self {
    Self {
      rv: crate::realized::two_scale::two_scale_rv(prices.as_array(), k),
    }
  }

  #[getter]
  fn variance(&self) -> f64 {
    self.rv
  }
}

#[pyclass(name = "PreAveragedVariance", unsendable)]
pub struct PyPreAveragedVariance {
  value: f64,
}

#[pymethods]
impl PyPreAveragedVariance {
  /// Pre-averaging realised variance (Jacod et al. 2009) with explicit `theta`.
  #[new]
  fn new<'py>(returns: PyReadonlyArray1<'py, f64>, theta: f64) -> Self {
    Self {
      value: crate::realized::pre_averaging::pre_averaged_variance(returns.as_array(), theta),
    }
  }

  #[getter]
  fn variance(&self) -> f64 {
    self.value
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "HarRv", unsendable)]
pub struct PyHarRv {
  inner: crate::realized::har::HarRv,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyHarRv {
  /// Fit HAR-RV (Corsi 2009) on a daily realised-variance history.
  #[new]
  fn new<'py>(daily_rv: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::realized::har::HarRv::fit(daily_rv.as_array()),
    }
  }

  #[getter]
  fn intercept(&self) -> f64 {
    self.inner.fit.intercept
  }
  #[getter]
  fn beta_d(&self) -> f64 {
    self.inner.fit.beta_d
  }
  #[getter]
  fn beta_w(&self) -> f64 {
    self.inner.fit.beta_w
  }
  #[getter]
  fn beta_m(&self) -> f64 {
    self.inner.fit.beta_m
  }
  #[getter]
  fn r_squared(&self) -> f64 {
    self.inner.fit.r_squared
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.fit.nobs
  }

  /// One-step-ahead point forecast given recent daily-RV history.
  fn forecast<'py>(&self, recent_daily_rv: PyReadonlyArray1<'py, f64>) -> f64 {
    self.inner.forecast(recent_daily_rv.as_array())
  }
}

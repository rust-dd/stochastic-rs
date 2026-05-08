//! PyO3 wrappers for `stochastic-rs-stats`.
//!
//! Hypothesis tests (Jarque-Bera, Anderson-Darling, Shapiro-Francia, ADF, KPSS),
//! Hurst-exponent estimators (Fukasawa, fOU v1/v2), Heston MLE, and the realised
//! variance / bipower-variation / jump tests, all exposed as `#[pyclass]` types
//! that take a numpy array and return a result object with the test statistic
//! and either a p-value or boolean rejection flag.

#![cfg(feature = "python")]
#![allow(clippy::too_many_arguments)]

use ndarray::ArrayView1;
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

/// ADF/KPSS bindings require the openblas feature (the underlying
/// `stationarity` module pulls in `ndarray-linalg`).
#[cfg(feature = "openblas")]
#[pyclass(name = "ADFTest", unsendable)]
pub struct PyADFTest {
  inner: crate::stationarity::adf::ADFResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyADFTest {
  /// `deterministic`: "n" (none), "c" (constant), or "ct" (constant + trend).
  /// `lag_selection`: "aic" (default), "bic", or a non-negative integer literal
  /// for a fixed lag order.
  #[new]
  #[pyo3(signature = (y, deterministic="c", lag_selection="aic", max_lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    deterministic: &str,
    lag_selection: &str,
    max_lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    use crate::stationarity::DeterministicTerm;
    use crate::stationarity::LagSelection;
    let det = match deterministic.to_ascii_lowercase().as_str() {
      "n" | "none" => DeterministicTerm::None,
      "c" | "constant" => DeterministicTerm::Constant,
      "ct" | "trend" | "constant+trend" => DeterministicTerm::ConstantTrend,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "deterministic must be 'n', 'c' or 'ct', got '{o}'"
        )));
      }
    };
    let sel = match lag_selection.to_ascii_lowercase().as_str() {
      "aic" => LagSelection::Aic,
      "bic" => LagSelection::Bic,
      s => match s.parse::<usize>() {
        Ok(p) => LagSelection::Fixed(p),
        Err(_) => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lag_selection must be 'aic', 'bic', or a non-negative integer, got '{s}'"
          )));
        }
      },
    };
    let cfg = crate::stationarity::adf::ADFConfig {
      deterministic: det,
      lag_selection: sel,
      max_lags,
      alpha,
    };
    let view = y.as_array();
    Ok(Self {
      inner: crate::stationarity::adf::adf_test(view, cfg),
    })
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
  #[getter]
  fn reject_unit_root(&self) -> bool {
    self.inner.reject_unit_root
  }
  /// Returns `(1%, 5%, 10%)` critical values.
  #[getter]
  fn critical_values(&self) -> (f64, f64, f64) {
    let cv = &self.inner.critical_values;
    (cv.one_percent, cv.five_percent, cv.ten_percent)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "KPSSTest", unsendable)]
pub struct PyKPSSTest {
  inner: crate::stationarity::kpss::KPSSResult,
}

#[cfg(feature = "openblas")]
#[pyclass(name = "PhillipsPerronTest", unsendable)]
pub struct PyPhillipsPerronTest {
  inner: crate::stationarity::phillips_perron::PhillipsPerronResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyPhillipsPerronTest {
  /// `deterministic`: "n" / "c" / "ct". `test_type`: "tau" or "rho".
  #[new]
  #[pyo3(signature = (y, deterministic="c", test_type="tau", lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    deterministic: &str,
    test_type: &str,
    lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    use crate::stationarity::DeterministicTerm;
    use crate::stationarity::phillips_perron::PPTestType;
    let det = match deterministic.to_ascii_lowercase().as_str() {
      "n" | "none" => DeterministicTerm::None,
      "c" | "constant" => DeterministicTerm::Constant,
      "ct" | "trend" => DeterministicTerm::ConstantTrend,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "deterministic must be 'n'/'c'/'ct', got '{o}'"
        )));
      }
    };
    let tt = match test_type.to_ascii_lowercase().as_str() {
      "tau" => PPTestType::Tau,
      "rho" => PPTestType::Rho,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "test_type must be 'tau' or 'rho', got '{o}'"
        )));
      }
    };
    let cfg = crate::stationarity::phillips_perron::PhillipsPerronConfig {
      deterministic: det,
      test_type: tt,
      lags,
      alpha,
    };
    Ok(Self {
      inner: crate::stationarity::phillips_perron::phillips_perron_test(y.as_array(), cfg),
    })
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }
  #[getter]
  fn reject_unit_root(&self) -> Option<bool> {
    self.inner.reject_unit_root
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "ERSTest", unsendable)]
pub struct PyERSTest {
  inner: crate::stationarity::ers_dfgls::ERSResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyERSTest {
  /// Elliott-Rothenberg-Stock DF-GLS unit-root test.
  /// `trend`: "c" (constant) or "ct" (constant + trend).
  #[new]
  #[pyo3(signature = (y, trend="c", lag_selection="aic", max_lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    lag_selection: &str,
    max_lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    use crate::stationarity::LagSelection;
    use crate::stationarity::ers_dfgls::ERSTrend;
    let tr = match trend.to_ascii_lowercase().as_str() {
      "c" | "constant" => ERSTrend::Constant,
      "ct" | "trend" => ERSTrend::ConstantTrend,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "trend must be 'c' or 'ct', got '{o}'"
        )));
      }
    };
    let sel = match lag_selection.to_ascii_lowercase().as_str() {
      "aic" => LagSelection::Aic,
      "bic" => LagSelection::Bic,
      s => match s.parse::<usize>() {
        Ok(p) => LagSelection::Fixed(p),
        Err(_) => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lag_selection must be 'aic'/'bic'/integer, got '{s}'"
          )));
        }
      },
    };
    let cfg = crate::stationarity::ers_dfgls::ERSConfig {
      trend: tr,
      lag_selection: sel,
      max_lags,
      alpha,
    };
    Ok(Self {
      inner: crate::stationarity::ers_dfgls::ers_dfgls_test(y.as_array(), cfg),
    })
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
  #[getter]
  fn reject_unit_root(&self) -> bool {
    self.inner.reject_unit_root
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "LeybourneMcCabeTest", unsendable)]
pub struct PyLeybourneMcCabeTest {
  inner: crate::stationarity::leybourne_mccabe::LeybourneMcCabeResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyLeybourneMcCabeTest {
  /// Leybourne-McCabe stationarity test (parametric bootstrap p-value).
  #[new]
  #[pyo3(signature = (y, trend="level", ar_lags=1, bootstrap_samples=400, bootstrap_seed=1234, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    ar_lags: usize,
    bootstrap_samples: usize,
    bootstrap_seed: u64,
    alpha: f64,
  ) -> PyResult<Self> {
    use crate::stationarity::leybourne_mccabe::LMTrend;
    let tr = match trend.to_ascii_lowercase().as_str() {
      "level" | "c" => LMTrend::Level,
      "trend" | "ct" => LMTrend::Trend,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "trend must be 'level' or 'trend', got '{o}'"
        )));
      }
    };
    let cfg = crate::stationarity::leybourne_mccabe::LeybourneMcCabeConfig {
      trend: tr,
      ar_lags,
      bootstrap_samples,
      bootstrap_seed,
      alpha,
    };
    Ok(Self {
      inner: crate::stationarity::leybourne_mccabe::leybourne_mccabe_test(y.as_array(), cfg),
    })
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn ar_lags(&self) -> usize {
    self.inner.ar_lags
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn reject_stationarity(&self) -> bool {
    self.inner.reject_stationarity
  }
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyKPSSTest {
  #[new]
  #[pyo3(signature = (y, trend="level", lags=None, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    trend: &str,
    lags: Option<usize>,
    alpha: f64,
  ) -> PyResult<Self> {
    let t = match trend.to_ascii_lowercase().as_str() {
      "level" | "c" => crate::stationarity::kpss::KPSSTrend::Level,
      "trend" | "ct" => crate::stationarity::kpss::KPSSTrend::Trend,
      o => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "trend must be 'level' or 'trend', got '{o}'"
        )));
      }
    };
    let cfg = crate::stationarity::kpss::KPSSConfig {
      trend: t,
      lags,
      alpha,
    };
    let view = y.as_array();
    Ok(Self {
      inner: crate::stationarity::kpss::kpss_test(view, cfg),
    })
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
  #[getter]
  fn used_lags(&self) -> usize {
    self.inner.used_lags
  }
  #[getter]
  fn reject_stationarity(&self) -> bool {
    self.inner.reject_stationarity
  }
}

#[pyclass(name = "FukasawaHurst", unsendable)]
pub struct PyFukasawaHurst {
  inner: crate::fukasawa_hurst::FukasawaResult,
}

#[pymethods]
impl PyFukasawaHurst {
  /// Estimate Fukasawa Hurst directly from a price series.
  #[staticmethod]
  fn from_prices<'py>(closes: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::fukasawa_hurst::estimate_from_prices(closes.as_array()),
    }
  }

  /// Estimate from log realised variance series, with `m` intra-bucket samples
  /// and `delta` sampling step.
  #[staticmethod]
  fn from_log_rv<'py>(log_rv: PyReadonlyArray1<'py, f64>, m: usize, delta: f64) -> Self {
    Self {
      inner: crate::fukasawa_hurst::estimate(log_rv.as_array(), m, delta),
    }
  }

  #[getter]
  fn hurst(&self) -> f64 {
    self.inner.hurst
  }
  #[getter]
  fn eta(&self) -> f64 {
    self.inner.eta
  }
  #[getter]
  fn neg_log_lik(&self) -> f64 {
    self.inner.neg_log_lik
  }
  #[getter]
  fn n_obs(&self) -> usize {
    self.inner.n_obs
  }
}

#[pyclass(name = "FouEstimate", unsendable)]
pub struct PyFouEstimate {
  inner: crate::fou_estimator::FouEstimateResult,
}

#[pymethods]
impl PyFouEstimate {
  /// V1 estimator (Daubechies-filter-based).
  #[staticmethod]
  #[pyo3(signature = (path, delta=None, hurst_override=None))]
  fn v1<'py>(
    path: PyReadonlyArray1<'py, f64>,
    delta: Option<f64>,
    hurst_override: Option<f64>,
  ) -> Self {
    Self {
      inner: crate::fou_estimator::estimate_fou_v1(
        path.as_array(),
        crate::fou_estimator::FilterType::Daubechies,
        delta,
        hurst_override,
      ),
    }
  }

  /// V2 estimator (moments-based, no linear filters).
  #[staticmethod]
  #[pyo3(signature = (path, delta=None, hurst_override=None))]
  fn v2<'py>(
    path: PyReadonlyArray1<'py, f64>,
    delta: Option<f64>,
    hurst_override: Option<f64>,
  ) -> Self {
    let n = path.as_array().len();
    Self {
      inner: crate::fou_estimator::estimate_fou_v2(path.as_array(), delta, n, hurst_override),
    }
  }

  #[getter]
  fn hurst(&self) -> f64 {
    self.inner.hurst
  }
  #[getter]
  fn sigma(&self) -> f64 {
    self.inner.sigma
  }
  #[getter]
  fn mu(&self) -> f64 {
    self.inner.mu
  }
  #[getter]
  fn theta(&self) -> f64 {
    self.inner.theta
  }
}

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

#[pyclass(name = "GaussianKDE", unsendable)]
pub struct PyGaussianKDE {
  inner: crate::gaussian_kde::GaussianKDE,
}

#[pymethods]
impl PyGaussianKDE {
  /// Construct a Gaussian KDE with explicit bandwidth.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, bandwidth: f64) -> Self {
    Self {
      inner: crate::gaussian_kde::GaussianKDE::new(data.as_array().to_owned(), bandwidth),
    }
  }

  /// Construct a Gaussian KDE with Silverman's rule-of-thumb bandwidth.
  #[staticmethod]
  fn silverman<'py>(data: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::gaussian_kde::GaussianKDE::with_silverman_bandwidth(
        data.as_array().to_owned(),
      ),
    }
  }

  fn evaluate(&self, x: f64) -> f64 {
    self.inner.evaluate(x)
  }

  fn evaluate_array<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let out = self.inner.evaluate_array(&x.as_array().to_owned());
    out.into_pyarray(py)
  }
}

#[pyclass(name = "TailIndex", unsendable)]
pub struct PyTailIndex {
  xi: f64,
  alpha: f64,
}

#[pymethods]
impl PyTailIndex {
  /// Hill-style tail-exponent estimator (Mancini 2008). Provide pre-computed
  /// `mean` and `var` of the centred returns.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, mean: f64, var: f64) -> Self {
    let view = data.as_array();
    let xi = crate::tail_index::estimate_tail_exponent(view, mean, var);
    let alpha = crate::tail_index::tail_exponent_to_cgmy_alpha(xi);
    Self { xi, alpha }
  }

  #[getter]
  fn tail_exponent(&self) -> f64 {
    self.xi
  }
  /// CGMY α parameter implied by the tail exponent (`Y = α + 1`).
  #[getter]
  fn cgmy_alpha(&self) -> f64 {
    self.alpha
  }
}

#[pyclass(name = "Leverage", unsendable)]
pub struct PyLeverage {
  rho: f64,
}

#[pymethods]
impl PyLeverage {
  /// Estimate leverage correlation $\rho$ between price and volatility from
  /// a closing-price series.
  #[new]
  fn new<'py>(closes: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      rho: crate::leverage::estimate_leverage_rho(closes.as_array()),
    }
  }

  #[getter]
  fn rho(&self) -> f64 {
    self.rho
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
  fn new<'py>(
    s: PyReadonlyArray1<'py, f64>,
    r: f64,
    delta: Option<f64>,
    max_iters: usize,
  ) -> Self {
    let default_cfg = crate::heston_nml_cekf::HestonNMLECEKFConfig::default();
    let cfg = crate::heston_nml_cekf::HestonNMLECEKFConfig {
      r,
      delta: delta.unwrap_or(default_cfg.delta),
      max_iters,
      ..default_cfg
    };
    let arr = s.as_array().to_owned();
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

#[pyclass(name = "Cusum", unsendable)]
pub struct PyCusum {
  inner: crate::econometrics::changepoint::CusumResult,
}

#[pymethods]
impl PyCusum {
  /// CUSUM control chart with reference value `k` (half the smallest shift in
  /// SD units) and threshold `h`.
  #[new]
  fn new<'py>(series: PyReadonlyArray1<'py, f64>, k: f64, h: f64) -> Self {
    Self {
      inner: crate::econometrics::changepoint::cusum(series.as_array(), k, h),
    }
  }

  fn upper<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.upper.clone().into_pyarray(py)
  }
  fn lower<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.lower.clone().into_pyarray(py)
  }
  #[getter]
  fn alarms(&self) -> Vec<usize> {
    self.inner.alarms.clone()
  }
}

#[pyclass(name = "Pelt", unsendable)]
pub struct PyPelt {
  inner: crate::econometrics::changepoint::PeltResult,
}

#[pymethods]
impl PyPelt {
  /// PELT changepoint detection with mean-shift cost and `penalty` per
  /// changepoint. `min_size` enforces a minimum segment length.
  #[new]
  #[pyo3(signature = (series, penalty, min_size=1))]
  fn new<'py>(series: PyReadonlyArray1<'py, f64>, penalty: f64, min_size: usize) -> Self {
    Self {
      inner: crate::econometrics::changepoint::pelt(series.as_array(), penalty, min_size),
    }
  }

  #[getter]
  fn changepoints(&self) -> Vec<usize> {
    self.inner.changepoints.clone()
  }
  #[getter]
  fn cost(&self) -> f64 {
    self.inner.cost
  }
}

/// Spectral / periodogram bindings — Welch / Bartlett / Parzen periodograms via FFT.
#[pyclass(name = "PeriodogramFFT", unsendable)]
pub struct PyPeriodogramFFT {
  inner: crate::spectral::PeriodogramResult,
}

#[pymethods]
impl PyPeriodogramFFT {
  /// FFT periodogram with default config (mean-detrend, Hann window, one-sided density).
  #[new]
  #[pyo3(signature = (signal, sampling_rate=1.0))]
  fn new<'py>(signal: PyReadonlyArray1<'py, f64>, sampling_rate: f64) -> Self {
    let cfg = crate::spectral::PeriodogramConfig {
      sampling_rate,
      ..crate::spectral::PeriodogramConfig::default()
    };
    Self {
      inner: crate::spectral::periodogram_fft(signal.as_array(), cfg),
    }
  }

  fn frequencies(&self) -> Vec<f64> {
    self.inner.frequencies.clone()
  }
  fn spectrum(&self) -> Vec<f64> {
    self.inner.spectrum.clone()
  }
  #[getter]
  fn resolution_hz(&self) -> f64 {
    self.inner.resolution_hz
  }
  #[getter]
  fn nfft(&self) -> usize {
    self.inner.nfft
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "EngleGranger", unsendable)]
pub struct PyEngleGranger {
  inner: crate::econometrics::cointegration::EngleGrangerResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyEngleGranger {
  /// Engle-Granger 2-step cointegration test for `y_t = α + β x_t + ε_t`.
  #[new]
  fn new<'py>(y: PyReadonlyArray1<'py, f64>, x: PyReadonlyArray1<'py, f64>) -> Self {
    Self {
      inner: crate::econometrics::cointegration::engle_granger_test(y.as_array(), x.as_array()),
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
  fn adf_statistic(&self) -> f64 {
    self.inner.adf_statistic
  }
  #[getter]
  fn critical_values(&self) -> (f64, f64, f64) {
    self.inner.critical_values
  }
  #[getter]
  fn reject_no_cointegration(&self) -> bool {
    self.inner.reject_no_cointegration
  }
  fn residuals<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.residuals.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "Johansen", unsendable)]
pub struct PyJohansen {
  inner: crate::econometrics::cointegration::JohansenResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyJohansen {
  /// Johansen trace test on an `(t, k)` matrix.
  #[new]
  #[pyo3(signature = (series, lags=1))]
  fn new<'py>(series: numpy::PyReadonlyArray2<'py, f64>, lags: usize) -> Self {
    Self {
      inner: crate::econometrics::cointegration::johansen_test(series.as_array(), lags),
    }
  }

  fn eigenvalues<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.eigenvalues.clone().into_pyarray(py)
  }
  fn trace_statistics<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_statistics.clone().into_pyarray(py)
  }
  fn trace_critical_5pct<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trace_critical_5pct.clone().into_pyarray(py)
  }
}

#[cfg(feature = "openblas")]
#[pyclass(name = "Granger", unsendable)]
pub struct PyGranger {
  inner: crate::econometrics::granger::GrangerResult,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyGranger {
  /// Granger causality of `x` → `y` with `lags` lags at significance `alpha`.
  #[new]
  #[pyo3(signature = (y, x, lags, alpha=0.05))]
  fn new<'py>(
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    lags: usize,
    alpha: f64,
  ) -> Self {
    Self {
      inner: crate::econometrics::granger::granger_causality(
        y.as_array(),
        x.as_array(),
        lags,
        alpha,
      ),
    }
  }

  #[getter]
  fn f_statistic(&self) -> f64 {
    self.inner.f_statistic
  }
  #[getter]
  fn p_value(&self) -> f64 {
    self.inner.p_value
  }
  #[getter]
  fn lags(&self) -> usize {
    self.inner.lags
  }
  #[getter]
  fn nobs(&self) -> usize {
    self.inner.nobs
  }
  #[getter]
  fn reject_no_causality(&self) -> bool {
    self.inner.reject_no_causality
  }
}

/// Random-walk Metropolis-Hastings with a Python-callable log target.
///
/// `log_target` is a Python function `numpy.ndarray -> float` that may return
/// `-inf` to encode hard constraints. The chain runs in pure Python-callback
/// mode (single-threaded, GIL-bound), so it is most useful for low-dimensional
/// parameter spaces. For performance-critical work consider porting the
/// log-density to Rust and exposing it as a closed-form distribution instead.
#[pyfunction]
#[pyo3(signature = (initial, log_target, proposal_scale, n_samples, burn_in=1000, seed=42))]
pub fn random_walk_metropolis<'py>(
  py: Python<'py>,
  initial: PyReadonlyArray1<'py, f64>,
  log_target: pyo3::Py<pyo3::PyAny>,
  proposal_scale: PyReadonlyArray1<'py, f64>,
  n_samples: usize,
  burn_in: usize,
  seed: u64,
) -> PyResult<(
  pyo3::Bound<'py, numpy::PyArray2<f64>>,
  pyo3::Bound<'py, numpy::PyArray1<f64>>,
  f64,
)> {
  use ndarray::ArrayView1;
  use numpy::IntoPyArray;
  let init_arr = initial.as_array().to_owned();
  let scale_arr = proposal_scale.as_array().to_owned();
  let log_target_ref = &log_target;
  let target_fn = |theta: ArrayView1<f64>| -> f64 {
    let theta_owned: Vec<f64> = theta.iter().copied().collect();
    let theta_pyarr = numpy::PyArray1::from_vec(py, theta_owned);
    let res = log_target_ref.call1(py, (theta_pyarr,));
    match res {
      Ok(obj) => obj.extract::<f64>(py).unwrap_or(f64::NEG_INFINITY),
      Err(_) => f64::NEG_INFINITY,
    }
  };
  let res = crate::filtering::mcmc::random_walk_metropolis(
    init_arr.view(),
    target_fn,
    scale_arr.view(),
    n_samples,
    burn_in,
    seed,
  );
  Ok((
    res.samples.into_pyarray(py),
    res.log_targets.into_pyarray(py),
    res.acceptance_rate,
  ))
}

#[cfg(feature = "openblas")]
#[pyclass(name = "GaussianHmm", unsendable)]
pub struct PyGaussianHmm {
  inner: crate::econometrics::hmm::GaussianHmm,
}

#[cfg(feature = "openblas")]
#[pymethods]
impl PyGaussianHmm {
  /// Construct a Gaussian-emission HMM with `K` hidden states.
  #[new]
  fn new<'py>(
    initial: PyReadonlyArray1<'py, f64>,
    transitions: numpy::PyReadonlyArray2<'py, f64>,
    means: PyReadonlyArray1<'py, f64>,
    stds: PyReadonlyArray1<'py, f64>,
  ) -> Self {
    Self {
      inner: crate::econometrics::hmm::GaussianHmm::new(
        initial.as_array().to_owned(),
        transitions.as_array().to_owned(),
        means.as_array().to_owned(),
        stds.as_array().to_owned(),
      ),
    }
  }

  fn n_states(&self) -> usize {
    self.inner.n_states()
  }

  fn log_likelihood<'py>(&self, observations: PyReadonlyArray1<'py, f64>) -> f64 {
    self.inner.log_likelihood(observations.as_array())
  }

  fn viterbi<'py>(
    &self,
    py: Python<'py>,
    observations: PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<usize>> {
    use numpy::IntoPyArray;
    self
      .inner
      .viterbi(observations.as_array())
      .into_pyarray(py)
  }

  /// Train via Baum-Welch EM and return `(iterations, log_likelihood, converged)`.
  fn baum_welch<'py>(
    &mut self,
    observations: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
  ) -> (usize, f64, bool) {
    let fit = self.inner.baum_welch(observations.as_array(), max_iter, tol);
    (fit.iterations, fit.log_likelihood, fit.converged)
  }
}

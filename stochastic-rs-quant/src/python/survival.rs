use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::credit::index::flat_survival;
use crate::credit::survival_curve::HazardInterpolation;
use crate::credit::survival_curve::SurvivalCurve;

fn interpolation(method: &str) -> PyResult<HazardInterpolation> {
  match method {
    "piecewise_constant_hazard" | "hazard" => Ok(HazardInterpolation::PiecewiseConstantHazard),
    "linear_survival" | "linear" => Ok(HazardInterpolation::LinearSurvival),
    other => Err(PyValueError::new_err(format!(
      "method must be 'piecewise_constant_hazard' or 'linear_survival', got '{other}'"
    ))),
  }
}

fn checked(times: &[f64], values: &[f64], what: &str) -> PyResult<()> {
  if times.is_empty() || times.len() != values.len() {
    return Err(PyValueError::new_err(format!(
      "times and {what} must be non-empty and of equal length, got {} and {}",
      times.len(),
      values.len()
    )));
  }
  if times.windows(2).any(|w| w[1] <= w[0])
    || times[0] < 0.0
    || !times.iter().all(|t| t.is_finite())
  {
    return Err(PyValueError::new_err(
      "times must be finite, non-negative and strictly increasing",
    ));
  }
  if !values.iter().all(|v| v.is_finite()) {
    return Err(PyValueError::new_err(format!("{what} must be finite")));
  }
  Ok(())
}

/// Term structure of survival probabilities: piecewise-constant hazard or
/// linear survival interpolation between the pillars, flat hazard
/// extrapolation beyond the last one. A `(0, 1)` anchor is always present,
/// so `len`, `times()` and `pillars()` count one more entry than the
/// constructor received unless a `t = 0` pillar was given. Accepted
/// anywhere a flat hazard rate is — `ExposureProfile.cva`, `CdsIndex`,
/// `CdoTranche`.
#[pyclass(module = "stochastic_rs", name = "SurvivalCurve", skip_from_py_object)]
#[derive(Clone)]
pub struct PySurvivalCurve {
  pub(crate) inner: SurvivalCurve<f64>,
}

#[pymethods]
impl PySurvivalCurve {
  /// Pillars `times` (years) with the hazard rate that applies up to each.
  #[new]
  #[pyo3(signature = (times, hazard_rates, method="piecewise_constant_hazard"))]
  fn new(times: Vec<f64>, hazard_rates: Vec<f64>, method: &str) -> PyResult<Self> {
    checked(&times, &hazard_rates, "hazard_rates")?;
    if hazard_rates.iter().any(|h| *h < 0.0) {
      return Err(PyValueError::new_err("hazard_rates must be non-negative"));
    }
    Ok(Self {
      inner: SurvivalCurve::from_hazard_rates(
        &Array1::from_vec(times),
        &Array1::from_vec(hazard_rates),
        interpolation(method)?,
      ),
    })
  }

  /// One hazard rate for every horizon.
  #[staticmethod]
  fn flat(hazard_rate: f64) -> PyResult<Self> {
    if !(hazard_rate.is_finite() && hazard_rate >= 0.0) {
      return Err(PyValueError::new_err(format!(
        "hazard_rate must be finite and non-negative, got {hazard_rate}"
      )));
    }
    Ok(Self {
      inner: flat_survival(hazard_rate),
    })
  }

  /// Pillars with the survival probability observed at each.
  #[staticmethod]
  #[pyo3(signature = (times, survival_probs, method="piecewise_constant_hazard"))]
  fn from_survival_probs(
    times: Vec<f64>,
    survival_probs: Vec<f64>,
    method: &str,
  ) -> PyResult<Self> {
    checked(&times, &survival_probs, "survival_probs")?;
    if survival_probs.iter().any(|s| !(*s > 0.0 && *s <= 1.0)) {
      return Err(PyValueError::new_err("survival_probs must lie in (0, 1]"));
    }
    Ok(Self {
      inner: SurvivalCurve::from_survival_probs(
        &Array1::from_vec(times),
        &Array1::from_vec(survival_probs),
        interpolation(method)?,
      ),
    })
  }

  /// Pillars with the cumulative default probability observed at each.
  #[staticmethod]
  #[pyo3(signature = (times, default_probs, method="piecewise_constant_hazard"))]
  fn from_default_probs(times: Vec<f64>, default_probs: Vec<f64>, method: &str) -> PyResult<Self> {
    checked(&times, &default_probs, "default_probs")?;
    if default_probs.iter().any(|d| !(*d >= 0.0 && *d < 1.0)) {
      return Err(PyValueError::new_err("default_probs must lie in [0, 1)"));
    }
    Ok(Self {
      inner: SurvivalCurve::from_default_probs(
        &Array1::from_vec(times),
        &Array1::from_vec(default_probs),
        interpolation(method)?,
      ),
    })
  }

  fn survival_probability(&self, t: f64) -> f64 {
    self.inner.survival_probability(t)
  }

  fn default_probability(&self, t: f64) -> f64 {
    self.inner.default_probability(t)
  }

  /// Probability of default in `(t1, t2]` given survival to `t1`.
  fn conditional_default_probability(&self, t1: f64, t2: f64) -> f64 {
    self.inner.conditional_default_probability(t1, t2)
  }

  /// Average hazard rate over `(t1, t2]`.
  fn forward_hazard(&self, t1: f64, t2: f64) -> f64 {
    self.inner.forward_hazard(t1, t2)
  }

  /// Average hazard rate over `(0, t]`.
  fn average_hazard(&self, t: f64) -> f64 {
    self.inner.average_hazard(t)
  }

  fn survival_probabilities(&self, times: Vec<f64>) -> Vec<f64> {
    self
      .inner
      .survival_probabilities(&Array1::from_vec(times))
      .to_vec()
  }

  fn default_probabilities(&self, times: Vec<f64>) -> Vec<f64> {
    self
      .inner
      .default_probabilities(&Array1::from_vec(times))
      .to_vec()
  }

  /// Pillar times, the `t = 0` anchor first.
  fn times(&self) -> Vec<f64> {
    self.inner.points().iter().map(|p| p.time).collect()
  }

  /// Survival probability at each pillar, `1.0` at the anchor first;
  /// `from_survival_probs(times(), pillars())` rebuilds the curve.
  fn pillars(&self) -> Vec<f64> {
    self
      .inner
      .points()
      .iter()
      .map(|p| p.survival_probability)
      .collect()
  }

  fn __len__(&self) -> usize {
    self.inner.len()
  }
}

/// A flat hazard rate (`float`) or a [`PySurvivalCurve`], as the credit
/// classes accept either.
pub(super) fn survival_input(obj: &Bound<'_, PyAny>) -> PyResult<SurvivalCurve<f64>> {
  if let Ok(curve) = obj.extract::<PyRef<'_, PySurvivalCurve>>() {
    return Ok(curve.inner.clone());
  }
  let hazard = obj
    .extract::<f64>()
    .map_err(|_| PyValueError::new_err("expected a flat hazard rate (float) or a SurvivalCurve"))?;
  Ok(PySurvivalCurve::flat(hazard)?.inner)
}

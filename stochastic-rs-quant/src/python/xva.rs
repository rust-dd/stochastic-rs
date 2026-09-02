use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use crate::credit::survival_curve::HazardInterpolation;
use crate::credit::survival_curve::SurvivalCurve;
use crate::risk::xva;

fn flat_survival(hazard_rate: f64) -> PyResult<SurvivalCurve<f64>> {
  if !(hazard_rate.is_finite() && hazard_rate >= 0.0) {
    return Err(PyValueError::new_err(
      "hazard_rate must be finite and non-negative",
    ));
  }
  Ok(SurvivalCurve::from_hazard_rates(
    &Array1::from_vec(vec![1.0, 5.0, 10.0, 30.0]),
    &Array1::from_vec(vec![hazard_rate; 4]),
    HazardInterpolation::PiecewiseConstantHazard,
  ))
}

/// Exposure profile (EPE / ENE / PFE on a date grid) with the valuation
/// adjustments integrated against a discount curve and flat hazard rates.
#[pyclass(name = "ExposureProfile", unsendable)]
pub struct PyExposureProfile {
  inner: xva::ExposureProfile,
}

#[pymethods]
impl PyExposureProfile {
  /// `mtm` is a `(paths, dates)` array of mark-to-market values.
  #[staticmethod]
  #[pyo3(signature = (mtm, times, quantile=0.95))]
  fn from_mtm(mtm: numpy::PyReadonlyArray2<'_, f64>, times: Vec<f64>, quantile: f64) -> Self {
    Self {
      inner: xva::ExposureProfile::from_mtm(&mtm.as_array().to_owned(), times, quantile),
    }
  }

  fn times(&self) -> Vec<f64> {
    self.inner.times.clone()
  }

  fn epe(&self) -> Vec<f64> {
    self.inner.epe.clone()
  }

  fn ene(&self) -> Vec<f64> {
    self.inner.ene.clone()
  }

  fn pfe(&self) -> Vec<f64> {
    self.inner.pfe.clone()
  }

  fn peak_epe(&self) -> f64 {
    self.inner.peak_epe()
  }

  fn average_epe(&self) -> f64 {
    self.inner.average_epe()
  }

  /// Unilateral CVA with a flat counterparty hazard rate.
  fn cva(&self, hazard_rate: f64, discount: &PyDiscountCurve, lgd: f64) -> PyResult<f64> {
    Ok(xva::cva(
      &self.inner,
      &flat_survival(hazard_rate)?,
      &discount.inner,
      lgd,
    ))
  }

  /// Unilateral DVA with a flat own hazard rate.
  fn dva(&self, own_hazard_rate: f64, discount: &PyDiscountCurve, lgd_own: f64) -> PyResult<f64> {
    Ok(xva::dva(
      &self.inner,
      &flat_survival(own_hazard_rate)?,
      &discount.inner,
      lgd_own,
    ))
  }

  /// Bilateral CVA: counterparty default weighted by the bank's survival.
  fn bilateral_cva(
    &self,
    hazard_rate: f64,
    own_hazard_rate: f64,
    discount: &PyDiscountCurve,
    lgd: f64,
  ) -> PyResult<f64> {
    Ok(xva::bilateral_cva(
      &self.inner,
      &flat_survival(hazard_rate)?,
      &flat_survival(own_hazard_rate)?,
      &discount.inner,
      lgd,
    ))
  }

  /// Symmetric FVA (`FCA − FBA`) for a funding spread.
  fn fva(&self, discount: &PyDiscountCurve, funding_spread: f64) -> f64 {
    xva::fva(&self.inner, &discount.inner, funding_spread)
  }

  fn fca(&self, discount: &PyDiscountCurve, funding_spread: f64) -> f64 {
    xva::fca(&self.inner, &discount.inner, funding_spread)
  }

  fn fba(&self, discount: &PyDiscountCurve, funding_spread: f64) -> f64 {
    xva::fba(&self.inner, &discount.inner, funding_spread)
  }
}

/// Payer swap exposure under Hull–White on a discount curve.
#[pyclass(name = "HullWhiteSwapExposure", unsendable)]
pub struct PyHullWhiteSwapExposure {
  inner: xva::irs::HullWhiteSwapExposure,
}

#[pymethods]
impl PyHullWhiteSwapExposure {
  /// `fixed_rate=None` sets the swap at par on `curve` when `profile` runs.
  #[new]
  #[pyo3(signature = (mean_reversion, sigma, notional, payment_times, accrual=1.0, fixed_rate=None, steps_per_year=52))]
  fn new(
    mean_reversion: f64,
    sigma: f64,
    notional: f64,
    payment_times: Vec<f64>,
    accrual: f64,
    fixed_rate: Option<f64>,
    steps_per_year: usize,
  ) -> Self {
    Self {
      inner: xva::irs::HullWhiteSwapExposure::new(
        mean_reversion,
        sigma,
        notional,
        fixed_rate.unwrap_or(f64::NAN),
        payment_times,
        accrual,
      )
      .with_steps_per_year(steps_per_year),
    }
  }

  /// Par fixed rate on `curve`.
  fn par_rate(&self, curve: &PyDiscountCurve) -> f64 {
    self.inner.par_rate(&curve.inner)
  }

  /// Exposure profile on the payment dates from `paths` simulated short-rate
  /// paths; a `NaN` fixed rate is replaced by the par rate on `curve`.
  #[pyo3(signature = (curve, paths=10_000, quantile=0.95, seed=None))]
  fn profile(
    &self,
    curve: &PyDiscountCurve,
    paths: usize,
    quantile: f64,
    seed: Option<u64>,
  ) -> PyExposureProfile {
    let mut swap = self.inner.clone();
    if swap.fixed_rate.is_nan() {
      swap.fixed_rate = swap.par_rate(&curve.inner);
    }
    let inner = match seed {
      Some(s) => swap.profile(
        &curve.inner,
        paths,
        quantile,
        stochastic_rs_core::simd_rng::Deterministic::new(s),
      ),
      None => swap.profile(
        &curve.inner,
        paths,
        quantile,
        stochastic_rs_core::simd_rng::Unseeded,
      ),
    };
    PyExposureProfile { inner }
  }
}

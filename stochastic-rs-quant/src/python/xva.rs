use pyo3::prelude::*;

use super::curves::PyDiscountCurve;
use super::survival::survival_input;
use crate::risk::xva;

/// Exposure profile (EPE / ENE / PFE on a date grid) with the valuation
/// adjustments integrated against a discount curve; every `hazard_rate`
/// argument takes a flat rate or a `SurvivalCurve`.
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

  /// Unilateral CVA against the counterparty's survival (flat rate or curve).
  fn cva(
    &self,
    hazard_rate: &Bound<'_, PyAny>,
    discount: &PyDiscountCurve,
    lgd: f64,
  ) -> PyResult<f64> {
    Ok(xva::cva(
      &self.inner,
      &survival_input(hazard_rate)?,
      &discount.inner,
      lgd,
    ))
  }

  /// Unilateral DVA against the bank's own survival (flat rate or curve).
  fn dva(
    &self,
    own_hazard_rate: &Bound<'_, PyAny>,
    discount: &PyDiscountCurve,
    lgd_own: f64,
  ) -> PyResult<f64> {
    Ok(xva::dva(
      &self.inner,
      &survival_input(own_hazard_rate)?,
      &discount.inner,
      lgd_own,
    ))
  }

  /// Bilateral CVA: counterparty default weighted by the bank's survival.
  fn bilateral_cva(
    &self,
    hazard_rate: &Bound<'_, PyAny>,
    own_hazard_rate: &Bound<'_, PyAny>,
    discount: &PyDiscountCurve,
    lgd: f64,
  ) -> PyResult<f64> {
    Ok(xva::bilateral_cva(
      &self.inner,
      &survival_input(hazard_rate)?,
      &survival_input(own_hazard_rate)?,
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

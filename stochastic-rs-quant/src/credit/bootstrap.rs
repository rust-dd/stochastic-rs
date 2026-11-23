//! Piecewise-constant hazard curve from CDS par spreads sorted by maturity: each `h_k` is the
//! Brent root that reprices the `k`-th quote given `h_1, …, h_{k−1}`.
//!
//! $$
//! h(t)=h_k\quad\text{for }t\in(t_{k-1},t_k]
//! $$
//!
//! Reference: O'Kane & Turnbull (2003), "Valuation of Credit Default Swaps", Lehman Brothers QCR
//! Reference: ISDA (2009), "The ISDA CDS Standard Model", <https://www.cdsmodel.com>

use chrono::NaiveDate;
use ndarray::Array1;
use stochastic_rs_distributions::brent;

use super::cds::CdsPosition;
use super::cds::CreditDefaultSwap;
use super::survival_curve::HazardInterpolation;
use super::survival_curve::SurvivalCurve;
use crate::QuantError;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::cashflows::CurveProvider;
use crate::traits::RealExt;

/// A single CDS quote fed into [`bootstrap_hazard`].
#[derive(Debug, Clone)]
pub struct CdsQuote {
  /// Contract maturity, after the bootstrap's effective and valuation dates.
  pub maturity: NaiveDate,
  /// Finite par (fair) running spread in decimal form (e.g. 0.01 = 100 bps).
  pub spread: f64,
  /// Recovery rate of this quote in `[0, 1)`; `None` takes `bootstrap_hazard`'s `default_recovery`.
  pub recovery_rate: Option<f64>,
  /// Premium-leg payment frequency (ISDA standard: quarterly).
  pub frequency: Frequency,
  /// Day count for the premium accrual (ISDA standard: ACT/360).
  pub premium_day_count: DayCountConvention,
}

impl CdsQuote {
  /// Build a quote with ISDA defaults (quarterly schedule, ACT/360).
  pub fn isda(maturity: NaiveDate, spread: f64) -> Self {
    Self {
      maturity,
      spread,
      recovery_rate: None,
      frequency: Frequency::Quarterly,
      premium_day_count: DayCountConvention::Actual360,
    }
  }
}

/// Hazards repricing each quote at par; `InvalidQuote` for a quote off its fields' domains or that
/// no hazard reprices, `InsufficientData` for none; panics unless `0 ≤ default_recovery < 1`.
pub fn bootstrap_hazard<T: RealExt>(
  valuation_date: NaiveDate,
  effective_date: NaiveDate,
  quotes: &[CdsQuote],
  default_recovery: T,
  discount: &(impl CurveProvider<T> + ?Sized),
  discount_day_count: DayCountConvention,
) -> Result<SurvivalCurve<T>, QuantError> {
  assert!(
    default_recovery >= T::zero() && default_recovery < T::one(),
    "default_recovery must satisfy `default_recovery >= T::zero() && default_recovery < T::one()`, got default_recovery = {default_recovery:?}"
  );
  if quotes.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "cds quotes",
      needed: 1,
      got: 0,
    });
  }

  let mut sorted = Vec::with_capacity(quotes.len());
  for (index, quote) in quotes.iter().enumerate() {
    let recovery = quote
      .recovery_rate
      .map_or(default_recovery, T::from_f64_fast);
    if let Some(reason) = defect(quote, recovery, effective_date, valuation_date) {
      return Err(QuantError::InvalidQuote { index, reason });
    }
    sorted.push((index, quote, recovery));
  }
  sorted.sort_by_key(|&(_, quote, _)| quote.maturity);

  let mut times = Vec::<T>::with_capacity(sorted.len());
  let mut hazards = Vec::<T>::with_capacity(sorted.len());

  for (index, quote, recovery) in sorted {
    let spread = T::from_f64_fast(quote.spread);

    let cds = CreditDefaultSwap::vanilla(
      CdsPosition::Buyer,
      T::one(),
      spread,
      recovery,
      effective_date,
      quote.maturity,
      quote.frequency,
      quote.premium_day_count,
    );

    let tau_k: T = DayCountConvention::Actual365Fixed.year_fraction(valuation_date, quote.maturity);

    let anchor_times = times.clone();
    let anchor_hazards = hazards.clone();

    let f = |h: f64| -> f64 {
      let h_t = T::from_f64_fast(h);
      let mut trial_times = anchor_times.clone();
      let mut trial_hazards = anchor_hazards.clone();
      trial_times.push(tau_k);
      trial_hazards.push(h_t);

      let curve = SurvivalCurve::from_hazard_rates(
        Array1::from(trial_times).view(),
        Array1::from(trial_hazards).view(),
        HazardInterpolation::PiecewiseConstantHazard,
      );

      let valuation = cds.valuation(valuation_date, discount_day_count, discount, &curve);
      (valuation.fair_spread - spread)
        .to_f64()
        .unwrap_or(f64::NAN)
    };

    let init_guess = spread.to_f64().unwrap_or(0.01).max(1e-8)
      / (1.0 - recovery.to_f64().unwrap_or(0.4)).max(1e-6);
    let (mut lo, mut hi) = (1e-10_f64, (init_guess * 10.0).max(1.0));
    while f(hi) < 0.0 && hi < 50.0 {
      hi *= 2.0;
    }
    while f(lo) > 0.0 && lo > 1e-14 {
      lo *= 0.5;
    }

    let h_k = brent::zero(lo, hi, 1e-12, &f).ok_or_else(|| QuantError::InvalidQuote {
      index,
      reason: format!(
        "no hazard rate in [{lo:?}, {hi:?}] reprices spread {:?} at maturity {}",
        quote.spread, quote.maturity
      ),
    })?;

    times.push(tau_k);
    hazards.push(T::from_f64_fast(h_k));
  }

  Ok(SurvivalCurve::from_hazard_rates(
    Array1::from(times).view(),
    Array1::from(hazards).view(),
    HazardInterpolation::PiecewiseConstantHazard,
  ))
}

/// Why `quote` cannot be bootstrapped, checked before `CreditDefaultSwap::new` could panic on it.
fn defect<T: RealExt>(
  quote: &CdsQuote,
  recovery: T,
  effective_date: NaiveDate,
  valuation_date: NaiveDate,
) -> Option<String> {
  let (spread, maturity) = (T::from_f64_fast(quote.spread), quote.maturity);
  if !spread.is_finite() {
    Some(format!(
      "spread must satisfy `spread.is_finite()`, got spread = {spread:?}"
    ))
  } else if !(recovery >= T::zero() && recovery < T::one()) {
    Some(format!(
      "recovery_rate must satisfy `recovery_rate >= T::zero() && recovery_rate < T::one()`, got recovery_rate = {recovery:?}"
    ))
  } else if !(maturity > effective_date && maturity > valuation_date) {
    Some(format!(
      "maturity must satisfy `maturity > effective_date && maturity > valuation_date`, got maturity = {maturity}, effective_date = {effective_date}, valuation_date = {valuation_date}"
    ))
  } else {
    None
  }
}

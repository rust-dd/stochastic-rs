//! Hazard-rate bootstrapping from a CDS par-spread term structure.
//!
//! Reference: O'Kane & Turnbull, "Valuation of Credit Default Swaps", Lehman
//! Brothers Quantitative Credit Research Quarterly (2003).
//!
//! Reference: ISDA, "The ISDA CDS Standard Model" (2009).
//!
//! Given a set of par-spread CDS quotes sorted by maturity, solve iteratively
//! for a piecewise-constant hazard-rate curve
//! $$
//! h(t)=h_k\quad\text{for }t\in(t_{k-1},t_k],
//! $$
//! such that the fair running spread of the $k$-th CDS matches its quoted
//! level $s_k$, given all previously solved hazard levels
//! $h_1,\ldots,h_{k-1}$.
//!
//! The fair-spread equation is monotone in $h_k$ (higher hazard → higher
//! protection PV, higher fair spread), so Brent's bracketing method converges
//! reliably.

use chrono::NaiveDate;
use ndarray::Array1;
use roots::SimpleConvergency;
use roots::find_root_brent;

use super::cds::CdsPosition;
use super::cds::CreditDefaultSwap;
use super::survival_curve::HazardInterpolation;
use super::survival_curve::SurvivalCurve;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Frequency;
use crate::quant::cashflows::CurveProvider;
use crate::traits::FloatExt;

/// A single CDS quote fed into [`bootstrap_hazard`].
#[derive(Debug, Clone)]
pub struct CdsQuote {
  /// Contract maturity.
  pub maturity: NaiveDate,
  /// Par (fair) running spread in decimal form (e.g. 0.01 = 100 bps).
  pub spread: f64,
  /// Recovery rate used for the bootstrap (defaults to 0.4 if `None`).
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

/// Bootstrap a piecewise-constant hazard survival curve from a CDS term
/// structure, given a risk-free discount curve.
///
/// The output curve stores survival probabilities at each quoted maturity and
/// uses [`HazardInterpolation::PiecewiseConstantHazard`] for interpolation.
pub fn bootstrap_hazard<T: FloatExt>(
  valuation_date: NaiveDate,
  effective_date: NaiveDate,
  quotes: &[CdsQuote],
  default_recovery: T,
  discount: &(impl CurveProvider<T> + ?Sized),
  discount_day_count: DayCountConvention,
) -> SurvivalCurve<T> {
  assert!(!quotes.is_empty(), "at least one CDS quote required");
  assert!(
    default_recovery >= T::zero() && default_recovery < T::one(),
    "default_recovery must lie in [0, 1)"
  );

  let mut sorted: Vec<CdsQuote> = quotes.to_vec();
  sorted.sort_by_key(|q| q.maturity);

  let mut times: Vec<T> = Vec::with_capacity(sorted.len());
  let mut hazards: Vec<T> = Vec::with_capacity(sorted.len());

  for (k, quote) in sorted.iter().enumerate() {
    let recovery = quote
      .recovery_rate
      .map(T::from_f64_fast)
      .unwrap_or(default_recovery);
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
      let h_t = T::from_f64_fast(h.max(0.0));
      let mut trial_times = anchor_times.clone();
      let mut trial_hazards = anchor_hazards.clone();
      trial_times.push(tau_k);
      trial_hazards.push(h_t);

      let curve = SurvivalCurve::from_hazard_rates(
        &Array1::from(trial_times),
        &Array1::from(trial_hazards),
        HazardInterpolation::PiecewiseConstantHazard,
      );

      let valuation = cds.valuation(valuation_date, discount_day_count, discount, &curve);
      (valuation.fair_spread - spread)
        .to_f64()
        .unwrap_or(f64::NAN)
    };

    let mut convergency = SimpleConvergency::<f64> {
      eps: 1e-12,
      max_iter: 80,
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

    let h_k = find_root_brent(lo, hi, &f, &mut convergency).unwrap_or(init_guess);
    let h_k = T::from_f64_fast(h_k.max(0.0));

    times.push(tau_k);
    hazards.push(h_k);

    let _ = k;
  }

  SurvivalCurve::from_hazard_rates(
    &Array1::from(times),
    &Array1::from(hazards),
    HazardInterpolation::PiecewiseConstantHazard,
  )
}

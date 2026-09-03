//! # CDS index
//!
//! $$
//! \mathrm{PV}_{\text{prot}} = \sum_i w_i\,\mathrm{PV}^{(i)}_{\text{prot}},\qquad
//! \mathrm{RPV01} = \sum_i w_i\,\mathrm{RPV01}^{(i)},\qquad
//! s^\star = \frac{\mathrm{PV}_{\text{prot}}}{\mathrm{RPV01}}
//! $$
//!
//! An untranched credit index (CDX / iTraxx style): a weighted portfolio of
//! single-name CDS on one common running coupon, valued name by name with
//! the crate's ISDA-style single-name engine and aggregated by notional
//! weight, so the index fair spread is the annuity-weighted average of the
//! names' fair spreads. Traded indices settle upfront on a quoted spread
//! through the ISDA standard model convention: a flat hazard rate is solved
//! so that a running CDS at the quoted spread has zero value (recovery 40 %),
//! and the upfront is the value of the coupon shortfall on that curve.
//!
//! Reference: O'Kane, D. (2008), *Modelling Single-name and Multi-name
//! Credit Derivatives*, Wiley, Ch. 10–11; ISDA (2009), *CDS Standard Model*.

use chrono::NaiveDate;
use ndarray::Array1;

use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::cashflows::CurveProvider;
use crate::credit::cds::CdsPosition;
use crate::credit::cds::CreditDefaultSwap;
use crate::credit::survival_curve::HazardInterpolation;
use crate::credit::survival_curve::SurvivalCurve;

/// One constituent of the index.
#[derive(Clone, Debug)]
pub struct IndexName {
  /// Notional weight (the weights sum to one).
  pub weight: f64,
  /// Recovery rate of the name.
  pub recovery: f64,
  /// Survival curve of the name.
  pub survival: SurvivalCurve<f64>,
}

/// Index valuation from the protection buyer's side.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexValuation {
  pub protection_leg_npv: f64,
  pub premium_leg_npv: f64,
  /// Present value of a running premium of one on the surviving notional.
  pub risky_annuity: f64,
  /// Index fair spread.
  pub fair_spread: f64,
  /// Buyer's net value `protection − premium`.
  pub net_npv: f64,
}

/// Untranched CDS index on a running coupon.
#[derive(Clone, Debug)]
pub struct CdsIndex {
  pub names: Vec<IndexName>,
  /// Running coupon (decimal).
  pub coupon: f64,
  pub notional: f64,
  pub effective_date: NaiveDate,
  pub maturity_date: NaiveDate,
  pub frequency: Frequency,
  pub day_count: DayCountConvention,
}

impl CdsIndex {
  /// Standard quarterly Actual/360 index.
  pub fn new(
    names: Vec<IndexName>,
    coupon: f64,
    notional: f64,
    effective_date: NaiveDate,
    maturity_date: NaiveDate,
  ) -> Self {
    assert!(!names.is_empty(), "an index needs at least one name");
    let total: f64 = names.iter().map(|n| n.weight).sum();
    assert!(
      (total - 1.0).abs() < 1e-8,
      "index weights must sum to one (got {total})"
    );
    assert!(
      effective_date < maturity_date,
      "effective date must precede maturity"
    );
    Self {
      names,
      coupon,
      notional,
      effective_date,
      maturity_date,
      frequency: Frequency::Quarterly,
      day_count: DayCountConvention::Actual360,
    }
  }

  /// Equally weighted index of names sharing one recovery rate.
  pub fn homogeneous(
    survivals: Vec<SurvivalCurve<f64>>,
    recovery: f64,
    coupon: f64,
    notional: f64,
    effective_date: NaiveDate,
    maturity_date: NaiveDate,
  ) -> Self {
    let weight = 1.0 / survivals.len() as f64;
    Self::new(
      survivals
        .into_iter()
        .map(|survival| IndexName {
          weight,
          recovery,
          survival,
        })
        .collect(),
      coupon,
      notional,
      effective_date,
      maturity_date,
    )
  }

  fn single_name(&self, spread: f64, recovery: f64) -> CreditDefaultSwap<f64> {
    CreditDefaultSwap::vanilla(
      CdsPosition::Buyer,
      1.0,
      spread,
      recovery,
      self.effective_date,
      self.maturity_date,
      self.frequency,
      self.day_count,
    )
  }

  /// Name-by-name valuation aggregated by weight, from the buyer's side.
  ///
  /// The fair spread is NaN when the weighted risky annuity is not positive.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount: &(impl CurveProvider<f64> + ?Sized),
  ) -> IndexValuation {
    let (mut protection, mut premium, mut annuity) = (0.0, 0.0, 0.0);
    for name in &self.names {
      let cds = self.single_name(self.coupon, name.recovery);
      let v = cds.valuation(
        valuation_date,
        DayCountConvention::Actual365Fixed,
        discount,
        &name.survival,
      );
      protection += name.weight * v.protection_leg_npv;
      premium += name.weight * v.premium_leg_npv;
      annuity += name.weight * v.risky_annuity;
    }
    let scale = self.notional;
    IndexValuation {
      protection_leg_npv: scale * protection,
      premium_leg_npv: scale * premium,
      risky_annuity: scale * annuity,
      fair_spread: if annuity > 0.0 {
        protection / annuity
      } else {
        f64::NAN
      },
      net_npv: scale * (protection - premium),
    }
  }

  /// Index fair spread.
  pub fn fair_spread(
    &self,
    valuation_date: NaiveDate,
    discount: &(impl CurveProvider<f64> + ?Sized),
  ) -> f64 {
    self.valuation(valuation_date, discount).fair_spread
  }

  /// ISDA standard-model upfront the protection buyer pays for the index
  /// at `quoted_spread`: the flat hazard rate that prices a running CDS at
  /// the quoted spread to zero (with `recovery`, 40 % by convention) values
  /// the coupon shortfall `quoted − coupon` on its risky annuity.
  pub fn isda_upfront(
    &self,
    valuation_date: NaiveDate,
    discount: &(impl CurveProvider<f64> + ?Sized),
    quoted_spread: f64,
    recovery: f64,
  ) -> f64 {
    let hazard = self.flat_hazard_for(valuation_date, discount, quoted_spread, recovery);
    let survival = flat_survival(hazard);
    let cds = self.single_name(self.coupon, recovery);
    let v = cds.valuation(
      valuation_date,
      DayCountConvention::Actual365Fixed,
      discount,
      &survival,
    );
    self.notional * v.net_npv
  }

  /// Flat hazard rate reproducing `quoted_spread` as the fair spread of a
  /// single running CDS with `recovery`, by bisection.
  pub fn flat_hazard_for(
    &self,
    valuation_date: NaiveDate,
    discount: &(impl CurveProvider<f64> + ?Sized),
    quoted_spread: f64,
    recovery: f64,
  ) -> f64 {
    assert!(
      quoted_spread > 0.0 && recovery < 1.0,
      "quoted spread must be positive and recovery below one"
    );
    let cds = self.single_name(quoted_spread, recovery);
    let fair_minus_quote = |h: f64| {
      cds
        .valuation(
          valuation_date,
          DayCountConvention::Actual365Fixed,
          discount,
          &flat_survival(h),
        )
        .fair_spread
        - quoted_spread
    };
    let (mut lo, mut hi) = (1e-10_f64, 1.0_f64);
    while fair_minus_quote(hi) < 0.0 && hi < 1e4 {
      hi *= 2.0;
    }
    for _ in 0..200 {
      let mid = 0.5 * (lo + hi);
      if fair_minus_quote(mid) > 0.0 {
        hi = mid;
      } else {
        lo = mid;
      }
      if (hi - lo) <= 1e-14 * hi {
        break;
      }
    }
    0.5 * (lo + hi)
  }
}

/// Flat-hazard survival curve out to fifty years.
pub fn flat_survival(hazard: f64) -> SurvivalCurve<f64> {
  SurvivalCurve::from_hazard_rates(
    &Array1::from_vec(vec![1.0, 5.0, 10.0, 30.0, 50.0]),
    &Array1::from_vec(vec![hazard; 5]),
    HazardInterpolation::PiecewiseConstantHazard,
  )
}

#[cfg(test)]
mod tests;

//! # XVA
//!
//! $$
//! \mathrm{CVA} = \mathrm{LGD}\sum_i D(t_i)\,\mathrm{EPE}(t_i)\,\bigl[S(t_{i-1}) - S(t_i)\bigr],\qquad
//! \mathrm{FVA} = \sum_i D(t_i)\,\bigl[\mathrm{EPE}(t_i) - \mathrm{ENE}(t_i)\bigr]\,s_f\,\Delta t_i
//! $$
//!
//! Valuation adjustments built on a simulated exposure profile. The core is
//! model-agnostic: [`ExposureProfile`] reduces a matrix of mark-to-market
//! values (scenario paths × exposure dates) to expected positive and
//! negative exposure and a potential-future-exposure quantile, and the
//! adjustments integrate those profiles against survival, discount and
//! funding curves on the exposure grid (rectangle rule with the exposure at
//! the interval end). Bilateral CVA/DVA weight each period by the other
//! party's survival to the start of the period. The Hull–White swap
//! exposure engine in [`irs`] shows the profile being produced by the
//! crate's own simulation stack.
//!
//! References: Gregory, J. (2015), *The xVA Challenge: Counterparty Credit
//! Risk, Funding, Collateral and Capital*, 3rd ed., Wiley, Ch. 7, 14, 15;
//! Green, A. (2016), *XVA: Credit, Funding and Capital Valuation
//! Adjustments*, Wiley, Ch. 3, 9.

pub mod irs;

use ndarray::Array2;

use crate::credit::survival_curve::SurvivalCurve;
use crate::curves::DiscountCurve;

/// Exposure profile on a date grid.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureProfile {
  /// Exposure dates in years, increasing and positive.
  pub times: Vec<f64>,
  /// Expected positive exposure `E[max(V, 0)]` per date.
  pub epe: Vec<f64>,
  /// Expected negative exposure `E[max(−V, 0)]` per date.
  pub ene: Vec<f64>,
  /// Potential future exposure: the `quantile` of `max(V, 0)` per date.
  pub pfe: Vec<f64>,
  /// Quantile level of `pfe`.
  pub quantile: f64,
}

impl ExposureProfile {
  /// Reduces `mtm` (rows = scenario paths, columns = `times`) to the
  /// profile; `quantile` is the PFE level (e.g. 0.95).
  pub fn from_mtm(mtm: &Array2<f64>, times: Vec<f64>, quantile: f64) -> Self {
    assert_eq!(mtm.ncols(), times.len(), "one MtM column per exposure date");
    assert!(mtm.nrows() > 0, "the MtM matrix needs at least one path");
    assert!(
      (0.0..=1.0).contains(&quantile),
      "quantile must lie in [0, 1]"
    );
    assert!(
      times.windows(2).all(|w| w[0] < w[1]) && times.first().is_some_and(|t| *t > 0.0),
      "exposure dates must be positive and increasing"
    );
    let paths = mtm.nrows() as f64;
    let mut epe = Vec::with_capacity(times.len());
    let mut ene = Vec::with_capacity(times.len());
    let mut pfe = Vec::with_capacity(times.len());
    for col in mtm.columns() {
      let mut positive: Vec<f64> = col.iter().map(|v| v.max(0.0)).collect();
      epe.push(positive.iter().sum::<f64>() / paths);
      ene.push(col.iter().map(|v| (-v).max(0.0)).sum::<f64>() / paths);
      positive.sort_by(|a, b| a.partial_cmp(b).expect("finite exposures"));
      let rank = ((positive.len() - 1) as f64 * quantile).round() as usize;
      pfe.push(positive[rank]);
    }
    Self {
      times,
      epe,
      ene,
      pfe,
      quantile,
    }
  }

  /// Profile from precomputed expected exposures (e.g. analytic ones).
  ///
  /// The `quantile` field is NaN: expectations carry no PFE level, so `pfe`
  /// is a copy of `epe` rather than a quantile of the exposure distribution.
  pub fn from_expected(times: Vec<f64>, epe: Vec<f64>, ene: Vec<f64>) -> Self {
    assert!(
      times.len() == epe.len() && times.len() == ene.len(),
      "one EPE and ENE per date"
    );
    let pfe = epe.clone();
    Self {
      times,
      epe,
      ene,
      pfe,
      quantile: f64::NAN,
    }
  }

  /// Peak expected positive exposure.
  pub fn peak_epe(&self) -> f64 {
    self.epe.iter().copied().fold(0.0, f64::max)
  }

  /// Time-averaged expected positive exposure (the "expected exposure"
  /// used for CVA-style capital charges), trapezoidal in time from `0`.
  pub fn average_epe(&self) -> f64 {
    let mut acc = 0.0;
    let mut prev_t = 0.0;
    let mut prev_e = 0.0;
    for (&t, &e) in self.times.iter().zip(&self.epe) {
      acc += 0.5 * (prev_e + e) * (t - prev_t);
      prev_t = t;
      prev_e = e;
    }
    acc / prev_t
  }
}

/// Sums `exposure(t_i) · D(t_i) · [S(t_{i−1}) − S(t_i)] · weight(t_{i−1})` over
/// the grid.
fn credit_integral(
  profile: &ExposureProfile,
  exposure: &[f64],
  survival: &SurvivalCurve<f64>,
  discount: &DiscountCurve<f64>,
  weight: impl Fn(f64) -> f64,
) -> f64 {
  let mut prev_t = 0.0;
  let mut acc = 0.0;
  for (&t, &e) in profile.times.iter().zip(exposure) {
    let dpd = survival.survival_probability(prev_t) - survival.survival_probability(t);
    acc += discount.discount_factor(t) * e * dpd * weight(prev_t);
    prev_t = t;
  }
  acc
}

/// Unilateral CVA: `LGD Σ D(t_i) EPE(t_i) [S_C(t_{i−1}) − S_C(t_i)]`.
pub fn cva(
  profile: &ExposureProfile,
  counterparty: &SurvivalCurve<f64>,
  discount: &DiscountCurve<f64>,
  lgd: f64,
) -> f64 {
  lgd * credit_integral(profile, &profile.epe, counterparty, discount, |_| 1.0)
}

/// Unilateral DVA: `LGD_B Σ D(t_i) ENE(t_i) [S_B(t_{i−1}) − S_B(t_i)]` with the
/// bank's own survival curve.
pub fn dva(
  profile: &ExposureProfile,
  own: &SurvivalCurve<f64>,
  discount: &DiscountCurve<f64>,
  lgd_own: f64,
) -> f64 {
  lgd_own * credit_integral(profile, &profile.ene, own, discount, |_| 1.0)
}

/// Bilateral CVA: the counterparty's default in each period weighted by the
/// bank's survival to the start of the period.
pub fn bilateral_cva(
  profile: &ExposureProfile,
  counterparty: &SurvivalCurve<f64>,
  own: &SurvivalCurve<f64>,
  discount: &DiscountCurve<f64>,
  lgd: f64,
) -> f64 {
  lgd
    * credit_integral(profile, &profile.epe, counterparty, discount, |t| {
      own.survival_probability(t)
    })
}

/// Bilateral DVA: the bank's default weighted by the counterparty's survival
/// to the start of the period.
pub fn bilateral_dva(
  profile: &ExposureProfile,
  own: &SurvivalCurve<f64>,
  counterparty: &SurvivalCurve<f64>,
  discount: &DiscountCurve<f64>,
  lgd_own: f64,
) -> f64 {
  lgd_own
    * credit_integral(profile, &profile.ene, own, discount, |t| {
      counterparty.survival_probability(t)
    })
}

fn funding_integral(
  profile: &ExposureProfile,
  exposure: &[f64],
  discount: &DiscountCurve<f64>,
  spread: f64,
) -> f64 {
  let mut prev_t = 0.0;
  let mut acc = 0.0;
  for (&t, &e) in profile.times.iter().zip(exposure) {
    acc += discount.discount_factor(t) * e * spread * (t - prev_t);
    prev_t = t;
  }
  acc
}

/// Funding cost adjustment: `Σ D(t_i) EPE(t_i) s_f Δt_i` for a funding
/// spread `s_f` paid on the uncollateralised positive exposure.
pub fn fca(profile: &ExposureProfile, discount: &DiscountCurve<f64>, funding_spread: f64) -> f64 {
  funding_integral(profile, &profile.epe, discount, funding_spread)
}

/// Funding benefit adjustment: `Σ D(t_i) ENE(t_i) s_f Δt_i`.
pub fn fba(profile: &ExposureProfile, discount: &DiscountCurve<f64>, funding_spread: f64) -> f64 {
  funding_integral(profile, &profile.ene, discount, funding_spread)
}

/// Symmetric FVA, `FCA − FBA`.
pub fn fva(profile: &ExposureProfile, discount: &DiscountCurve<f64>, funding_spread: f64) -> f64 {
  fca(profile, discount, funding_spread) - fba(profile, discount, funding_spread)
}

#[cfg(test)]
mod tests;

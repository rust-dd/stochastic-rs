//! Credit Default Swap pricing under the ISDA-style hazard-rate framework.
//!
//! Reference: O'Kane & Turnbull, "Valuation of Credit Default Swaps", Lehman
//! Brothers Quantitative Credit Research Quarterly (2003).
//!
//! Reference: ISDA, "The ISDA CDS Standard Model" (2009),
//! <https://www.cdsmodel.com/>.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer, 2nd ed. (2006), Chapter 22.
//!
//! Given a hazard-driven survival curve $Q(t)$, a risk-free discount curve
//! $D(t)$, a constant recovery $R\in[0,1)$ and a running contractual spread
//! $s$, the CDS legs are
//!
//! $$
//! \mathrm{PV}_{\text{prem}} = N\,s\!\left[\sum_i \alpha_i\,Q(t_i)\,D(t_i)\,+\,\int_0^T D(u)(u-t_{i(u)-1})\bigl(-dQ(u)\bigr)\right],
//! $$
//!
//! $$
//! \mathrm{PV}_{\text{prot}}
//! = N(1-R)\int_0^T D(u)\,\bigl(-dQ(u)\bigr).
//! $$
//!
//! The second term in the premium PV is the accrual on default.  The
//! protection integral and the accrual piece are approximated on a daily
//! grid (ISDA standard), splitting each coupon period by the trapezoidal rule
//! on $D(u)Q(u)$ times the finite increment $Q(u_k)-Q(u_{k+1})$.

use std::fmt::Display;

use chrono::NaiveDate;

use super::survival_curve::SurvivalCurve;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Frequency;
use crate::quant::calendar::Schedule;
use crate::quant::cashflows::CurveProvider;
use crate::traits::FloatExt;

/// Pay / receive direction of a CDS contract.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CdsPosition {
  /// Protection buyer pays the running premium and receives $N(1-R)$ on
  /// default.  Net PV is $\mathrm{PV}_{\text{prot}}-\mathrm{PV}_{\text{prem}}$.
  #[default]
  Buyer,
  /// Protection seller — opposite sign of the buyer.
  Seller,
}

impl Display for CdsPosition {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Buyer => write!(f, "Protection buyer"),
      Self::Seller => write!(f, "Protection seller"),
    }
  }
}

/// Full CDS valuation breakdown.
#[derive(Debug, Clone)]
pub struct CdsValuation<T: FloatExt> {
  /// Discounted expected protection payment (present value of $N(1-R)$ on
  /// default).
  pub protection_leg_npv: T,
  /// Discounted expected fee payments at the running spread $s$.
  pub premium_leg_coupons_npv: T,
  /// Discounted expected fee accrual from the last coupon to the default time.
  pub premium_leg_accrual_npv: T,
  /// Total premium-leg PV (coupons + accrual on default).
  pub premium_leg_npv: T,
  /// Present value of a 1 bp running premium (aka risky PV01).
  pub risky_annuity: T,
  /// Fair running spread $s^\star=\mathrm{PV}_{\text{prot}}/\mathrm{PV}_{\text{risky-annuity}}$.
  pub fair_spread: T,
  /// Net signed PV consistent with the contract direction.
  pub net_npv: T,
  /// Direction used for the sign convention.
  pub direction: CdsPosition,
}

/// Running-spread Credit Default Swap contract.
#[derive(Debug, Clone)]
pub struct CreditDefaultSwap<T: FloatExt> {
  /// Protection buyer / seller flag.
  pub direction: CdsPosition,
  /// Contract notional.
  pub notional: T,
  /// Running contractual spread (annualised, decimal form; e.g. 0.01 = 100 bps).
  pub spread: T,
  /// Deterministic recovery rate $R$ used on default.
  pub recovery_rate: T,
  /// Premium accrual schedule (coupon start / end / payment dates).
  pub premium_schedule: Schedule,
  /// Day count convention for the premium accrual.
  pub day_count: DayCountConvention,
  /// Contract protection effective date (inclusive).
  pub effective_date: NaiveDate,
  /// Contract maturity (protection end, inclusive).
  pub maturity_date: NaiveDate,
  /// Whether accrual on default is paid (ISDA standard is `true`).
  pub accrual_on_default: bool,
  /// Integration step (in days) used on the fine grid for the protection and
  /// accrual integrals. The ISDA standard uses 1.
  pub integration_step_days: i64,
}

impl<T: FloatExt> CreditDefaultSwap<T> {
  /// Build a running-spread CDS using the provided premium schedule.
  pub fn new(
    direction: CdsPosition,
    notional: T,
    spread: T,
    recovery_rate: T,
    premium_schedule: Schedule,
    day_count: DayCountConvention,
    effective_date: NaiveDate,
    maturity_date: NaiveDate,
  ) -> Self {
    assert!(
      recovery_rate >= T::zero() && recovery_rate < T::one(),
      "recovery_rate must lie in [0, 1)"
    );
    assert!(
      premium_schedule.adjusted_dates.len() >= 2,
      "premium schedule must contain at least two dates"
    );
    assert!(
      effective_date < maturity_date,
      "effective_date must precede maturity_date"
    );
    Self {
      direction,
      notional,
      spread,
      recovery_rate,
      premium_schedule,
      day_count,
      effective_date,
      maturity_date,
      accrual_on_default: true,
      integration_step_days: 1,
    }
  }

  /// Build a vanilla running CDS with a quarterly schedule (standard post-2009).
  pub fn vanilla(
    direction: CdsPosition,
    notional: T,
    spread: T,
    recovery_rate: T,
    effective_date: NaiveDate,
    maturity_date: NaiveDate,
    frequency: Frequency,
    day_count: DayCountConvention,
  ) -> Self {
    let schedule = crate::quant::calendar::ScheduleBuilder::new(effective_date, maturity_date)
      .frequency(frequency)
      .build();
    Self::new(
      direction,
      notional,
      spread,
      recovery_rate,
      schedule,
      day_count,
      effective_date,
      maturity_date,
    )
  }

  /// Override the protection-integral granularity. Smaller values approach the
  /// ISDA daily integration; larger values trade accuracy for speed.
  pub fn with_integration_step(mut self, days: i64) -> Self {
    assert!(days >= 1, "integration step must be at least 1 day");
    self.integration_step_days = days;
    self
  }

  /// Toggle accrual on default.
  pub fn with_accrual_on_default(mut self, enabled: bool) -> Self {
    self.accrual_on_default = enabled;
    self
  }

  /// Full valuation breakdown.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    discount: &(impl CurveProvider<T> + ?Sized),
    survival: &SurvivalCurve<T>,
  ) -> CdsValuation<T> {
    let discount = discount.discount_curve();

    let mut premium_coupons_unit = T::zero();
    let mut premium_accrual_unit = T::zero();
    let mut protection_unit = T::zero();

    for window in self.premium_schedule.adjusted_dates.windows(2) {
      let coupon_start = window[0];
      let coupon_end = window[1];
      if coupon_end <= valuation_date {
        continue;
      }
      let alpha: T = self.day_count.year_fraction(coupon_start, coupon_end);
      let payment_tau: T = tau_year_fraction(discount_day_count, valuation_date, coupon_end);
      let df_end = discount.discount_factor(payment_tau);
      let q_end = survival.survival_probability(tau_year_fraction(
        DayCountConvention::Actual365Fixed,
        valuation_date,
        coupon_end,
      ));

      premium_coupons_unit += alpha * df_end * q_end;

      let (accrual_unit, protection_period) = self.integrate_period(
        valuation_date,
        discount_day_count,
        coupon_start.max(self.effective_date).max(valuation_date),
        coupon_end.min(self.maturity_date),
        discount,
        survival,
      );

      if self.accrual_on_default {
        premium_accrual_unit += accrual_unit;
      }
      protection_unit += protection_period;
    }

    let premium_leg_coupons_npv = self.notional * self.spread * premium_coupons_unit;
    let premium_leg_accrual_npv = self.notional * self.spread * premium_accrual_unit;
    let premium_leg_npv = premium_leg_coupons_npv + premium_leg_accrual_npv;
    let protection_leg_npv = self.notional * (T::one() - self.recovery_rate) * protection_unit;

    let risky_annuity = self.notional * (premium_coupons_unit + premium_accrual_unit);
    let fair_spread = if risky_annuity.abs() <= T::min_positive_val() {
      T::zero()
    } else {
      protection_leg_npv / risky_annuity
    };

    let net_npv = match self.direction {
      CdsPosition::Buyer => protection_leg_npv - premium_leg_npv,
      CdsPosition::Seller => premium_leg_npv - protection_leg_npv,
    };

    CdsValuation {
      protection_leg_npv,
      premium_leg_coupons_npv,
      premium_leg_accrual_npv,
      premium_leg_npv,
      risky_annuity,
      fair_spread,
      net_npv,
      direction: self.direction,
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    discount: &(impl CurveProvider<T> + ?Sized),
    survival: &SurvivalCurve<T>,
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, discount, survival)
      .net_npv
  }

  /// Fair (par) running spread consistent with the provided curves.
  pub fn fair_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    discount: &(impl CurveProvider<T> + ?Sized),
    survival: &SurvivalCurve<T>,
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, discount, survival)
      .fair_spread
  }

  /// Risky PV01 — PV of a 1-unit running spread (ISDA annuity).
  pub fn risky_annuity(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    discount: &(impl CurveProvider<T> + ?Sized),
    survival: &SurvivalCurve<T>,
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, discount, survival)
      .risky_annuity
  }

  /// Discrete trapezoidal integration of the accrual-on-default and protection
  /// integrals over a single coupon period.
  fn integrate_period(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    period_start: NaiveDate,
    period_end: NaiveDate,
    discount: &crate::quant::curves::DiscountCurve<T>,
    survival: &SurvivalCurve<T>,
  ) -> (T, T) {
    if period_end <= period_start {
      return (T::zero(), T::zero());
    }
    let total_days = (period_end - period_start).num_days();
    if total_days <= 0 {
      return (T::zero(), T::zero());
    }
    let step = self.integration_step_days.min(total_days).max(1);

    let mut cursor = period_start;
    let mut df_prev = discount.discount_factor(tau_year_fraction(
      discount_day_count,
      valuation_date,
      cursor,
    ));
    let mut q_prev = survival.survival_probability(tau_year_fraction(
      DayCountConvention::Actual365Fixed,
      valuation_date,
      cursor,
    ));

    let mut accrual_unit = T::zero();
    let mut protection_unit = T::zero();
    let half = T::from_f64_fast(0.5);

    while cursor < period_end {
      let delta_days = step.min((period_end - cursor).num_days());
      let next = cursor + chrono::Duration::days(delta_days);

      let df_next = discount.discount_factor(tau_year_fraction(
        discount_day_count,
        valuation_date,
        next,
      ));
      let q_next = survival.survival_probability(tau_year_fraction(
        DayCountConvention::Actual365Fixed,
        valuation_date,
        next,
      ));

      let df_mid = half * (df_prev + df_next);
      let dq = q_prev - q_next;

      // Mid-period accrual fraction α(period_start, u_mid).
      let alpha_to_cursor: T = self.day_count.year_fraction(period_start, cursor);
      let alpha_slice: T = self.day_count.year_fraction(cursor, next);
      let accrual_factor_mid = alpha_to_cursor + half * alpha_slice;

      accrual_unit += df_mid * dq * accrual_factor_mid;
      protection_unit += df_mid * dq;

      cursor = next;
      df_prev = df_next;
      q_prev = q_next;
    }

    (accrual_unit, protection_unit)
  }
}

fn tau_year_fraction<T: FloatExt>(
  dcc: DayCountConvention,
  valuation_date: NaiveDate,
  target: NaiveDate,
) -> T {
  if target <= valuation_date {
    T::zero()
  } else {
    dcc.year_fraction(valuation_date, target)
  }
}

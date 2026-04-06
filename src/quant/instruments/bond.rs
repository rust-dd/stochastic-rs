//! Fixed-income bond analytics.
//!
//! $$
//! P=\sum_{i=1}^{n}\frac{CF_i}{(1+y/m)^{m t_i}},\qquad
//! D_{\mathrm{Mac}}=\frac{\sum_i t_i\,PV_i}{P}
//! $$
//!
//! Reference: Fabozzi, "Fixed Income Mathematics", 4th ed. (2015).
//!
//! Reference: Ivanovski, Stojanovski & Ivanovska, "Interest Rate Risk of Bond
//! Prices on Macedonian Stock Exchange - Empirical Test of the Duration,
//! Modified Duration and Convexity and Bonds Valuation", arXiv:1206.6998 (2012).

use chrono::NaiveDate;
use ndarray::Array1;
use roots::SimpleConvergency;
use roots::find_root_brent;

use super::super::calendar::DayCountConvention;
use super::super::calendar::Frequency;
use super::super::calendar::Schedule;
use super::super::cashflows::AccrualPeriod;
use super::super::cashflows::Cashflow;
use super::super::cashflows::CashflowPricer;
use super::super::cashflows::CurveProvider;
use super::super::cashflows::FloatingIndex;
use super::super::cashflows::Leg;
use super::super::cashflows::NotionalSchedule;
use super::super::cashflows::SimpleCashflow;
use super::super::curves::Compounding;
use crate::traits::FloatExt;

/// Dirty / clean bond price decomposition.
#[derive(Debug, Clone)]
pub struct BondPrice<T: FloatExt> {
  /// Present value including accrued interest.
  pub dirty_price: T,
  /// Accrued interest at settlement / valuation.
  pub accrued_interest: T,
  /// Present value net of accrued interest.
  pub clean_price: T,
}

/// Standard fixed-rate bond analytics.
#[derive(Debug, Clone)]
pub struct BondAnalytics<T: FloatExt> {
  /// Dirty price including accrued interest.
  pub dirty_price: T,
  /// Clean price excluding accrued interest.
  pub clean_price: T,
  /// Accrued interest at settlement / valuation.
  pub accrued_interest: T,
  /// Yield-to-maturity under the supplied compounding convention.
  pub yield_to_maturity: T,
  /// Macaulay duration in years.
  pub macaulay_duration: T,
  /// Modified duration.
  pub modified_duration: T,
  /// Convexity.
  pub convexity: T,
}

/// Bullet fixed-rate bond backed by a deterministic coupon leg.
#[derive(Debug, Clone)]
pub struct FixedRateBond<T: FloatExt> {
  /// Face amount redeemed at maturity.
  pub face_value: T,
  /// Annual coupon rate.
  pub coupon_rate: T,
  /// Coupon frequency used for the standard market yield convention.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  leg: Leg<T>,
}

impl<T: FloatExt> FixedRateBond<T> {
  /// Build a bullet fixed-rate bond from a payment schedule.
  pub fn new(
    schedule: &Schedule,
    face_value: T,
    coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      schedule.adjusted_dates.len() >= 2,
      "bond schedule must contain at least two dates"
    );
    let maturity = *schedule.adjusted_dates.last().unwrap();
    let leg = Leg::fixed_rate(
      schedule,
      NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, face_value),
      coupon_rate,
      coupon_day_count,
    )
    .with_redemption(maturity, face_value);

    Self {
      face_value,
      coupon_rate,
      coupon_frequency,
      coupon_day_count,
      leg,
    }
  }

  /// Borrow the underlying deterministic cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Bond maturity date.
  pub fn maturity_date(&self) -> NaiveDate {
    self
      .leg
      .cashflows()
      .last()
      .map(Cashflow::payment_date)
      .unwrap()
  }

  /// Standard YTM compounding convention for the coupon frequency.
  pub fn standard_yield_compounding(&self) -> Compounding {
    Compounding::Periodic(self.coupon_frequency.periods_per_year())
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    price_deterministic_leg_from_curve(&self.leg, valuation_date, discount_day_count, curves)
  }

  /// Accrued interest at settlement.
  pub fn accrued_interest(&self, settlement_date: NaiveDate) -> T {
    accrued_interest_for_deterministic_leg(&self.leg, settlement_date)
  }

  /// Dirty price under a yield-to-maturity assumption.
  pub fn dirty_price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    dirty_price_from_yield_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Clean price under a yield-to-maturity assumption.
  pub fn clean_price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    self.dirty_price_from_yield(
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ) - self.accrued_interest(settlement_date)
  }

  /// Solve the yield-to-maturity implied by a dirty price.
  pub fn yield_to_maturity_from_dirty_price(
    &self,
    settlement_date: NaiveDate,
    dirty_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    yield_to_maturity_from_dirty_price_for_leg(
      &self.leg,
      settlement_date,
      dirty_price,
      yield_day_count,
      compounding,
    )
  }

  /// Solve the yield-to-maturity implied by a clean price.
  pub fn yield_to_maturity_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    clean_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    self.yield_to_maturity_from_dirty_price(
      settlement_date,
      clean_price + self.accrued_interest(settlement_date),
      yield_day_count,
      compounding,
    )
  }

  /// Macaulay duration in years.
  pub fn macaulay_duration(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    macaulay_duration_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Modified duration computed by a finite-difference yield bump.
  pub fn modified_duration(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    modified_duration_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Convexity computed by a finite-difference yield bump.
  pub fn convexity(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    convexity_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Full analytics implied by the current curve stack.
  pub fn analytics_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> BondAnalytics<T> {
    let price = self.price_from_curve(valuation_date, discount_day_count, curves);
    bond_analytics_from_dirty_price(
      &self.leg,
      valuation_date,
      price.dirty_price,
      price.accrued_interest,
      yield_day_count,
      compounding,
    )
  }

  /// Full analytics implied by a clean market price.
  pub fn analytics_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    clean_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> BondAnalytics<T> {
    let accrued_interest = self.accrued_interest(settlement_date);
    bond_analytics_from_dirty_price(
      &self.leg,
      settlement_date,
      clean_price + accrued_interest,
      accrued_interest,
      yield_day_count,
      compounding,
    )
  }

  /// Price using a constant continuously-compounded Z-spread over the discount curve.
  pub fn dirty_price_from_z_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    z_spread: T,
  ) -> T {
    price_with_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      discount_day_count,
      curves,
      z_spread,
    )
  }

  /// Solve the Z-spread implied by a dirty market price.
  pub fn z_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    solve_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      market_dirty_price,
      discount_day_count,
      curves,
      T::zero(),
    )
  }

  /// Solve the Z-spread implied by a clean market price.
  pub fn z_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self.z_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
    )
  }

  /// Price using a constant OAS and an externally supplied embedded option value.
  pub fn dirty_price_from_option_adjusted_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    oas: T,
    embedded_option_value: T,
  ) -> T {
    self.dirty_price_from_z_spread(valuation_date, discount_day_count, curves, oas)
      - embedded_option_value
  }

  /// Solve the OAS implied by a dirty market price and an embedded option value.
  pub fn option_adjusted_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    embedded_option_value: T,
  ) -> T {
    solve_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      market_dirty_price,
      discount_day_count,
      curves,
      embedded_option_value,
    )
  }

  /// Solve the OAS implied by a clean market price and an embedded option value.
  pub fn option_adjusted_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    embedded_option_value: T,
  ) -> T {
    self.option_adjusted_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
      embedded_option_value,
    )
  }

  /// Approximate par asset-swap spread for the remaining bond life.
  pub fn asset_swap_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    let annuity = fixed_leg_spread_annuity(&self.leg, valuation_date, discount_day_count, curves);
    if annuity.abs() <= T::min_positive_val() {
      return T::zero();
    }

    let maturity_tau = discount_day_count.year_fraction(valuation_date, self.maturity_date());
    let fair_swap_rate = (self.face_value
      - self.face_value * curves.discount_curve().discount_factor(maturity_tau))
      / annuity;
    self.coupon_rate - fair_swap_rate + (self.face_value - market_dirty_price) / annuity
  }

  /// Approximate par asset-swap spread from a clean price.
  pub fn asset_swap_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self.asset_swap_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
    )
  }
}

/// Floating-rate note backed by a floating coupon leg plus redemption.
#[derive(Debug, Clone)]
pub struct FloatingRateBond<T: FloatExt> {
  /// Face amount redeemed at maturity.
  pub face_value: T,
  /// Floating index.
  pub index: FloatingIndex<T>,
  /// Quoted spread over the index.
  pub spread: T,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  leg: Leg<T>,
}

impl<T: FloatExt> FloatingRateBond<T> {
  /// Build a floating-rate note from a schedule.
  pub fn new(
    schedule: &Schedule,
    face_value: T,
    index: FloatingIndex<T>,
    spread: T,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      schedule.adjusted_dates.len() >= 2,
      "bond schedule must contain at least two dates"
    );
    let maturity = *schedule.adjusted_dates.last().unwrap();
    let leg = Leg::floating_rate(
      schedule,
      NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, face_value),
      index.clone(),
      spread,
      coupon_day_count,
    )
    .with_redemption(maturity, face_value);

    Self {
      face_value,
      index,
      spread,
      coupon_day_count,
      leg,
    }
  }

  /// Borrow the underlying cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    let summary =
      CashflowPricer::new(valuation_date, discount_day_count).summarize_leg(&self.leg, curves);
    BondPrice {
      dirty_price: summary.dirty_npv,
      accrued_interest: summary.accrued_interest,
      clean_price: summary.clean_npv,
    }
  }

  /// Accrued interest under the projected or observed fixing state.
  pub fn accrued_interest(
    &self,
    valuation_date: NaiveDate,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    CashflowPricer::new(valuation_date, self.coupon_day_count)
      .leg_accrued_interest(&self.leg, curves)
  }
}

/// Fixed-rate amortizing bond with explicit outstanding-notional schedule.
#[derive(Debug, Clone)]
pub struct AmortizingFixedRateBond<T: FloatExt> {
  /// Initial outstanding notional.
  pub initial_notional: T,
  /// Annual coupon rate.
  pub coupon_rate: T,
  /// Coupon frequency used for the standard market yield convention.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  notionals: NotionalSchedule<T>,
  leg: Leg<T>,
}

impl<T: FloatExt> AmortizingFixedRateBond<T> {
  /// Build an amortizing fixed-rate bond from an outstanding-notional schedule.
  pub fn new(
    schedule: &Schedule,
    notionals: NotionalSchedule<T>,
    coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    notionals.validate(periods);
    assert!(
      periods > 0,
      "bond schedule must contain at least one period"
    );

    let mut leg = Leg::fixed_rate(schedule, notionals.clone(), coupon_rate, coupon_day_count);
    for (idx, window) in schedule.adjusted_dates.windows(2).enumerate() {
      let current = notionals.notionals()[idx];
      let next = if idx + 1 < notionals.len() {
        notionals.notionals()[idx + 1]
      } else {
        T::zero()
      };
      assert!(
        next <= current,
        "amortizing bond requires a non-increasing outstanding-notional schedule"
      );
      let principal_payment = current - next;
      if principal_payment.abs() > T::min_positive_val() {
        leg.push(Cashflow::Simple(SimpleCashflow {
          payment_date: window[1],
          amount: principal_payment,
        }));
      }
    }

    Self {
      initial_notional: notionals.notionals()[0],
      coupon_rate,
      coupon_frequency,
      coupon_day_count,
      notionals,
      leg,
    }
  }

  /// Borrow the outstanding-notional schedule.
  pub fn notionals(&self) -> &NotionalSchedule<T> {
    &self.notionals
  }

  /// Borrow the underlying deterministic cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Standard YTM compounding convention for the coupon frequency.
  pub fn standard_yield_compounding(&self) -> Compounding {
    Compounding::Periodic(self.coupon_frequency.periods_per_year())
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    price_deterministic_leg_from_curve(&self.leg, valuation_date, discount_day_count, curves)
  }

  /// Accrued interest at settlement.
  pub fn accrued_interest(&self, settlement_date: NaiveDate) -> T {
    accrued_interest_for_deterministic_leg(&self.leg, settlement_date)
  }

  /// Full analytics implied by the current curve stack.
  pub fn analytics_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> BondAnalytics<T> {
    let price = self.price_from_curve(valuation_date, discount_day_count, curves);
    bond_analytics_from_dirty_price(
      &self.leg,
      valuation_date,
      price.dirty_price,
      price.accrued_interest,
      yield_day_count,
      compounding,
    )
  }

  /// Price using a constant continuously-compounded Z-spread over the discount curve.
  pub fn dirty_price_from_z_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    z_spread: T,
  ) -> T {
    price_with_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      discount_day_count,
      curves,
      z_spread,
    )
  }

  /// Solve the Z-spread implied by a dirty market price.
  pub fn z_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    solve_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      market_dirty_price,
      discount_day_count,
      curves,
      T::zero(),
    )
  }
}

/// Deterministic inflation-linked bond using projected index ratios.
#[derive(Debug, Clone)]
pub struct InflationLinkedBond<T: FloatExt> {
  /// Real face amount redeemed at maturity before indexation.
  pub real_face_value: T,
  /// Real coupon rate.
  pub real_coupon_rate: T,
  /// Coupon frequency.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  /// Index ratio at the start of the first coupon period.
  pub base_index_ratio: T,
  index_ratios: Array1<T>,
  periods: Vec<AccrualPeriod<T>>,
  leg: Leg<T>,
}

impl<T: FloatExt> InflationLinkedBond<T> {
  /// Build an inflation-linked bond from projected index ratios for each payment date.
  pub fn new(
    schedule: &Schedule,
    real_face_value: T,
    real_coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
    base_index_ratio: T,
    index_ratios: Array1<T>,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    assert!(
      periods > 0,
      "bond schedule must contain at least one period"
    );
    assert_eq!(
      index_ratios.len(),
      periods,
      "expected {periods} projected index ratios, got {}",
      index_ratios.len()
    );

    let accrual_periods: Vec<_> = schedule
      .adjusted_dates
      .windows(2)
      .map(|window| AccrualPeriod::new(window[0], window[1], window[1], coupon_day_count))
      .collect();
    let mut cashflows = Vec::with_capacity(periods + 1);
    for (idx, period) in accrual_periods.iter().enumerate() {
      let ratio = index_ratios[idx];
      let coupon_amount = real_face_value * real_coupon_rate * period.accrual_factor * ratio;
      cashflows.push(Cashflow::Simple(SimpleCashflow {
        payment_date: period.payment_date,
        amount: coupon_amount,
      }));
    }
    let maturity = accrual_periods.last().unwrap().payment_date;
    cashflows.push(Cashflow::Simple(SimpleCashflow {
      payment_date: maturity,
      amount: real_face_value * index_ratios[index_ratios.len() - 1],
    }));

    Self {
      real_face_value,
      real_coupon_rate,
      coupon_frequency,
      coupon_day_count,
      base_index_ratio,
      index_ratios,
      periods: accrual_periods,
      leg: Leg::from_cashflows(cashflows),
    }
  }

  /// Borrow the projected payment-date index ratios.
  pub fn index_ratios(&self) -> &Array1<T> {
    &self.index_ratios
  }

  /// Borrow the projected deterministic cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let dirty_price = pricer.leg_npv(&self.leg, curves);
    let accrued_interest = self.accrued_interest(valuation_date);
    BondPrice {
      dirty_price,
      accrued_interest,
      clean_price: dirty_price - accrued_interest,
    }
  }

  /// Accrued real coupon interest indexed by the interpolated reference ratio.
  pub fn accrued_interest(&self, as_of: NaiveDate) -> T {
    for (idx, period) in self.periods.iter().enumerate() {
      if as_of > period.accrual_start && as_of < period.accrual_end {
        let full_factor = period.accrual_factor;
        if full_factor.abs() <= T::min_positive_val() {
          return T::zero();
        }
        let accrued_factor = period.accrued_factor(as_of);
        let elapsed_weight = accrued_factor / full_factor;
        let start_ratio = if idx == 0 {
          self.base_index_ratio
        } else {
          self.index_ratios[idx - 1]
        };
        let end_ratio = self.index_ratios[idx];
        let interpolated_ratio = start_ratio + (end_ratio - start_ratio) * elapsed_weight;
        return self.real_face_value * self.real_coupon_rate * accrued_factor * interpolated_ratio;
      }
    }
    T::zero()
  }
}

/// Zero-coupon bond priced directly from the discount curve.
#[derive(Debug, Clone)]
pub struct ZeroCouponBond<T: FloatExt> {
  /// Face amount paid at maturity.
  pub face_value: T,
  /// Maturity date.
  pub maturity_date: NaiveDate,
}

impl<T: FloatExt> ZeroCouponBond<T> {
  /// Create a zero-coupon bond.
  pub fn new(face_value: T, maturity_date: NaiveDate) -> Self {
    Self {
      face_value,
      maturity_date,
    }
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    if self.maturity_date < valuation_date {
      return T::zero();
    }
    let tau = discount_day_count.year_fraction(valuation_date, self.maturity_date);
    self.face_value * curves.discount_curve().discount_factor(tau)
  }

  /// Present value from a quoted yield.
  pub fn price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    if self.maturity_date < settlement_date {
      return T::zero();
    }
    let tau = yield_day_count.year_fraction(settlement_date, self.maturity_date);
    self.face_value * compounding.discount_factor(yield_to_maturity, tau)
  }

  /// Yield-to-maturity implied by a price.
  pub fn yield_to_maturity(
    &self,
    settlement_date: NaiveDate,
    price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    if price <= T::zero() || self.maturity_date <= settlement_date {
      return T::zero();
    }
    let tau = yield_day_count.year_fraction(settlement_date, self.maturity_date);
    let discount_factor = price / self.face_value;
    compounding.zero_rate(discount_factor, tau)
  }
}

fn bond_analytics_from_dirty_price<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  dirty_price: T,
  accrued_interest: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> BondAnalytics<T> {
  let yield_to_maturity = yield_to_maturity_from_dirty_price_for_leg(
    leg,
    settlement_date,
    dirty_price,
    yield_day_count,
    compounding,
  );
  BondAnalytics {
    dirty_price,
    clean_price: dirty_price - accrued_interest,
    accrued_interest,
    yield_to_maturity,
    macaulay_duration: macaulay_duration_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
    modified_duration: modified_duration_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
    convexity: convexity_for_leg(
      leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ),
  }
}

fn price_deterministic_leg_from_curve<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> BondPrice<T> {
  let summary = CashflowPricer::new(valuation_date, discount_day_count).summarize_leg(leg, curves);
  BondPrice {
    dirty_price: summary.dirty_npv,
    accrued_interest: summary.accrued_interest,
    clean_price: summary.clean_npv,
  }
}

fn accrued_interest_for_deterministic_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Fixed(coupon) => coupon.accrued_interest(settlement_date),
      Cashflow::Simple(_) => T::zero(),
      Cashflow::Floating(_) | Cashflow::Cms(_) => {
        unreachable!("deterministic bond legs must not contain stochastic coupons")
      }
    })
    .fold(T::zero(), |acc, value| acc + value)
}

fn dirty_price_from_yield_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  leg
    .cashflows()
    .iter()
    .filter(|cashflow| cashflow.payment_date() >= settlement_date)
    .map(|cashflow| {
      let tau = yield_day_count.year_fraction(settlement_date, cashflow.payment_date());
      let amount = deterministic_cashflow_amount(cashflow);
      amount * compounding.discount_factor(yield_to_maturity, tau)
    })
    .fold(T::zero(), |acc, value| acc + value)
}

fn yield_to_maturity_from_dirty_price_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  dirty_price: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  if dirty_price <= T::zero() {
    return T::zero();
  }

  let target = dirty_price.to_f64().unwrap();
  let min_yield = min_yield_for_leg(leg, settlement_date, yield_day_count, compounding);
  let mut low = min_yield + 1e-8;
  let mut high = 0.10f64;
  let f = |y: f64| {
    dirty_price_from_yield_for_leg(
      leg,
      settlement_date,
      T::from_f64_fast(y),
      yield_day_count,
      compounding,
    )
    .to_f64()
    .unwrap()
      - target
  };

  let mut f_low = f(low);
  let mut expand_low = 0;
  while f_low < 0.0 && matches!(compounding, Compounding::Continuous) && expand_low < 16 {
    low *= 2.0;
    f_low = f(low);
    expand_low += 1;
  }

  let mut f_high = f(high);
  let mut expand_high = 0;
  while f_high > 0.0 && expand_high < 32 {
    high = high * 2.0 + 0.05;
    f_high = f(high);
    expand_high += 1;
  }

  if f_low.abs() < 1e-12 {
    return T::from_f64_fast(low);
  }
  if f_high.abs() < 1e-12 {
    return T::from_f64_fast(high);
  }
  if f_low * f_high > 0.0 {
    panic!(
      "failed to bracket yield-to-maturity for dirty price {} between {low} and {high}",
      dirty_price.to_f64().unwrap()
    );
  }

  let mut convergency = SimpleConvergency {
    eps: 1e-12,
    max_iter: 100,
  };
  let root = find_root_brent(low, high, f, &mut convergency)
    .expect("failed to solve yield-to-maturity with Brent root finder");
  T::from_f64_fast(root)
}

fn macaulay_duration_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let dirty_price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if dirty_price <= T::zero() {
    return T::zero();
  }

  let weighted = leg
    .cashflows()
    .iter()
    .filter(|cashflow| cashflow.payment_date() >= settlement_date)
    .map(|cashflow| {
      let tau = yield_day_count.year_fraction(settlement_date, cashflow.payment_date());
      let amount = deterministic_cashflow_amount(cashflow);
      let pv = amount * compounding.discount_factor(yield_to_maturity, tau);
      tau * pv
    })
    .fold(T::zero(), |acc, value| acc + value);

  weighted / dirty_price
}

fn modified_duration_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if price <= T::zero() {
    return T::zero();
  }

  let y = yield_to_maturity.to_f64().unwrap();
  let h = (1e-5f64).max(1e-5 * (1.0 + y.abs()));
  let down =
    (y - h).max(min_yield_for_leg(leg, settlement_date, yield_day_count, compounding) + 1e-8);
  let up = y + h;
  let p_down = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(down),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_up = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(up),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p = price.to_f64().unwrap();
  T::from_f64_fast(-(p_up - p_down) / ((up - down) * p))
}

fn convexity_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_to_maturity: T,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> T {
  let price = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    yield_to_maturity,
    yield_day_count,
    compounding,
  );
  if price <= T::zero() {
    return T::zero();
  }

  let y = yield_to_maturity.to_f64().unwrap();
  let h = (1e-4f64).max(1e-4 * (1.0 + y.abs()));
  let down =
    (y - h).max(min_yield_for_leg(leg, settlement_date, yield_day_count, compounding) + 1e-8);
  let up = y + h;
  let mid = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(y),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_down = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(down),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p_up = dirty_price_from_yield_for_leg(
    leg,
    settlement_date,
    T::from_f64_fast(up),
    yield_day_count,
    compounding,
  )
  .to_f64()
  .unwrap();
  let p = price.to_f64().unwrap();
  let h_eff = (up - down) / 2.0;
  T::from_f64_fast((p_up - 2.0 * mid + p_down) / (h_eff * h_eff * p))
}

fn min_yield_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  settlement_date: NaiveDate,
  yield_day_count: DayCountConvention,
  compounding: Compounding,
) -> f64 {
  match compounding {
    Compounding::Continuous => -1.0,
    Compounding::Periodic(periods_per_year) => -(periods_per_year as f64) + 1e-6,
    Compounding::Simple => {
      let max_tau = leg
        .cashflows()
        .iter()
        .filter(|cashflow| cashflow.payment_date() >= settlement_date)
        .map(|cashflow| {
          yield_day_count
            .year_fraction::<T>(settlement_date, cashflow.payment_date())
            .to_f64()
            .unwrap()
        })
        .fold(0.0f64, f64::max);
      if max_tau <= 0.0 {
        -1.0
      } else {
        -1.0 / max_tau + 1e-6
      }
    }
  }
}

fn price_with_constant_spread_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
  spread: T,
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| {
      let payment_date = cashflow.payment_date();
      if payment_date < valuation_date {
        return T::zero();
      }
      let tau = discount_day_count.year_fraction(valuation_date, payment_date);
      let df = curves.discount_curve().discount_factor(tau) * (-spread * tau).exp();
      df * deterministic_cashflow_amount(cashflow)
    })
    .fold(T::zero(), |acc, value| acc + value)
}

fn solve_constant_spread_for_leg<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  market_dirty_price: T,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
  option_cost: T,
) -> T {
  if market_dirty_price <= T::zero() {
    return T::zero();
  }

  let target = market_dirty_price.to_f64().unwrap();
  let f = |spread: f64| {
    let dirty = price_with_constant_spread_for_leg(
      leg,
      valuation_date,
      discount_day_count,
      curves,
      T::from_f64_fast(spread),
    )
    .to_f64()
    .unwrap();
    dirty - option_cost.to_f64().unwrap() - target
  };

  if f(0.0).abs() < 1e-12 {
    return T::zero();
  }

  let mut low = -0.05f64;
  let mut high = 0.05f64;
  let mut f_low = f(low);
  let mut f_high = f(high);
  let mut expand = 0;
  while f_low * f_high > 0.0 && expand < 40 {
    if f_low < 0.0 {
      low = low * 2.0 - 0.01;
      f_low = f(low);
    } else {
      high = high * 2.0 + 0.01;
      f_high = f(high);
    }
    expand += 1;
  }

  if f_low * f_high > 0.0 {
    panic!(
      "failed to bracket constant spread for dirty price {} between {low} and {high}",
      market_dirty_price.to_f64().unwrap()
    );
  }

  let mut convergency = SimpleConvergency {
    eps: 1e-12,
    max_iter: 100,
  };
  let root =
    find_root_brent(low, high, f, &mut convergency).expect("failed to solve constant spread");
  T::from_f64_fast(root)
}

fn fixed_leg_spread_annuity<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Fixed(coupon) if coupon.period.payment_date >= valuation_date => {
        let tau = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
        let df = curves.discount_curve().discount_factor(tau);
        df * coupon.notional * coupon.period.accrual_factor
      }
      _ => T::zero(),
    })
    .fold(T::zero(), |acc, value| acc + value)
}

fn deterministic_cashflow_amount<T: FloatExt>(cashflow: &Cashflow<T>) -> T {
  match cashflow {
    Cashflow::Fixed(coupon) => coupon.amount(),
    Cashflow::Simple(cashflow) => cashflow.amount,
    Cashflow::Floating(_) | Cashflow::Cms(_) => {
      unreachable!("deterministic bond legs must not contain stochastic coupons")
    }
  }
}
